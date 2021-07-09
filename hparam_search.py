# Hyperparamter searching with PLAsTiCC data
# Code has so far proven too slow to get a good sampling of the data
#
## Author: Kara Ponder (SLAC)

import numpy as np

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

import os

import transformer2 as tran

logdir = 'your/scratch/'

data_dir = 'your/plasticc_training_data/'
data_extension = '_nobs50_timestep3_singleclass_noaug_train60_minmax_nospline_v02ci()_zNone_bTrue_ig(88,92,65,16,53,6).npy'

batch_size = 64
N_days = 50 + 1
Nf = 6 # number of filters
lc_length = 50 +1 #
target_vocab_size = 6 

# Read in dataset
_lc_data = np.load(data_dir + 'X_train' + data_extension)
_wgt_map = np.load(data_dir + 'X_wgtmap_train' + data_extension)

_pre_mask_map = np.ma.masked_values(_lc_data, 0)
_mask_map = np.ones(np.shape(_lc_data))
_mask_map[_pre_mask_map.mask] = 0.0

_dataset = tf.data.Dataset.from_tensor_slices((_lc_data, _mask_map, _wgt_map))

_lc_data_valid = np.load(data_dir + 'X_valid' + data_extension)
_wgt_map_valid = np.load(data_dir + 'X_wgtmap_valid' + data_extension)

_pre_mask_map_valid = np.ma.masked_values(_lc_data_valid, 0)
_mask_map_valid = np.ones(np.shape(_lc_data_valid))
_mask_map_valid[_pre_mask_map_valid.mask] = 0.0

_dataset_valid = tf.data.Dataset.from_tensor_slices((_lc_data_valid, _mask_map_valid, _wgt_map_valid))

_batch_ds = _dataset.batch(batch_size)
_batch_ds_valid = _dataset_valid.batch(batch_size)


# All of the hyperparameters I was trying to teset
HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([6, 8, 10])) # 12
HP_DFF = hp.HParam('dff', hp.Discrete([64, 512])) # 32, , 1024, 256,128, #64, 512
HP_DMODEL = hp.HParam('d_model', hp.Discrete([64, 512])) # 32, 1024, 256,128, 
HP_NUM_HEADS = hp.HParam('num_heads', hp.Discrete([4, 8])) #16, 32

HP_KLD_ALPHA = hp.HParam('kld_alpha', hp.RealInterval(0.01, 0.8))
HP_KLD_RHO = hp.HParam('kld_rho',  hp.RealInterval(0.0000000001, 0.01))

HP_DROPOUT = hp.HParam('dropout_rate', hp.RealInterval(0.0, 0.8))


#METRIC_ACCURACY = 'accuracy'
METRIC_MSE = 'mean_squared_error'
METRIC_MAE = 'mean_absolute_error'
METRIC_RMSE = 'root_mean_squared_error'

with tf.summary.create_file_writer(logdir).as_default():
    hp.hparams_config(
                      hparams=[HP_NUM_LAYERS, 
                               HP_DFF, #
                               HP_DMODEL, #
                               HP_NUM_HEADS,
                               HP_KLD_ALPHA, 
                               HP_KLD_RHO,
                               HP_DROPOUT
                              ],
                      metrics=[hp.Metric(METRIC_MSE, display_name='MSE'),
                               hp.Metric(METRIC_MAE, display_name='MAE'),
                               hp.Metric(METRIC_RMSE, display_name='RMSE'),
                              ],
                      )
    
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=10, exponential=-1.5):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        self.exponential = exponential

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(tf.multiply(self.d_model, 100000.0)) * tf.math.minimum(arg1, arg2)
    
num_batches = 0
for (batch, _) in enumerate(_batch_ds):
    num_batches = batch

num_batches_valid = 0
for (batch, _) in enumerate(_batch_ds_valid):
    num_batches_valid = batch

    
# Set up to run the fit
def generator(data_set):
    while True:
        for in_batch, mask_batch, wgt_batch in data_set: #tar_batch, 
            yield ( [in_batch , in_batch[:, :-1, :],  mask_batch[:, 1:, :], wgt_batch] , in_batch[:, 1:, :])



def train_test_model(hparams, logdir):
    num_layers = hparams[HP_NUM_LAYERS] # 8
    d_model = hparams[HP_DMODEL]
    dff = hparams[HP_DFF]
    num_heads = hparams[HP_NUM_HEADS]
    
    kld_alpha = hparams[HP_KLD_ALPHA] #0.3
    kld_rho = hparams[HP_KLD_RHO]
    
    dropout_rate = hparams[HP_DROPOUT]

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    def loss_kld(layer1, layer2, alpha=kld_alpha):
        alpha = tf.constant(alpha, dtype=tf.float32)
        layer1 = layer1[0]
        layer1 = tf.math.abs(layer1)
        layer2 = layer2[0]
        layer2 = tf.math.abs(layer2)

        def loss(y_true, y_pred):
            ones = tf.ones(layer1.shape, dtype=tf.float32)
            rhoc = tf.cast(kld_rho, dtype='float32') #kld_rho
            rho = rhoc*ones

            def kld(layer):
                kld_1 = tf.math.multiply(rhoc, tf.math.log(tf.math.divide_no_nan(rho, layer)))
                kld_2 = tf.math.multiply((1.0 - rhoc), tf.math.divide_no_nan(tf.math.log(ones-rho), tf.math.log(ones-layer)))
                return tf.reduce_sum(kld_1 + kld_2) #kld_1_without_nans + kld_2_without_nans)

            mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
            rmse = tf.math.sqrt(mse)
            return rmse + tf.multiply(alpha, (kld(layer1) + kld(layer2)))
        return loss


    # Define the model
    encoder = tran.Encoder(num_layers, d_model, num_heads, dff,
                           lc_length, dropout_rate, embed=True)
    decoder = tran.Decoder(num_layers, d_model, num_heads, dff,
                           lc_length, dropout_rate, embed=True)
    final_layer = tf.keras.layers.Dense(target_vocab_size)


    inp = tf.keras.layers.Input(shape=(None,Nf))
    target = tf.keras.layers.Input(shape=(None,Nf))
    maskmap = tf.keras.layers.Input(shape=(None,Nf))

    x = encoder(inp)
    x = decoder(target, x, mask=tran.create_decoder_masks(inp, target))
    x = final_layer(x)
    mx = tf.keras.layers.Multiply()([x, maskmap])
    model = tf.keras.models.Model(inputs=[inp, target, maskmap], outputs=mx)

    
    loss_object = loss_kld(model.get_layer(name='encoder').get_weights(),
                           model.get_layer(name='decoder').get_weights())
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real,0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    
    model.compile(optimizer=optimizer, loss=loss_function,
                  metrics=[tf.keras.metrics.MeanSquaredError(),
                           tf.keras.metrics.MeanAbsoluteError(),
                           tf.keras.metrics.RootMeanSquaredError(),
                          ]) #loss_object)

    history = model.fit(x = generator(_batch_ds),
                        validation_data = generator(_batch_ds_valid),
                        epochs=25,
                        steps_per_epoch=num_batches,
                        validation_steps = num_batches_valid,
                        callbacks=[
                                   tf.keras.callbacks.TensorBoard(logdir),  # log metrics
                                   hp.KerasCallback(logdir, hparams),  # log hparams
                                   ],
                        #verbose=0,
                        )
    
    _, mse, mae, rmse = model.evaluate(generator(_batch_ds_valid), steps=num_batches_valid)
    
    return mse, mae, rmse
                                   
def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        mse, mae, rmse = train_test_model(hparams, run_dir)
        tf.summary.scalar(METRIC_MSE, mse, step=1)
        tf.summary.scalar(METRIC_MAE, mae, step=1)
        tf.summary.scalar(METRIC_RMSE, rmse, step=1)


session_num = 0

for num_layers in HP_NUM_LAYERS.domain.values:
    for dff in HP_DFF.domain.values:
        for d_model in HP_DMODEL.domain.values:
            for num_heads in HP_NUM_HEADS.domain.values:
                for kld_alpha in np.linspace(HP_KLD_ALPHA.domain.min_value, HP_KLD_ALPHA.domain.max_value, 3):
                    for kld_rho in np.linspace(HP_KLD_RHO.domain.min_value, HP_KLD_RHO.domain.max_value, 3):
                        for dropout in np.linspace(HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value, 3):
                            tf.keras.backend.clear_session()
                            path = logdir + f'/{session_num}'
                            os.mkdir(path)
                            hparams = {
                                       HP_NUM_LAYERS: num_layers,
                                       HP_DFF : dff,
                                       HP_DMODEL: d_model,
                                       HP_NUM_HEADS: num_heads,
                                       HP_KLD_ALPHA: kld_alpha,
                                       HP_KLD_RHO: kld_rho,
                                       HP_DROPOUT: dropout,
                                  }
                            run_name = "run-%d" % session_num
                            run(path, hparams)
                            #train_test_model(hparams, path)
                            session_num += 1