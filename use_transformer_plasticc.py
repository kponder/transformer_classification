# Uses the Transformer for the PLAsTiCC data. 
# Arguments are the hyperparameters
## Author: Kara Ponder (SLAC)

import tensorflow as tf

import time
import argparse
import numpy as np

import transformer as tran

start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--loss', default='MSE', type=str,
                    help='Specify which metric to use')
parser.add_argument('--embed', action='store_true',
                    default=False,
                    help='Embed the 6 filters into d_model number')
parser.add_argument('--num_layers', default=8, type=int,
                    help='Number of Layers')
parser.add_argument('--d_model', default=6, type=int,
                    help='')
parser.add_argument('--dff', default=64, type=int,
                    help='Hidden Layer Size of FFN')
parser.add_argument('--num_heads', default=6, type=int,
                    help='Number of attention heads')
parser.add_argument('--batch', default=64, type=int,
                    help='Batch Size')
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of epochs')
parser.add_argument('--step_size', default=0.00001, type=float,
                    help='Adam step size')
parser.add_argument('--use_adjustable', action='store_true',
                    default=False,
                    help='Use changing learning rate.')
parser.add_argument('--beta_1', default=0.9, type=float,
                    help='Adam beta_1')
parser.add_argument('--beta_2', default=0.98, type=float,
                    help='Adam beta_2')
parser.add_argument('--epsilon', default=1e-9, type=float,
                    help='Adam epsilon')
parser.add_argument('--warmup_learning', default=10, type=int,
                    help='Warm up iterations for adjustable learning')
parser.add_argument('--exponential_learning', default=-1.5, type=float,
                    help='Exponential decay for adjustable learning')
parser.add_argument('--kld_alpha', default=0.3, type=float,
                    help='KLD Regularization Parameter')
parser.add_argument('--kld_rho', default=0.00001, type=float,
                    help='KLD Rho Parameter (close to zero)')
parser.add_argument('--dropout', default=0.0, type=float,
                    help='Dropout rate')
parser.add_argument('--extension', default='', type=str,
                    help='Extension to file name')
parser.add_argument('--wgt_map', action='store_true',
                    default=False,
                    help='Weight the loss function')
parser.add_argument('--data_dir', default='./', type=str,
                    help='Location of data')
parser.add_argument('--data_extension', default='_2ci()_zNone_bTrue_ig().npy', type=str,
                    help='File extension for data')
parser.add_argument('--output_dir', default='./', type=str,
                    help='Location for output data')
args = parser.parse_args()

# Set parameters
d_model = args.d_model # input vector must have length d_model
target_vocab_size = 6  # possible results to choose from

lc_length = 50 +1 # light curve length
input_vocab_size = lc_length

## hyperparameters:
num_layers = args.num_layers
dropout_rate = args.dropout
dff = args.dff # hidden layer size of the feed forward network, needs to be larger than 24, factor of 2^x
num_heads = args.num_heads # d_model % num_heads == 0

# LC stuff
N_days = 50 + 1
Nf = 6 # number of filters


batch_size = args.batch
EPOCHS = args.epochs


if args.embed:
    assert d_model > target_vocab_size, 'If embedding, d_model > 6'

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=10, exponential=-1.5):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
        self.exponential = exponential

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** self.exponential)

        return tf.math.rsqrt(tf.multiply(self.d_model, 10000.0)) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model, 
                               warmup_steps=args.warmup_learning,
                               exponential=args.exponential_learning)

if args.use_adjustable:
    optimizer = tf.keras.optimizers.Adam(learning_rate, 
                                         beta_1=args.beta_1, 
                                         beta_2=args.beta_2, 
                                         epsilon=args.epsilon)
else:
    optimizer = tf.keras.optimizers.Adam(args.step_size)
    

class RMSE(tf.keras.losses.Loss):
    def __init__(self, name="rmse"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        return tf.math.sqrt(mse)

def loss_kld(layer1, layer2, alpha=args.kld_alpha):
    alpha = tf.constant(alpha, dtype=tf.float32)
    layer1 = layer1[0]
    layer1 = tf.math.abs(layer1)
    layer2 = layer2[0]
    layer2 = tf.math.abs(layer2)

    def loss(y_true, y_pred):
        ones = tf.ones(layer1.shape, dtype=tf.float32)
        rhoc = args.kld_rho
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
                       lc_length, dropout_rate, args.embed)

decoder = tran.Decoder(num_layers, d_model, num_heads, dff,
                       lc_length, dropout_rate, args.embed)
final_layer = tf.keras.layers.Dense(target_vocab_size)

if args.embed:
    inp = tf.keras.layers.Input(shape=(None,Nf))
    target = tf.keras.layers.Input(shape=(None,Nf))
    maskmap = tf.keras.layers.Input(shape=(None,Nf))
else:
    inp = tf.keras.layers.Input(shape=(None,None))
    target = tf.keras.layers.Input(shape=(None,None))
    mask = tf.keras.layers.Input(shape=(None,None))
    
x = encoder(inp)
x = decoder(target, x, mask=tran.create_decoder_masks(inp, target))
x = final_layer(x)


mx = tf.keras.layers.Multiply()([x, maskmap])
model = tf.keras.models.Model(inputs=[inp, target, maskmap], outputs=mx)

loss_dict = {'MSE': tf.keras.losses.MeanSquaredError(),
             'MSLE': tf.keras.losses.MeanSquaredLogarithmicError(),
             'Huber': tf.keras.losses.Huber(),
             'MAE': tf.keras.losses.MeanAbsoluteError(),
             'LCE': tf.keras.losses.LogCosh(),
             'RMSE': RMSE(),
             'KLD_RMSE':loss_kld(model.get_layer(name='encoder').get_weights(),
                                 model.get_layer(name='decoder').get_weights()),
             }

loss_object = loss_dict[args.loss]

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')

# Compile and run the model
model.compile(optimizer=optimizer, loss=loss_function)#loss_object)


# Read in dataset
lc_data = np.load(args.data_dir + 'X_train' + args.data_extension)
wgt_map = np.load(args.data_dir + 'X_wgtmap_train' + args.data_extension)

pre_mask_map = np.ma.masked_values(lc_data, 0)
mask_map = np.ones(np.shape(lc_data))
mask_map[pre_mask_map.mask] = 0.0
dataset = tf.data.Dataset.from_tensor_slices((lc_data, mask_map, wgt_map))

lc_data_valid = np.load(args.data_dir + 'X_valid' + args.data_extension)
wgt_map_valid = np.load(args.data_dir + 'X_wgtmap_valid' + args.data_extension)

pre_mask_map_valid = np.ma.masked_values(lc_data_valid, 0)
mask_map_valid = np.ones(np.shape(lc_data_valid))
mask_map_valid[pre_mask_map_valid.mask] = 0.0

dataset_valid = tf.data.Dataset.from_tensor_slices((lc_data_valid, mask_map_valid, wgt_map_valid))

batch_ds = dataset.batch(batch_size)
batch_ds_valid = dataset_valid.batch(batch_size)


num_batches = 0
for (batch, _) in enumerate(batch_ds):
    num_batches = batch

num_batches_valid = 0
for (batch, _) in enumerate(batch_ds_valid):
    num_batches_valid = batch
    
# Set up to run the fit
def generator(data_set):
    while True:
        for in_batch, mask_batch, wgt_batch in data_set:
            yield ( [in_batch , in_batch[:, :-1, :],  mask_batch[:, 1:, :], wgt_batch] , in_batch[:, 1:, :])   

history = model.fit(x = generator(batch_ds),
                    validation_data = generator(batch_ds_valid),
                    epochs=EPOCHS,
                    steps_per_epoch=num_batches,
                    validation_steps =num_batches_valid,
                    verbose=0,
                    )


model.save_weights(args.output_dir + f'transformer2_{args.loss}_{args.d_model}dmodel_{args.num_layers}layer_{args.dff}dff_{args.num_heads}heads_{args.dropout}dropout_{args.step_size}step_10000scale_{args.beta_1}beta1_{args.beta_2}beta2_{args.epsilon}epsilon_{args.warmup_learning}warmup_{args.exponential_learning}exponential_{args.kld_alpha}kldalpha_{args.kld_rho}kldrho_' + args.extension + '.h5')


np.savetxt(args.output_dir + f'loss_{args.loss}_{args.d_model}dmodel_{args.num_layers}layer_{args.dff}dff_{args.num_heads}heads_{args.dropout}dropout_{args.step_size}step_10000scale_{args.beta_1}beta1_{args.beta_2}beta2_{args.epsilon}epsilon_{args.warmup_learning}warmup_{args.exponential_learning}exponential_{args.kld_alpha}kldalpha_{args.kld_rho}kldrho_' + args.extension + '.txt', history.history['loss'])

np.savetxt(args.output_dir + f'val_loss_{args.loss}_{args.d_model}dmodel_{args.num_layers}layer_{args.dff}dff_{args.num_heads}heads_{args.dropout}dropout_{args.step_size}step_10000scale_{args.beta_1}beta1_{args.beta_2}beta2_{args.epsilon}epsilon_{args.warmup_learning}warmup_{args.exponential_learning}exponential_{args.kld_alpha}kldalpha_{args.kld_rho}kldrho_' + args.extension + '.txt', history.history['val_loss'])
