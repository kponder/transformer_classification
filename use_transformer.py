# Uses the Transformer for the testing data in repo. 
# Arguments are the hyperparameters
## Author: Kara Ponder (SLAC)

# This one has the choice of being a TRUE AE or using an autoencoder where the target is the true LC since this is test data.


import tensorflow as tf

import time
import argparse
import numpy as np

import transformer as tran

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
parser.add_argument('--approx_ae', destination='true_ae', action='store_false',
                    default=True,
                    help='Create a _approximate_ AE (LC->Model) instead of LC->LC')
args = parser.parse_args()

# Set parameters
d_model = args.d_model # input vector must have length d_model
target_vocab_size = 6  # possible results to choose from

lc_length = 100 +1 # light curve length
input_vocab_size = lc_length

## hyperparameters:
num_layers = args.num_layers
dropout_rate = args.dropout
dff = args.dff # hidden layer size of the feed forward network, needs to be larger than 24, factor of 2^x
num_heads = args.num_heads # d_model % num_heads == 0

# LC stuff
N = 10000 # number of objects
N_days = 100 + 1
Nf = 6 # number of filters
num_classes = 4


batch_size = args.batch
EPOCHS = args.epochs

if args.embed:
    assert d_model > target_vocab_size, 'If embedding, d_model > 6'
    
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


# Read in dataset
lc_data = np.load('lc_data.npy')
real_lc_data = np.load('real_lc.npy')

if args.wgt_map:
    mask_map = np.load('weightmap.npy')
    wgt_map = mask_map * 1/0.1**2
    wgt_map[np.where(wgt_map == 0)] = 1/2**2
    dataset = tf.data.Dataset.from_tensor_slices((lc_data, real_lc_data, mask_map, wgt_map))
else:
    dataset = tf.data.Dataset.from_tensor_slices((lc_data, real_lc_data))

batch_ds = dataset.batch(batch_size)


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

if args.wgt_map:
    mx = tf.keras.layers.Multiply()([x, maskmap])
    model = tf.keras.models.Model(inputs=[inp, target, maskmap], outputs=mx)
else:
    model = tf.keras.models.Model(inputs=[inp, target], outputs=x)

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
train_loss = tf.keras.metrics.Mean(name='train_loss')

# Compile and run the model
model.compile(optimizer=optimizer, loss=loss_object)

num_batches = 0
for (batch, _) in enumerate(batch_ds):
    num_batches = batch

    
# Set up to run the fit
def generator(data_set):
    while True:
        if args.wgt_map: 
            for in_batch, tar_batch, mask_batch, wgt_batch in data_set:
                if args.true_ae:
                    yield ( [in_batch , in_batch[:, :-1, :],  mask_batch[:, 1:, :], wgt_batch] , in_batch[:, 1:, :])
                else:
                    yield ( [in_batch , tar_batch[:, :-1, :],  mask_batch[:, 1:, :], wgt_batch] , tar_batch[:, 1:, :])
        else:
            for in_batch, tar_batch in data_set:
                yield ( [in_batch , tar_batch[:, :-1, :]] , tar_batch[:, 1:, :])
                

history = model.fit(x = generator(batch_ds),
                    #validation_data = generator(val_dataset),
                    epochs=EPOCHS,
                    steps_per_epoch=num_batches,
                    #validation_steps = val_batches,
                    verbose=0,
                    )


model.save_weights('transformer2_weights_%s_%sdmodel_%slayer_%sdff_%sheads_%sdropout_%sstep%s.h5'%(args.loss, args.d_model, args.num_layers, args.dff, args.num_heads, args.dropout, args.step_size, args.extension))


np.savetxt('loss_%s_%sdmodel_%slayer_%sdff_%sheads_%sdropout_%sstep%s.txt'%(args.loss,  args.d_model, args.num_layers, args.dff, args.num_heads, args.dropout, args.step_size, args.extension), history.history['loss'])