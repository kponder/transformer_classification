# transformer_classification
Using the Transformer architecture to classify astrophysical transients.

This repository contains:
- Jupyter notebooks
    - Transformer_testing.ipynb : Original testing notebook for transformer. Use for model development
    - AE_Classifier.ipynb : The classifier associated with the transformer. Needs to be converted to script. Takes the trained weights for the encoder part of the transformer, adds a feed-forward neural network to do final classification with labels.
    - Simple_RNN.ipynb : Simple Recurrent Neural Network based on the original RAPID code 
    - Encoder_FFN_classifier.ipynb : Trains just the Transformer Encoder simultaneously as a feed-forward network to go straight from light curves to labels. Similar to Allam et al (2021)
    - Check_Results.ipynb : Example of ways to look at the results
    - PLAsTiCC_Playground.ipynb : Explores the PLAsTiCC data and generating a version that can be saved and reduced. Testing place to check for transformer implementation updates for the PLAsTiCC data. 
- Python scripts
    - transformer.py : Defines all elements for the transformer model on light curve-like data
    - use_transformer.py : Use transformer model on test data
    - use_transformer_plasticc.py : Use transformer model on PLAsTiCC data
    - hparam_search.py : Using tensorboard to do hyperparameter searching on the command line. 
    - make_test_data.py : Used to make data below while dropping 90% of LC points. Can change this number and generate new data.
- Test data: Generated very simplistic data using the Bazin function with 4 classes.
    - lc_data.npy
    - real_lc_data.npy
    - label.npy
    - weightmap.npy
    - min_max.txt : normalization constants per light curve
    
    
Attempts to integrate this into `astrorapid` (RAPID (Muthukrishna et al (2019))) can be found in this forked repository: https://github.com/kponder/astrorapid

This codes take in lightcurves with 6 filters (LSST-like).
The typical workflow embeds the 6 filters into a 128 component vector (like a word encoding) and then pass that to the positional encoder in the Transformer. This encoding can be turned on and off with flags and the length of the vector should be treated as a hyper parameter. `embed` flag should be used to use the embedding.

The test data can be used as a "True" Autoencoder mapping data to data or it can map data to real data. We do not know the actual real data for PLAsTiCC, so the True Autoencoder is the default usage in `use_transformer*`. 

Do not have to use the errors from the lightcurves. Used the `wgt_map` flag in `use_transformer*` to _use_ the weight map. This is typically used but not yet the default.

There are different options for the loss function code. Most are tensorflow defaults but not all:
- 'MSE': tf.keras.losses.MeanSquaredError(),
- 'MSLE': tf.keras.losses.MeanSquaredLogarithmicError(),
- 'Huber': tf.keras.losses.Huber(),
- 'MAE': tf.keras.losses.MeanAbsoluteError(),
- 'LCE': tf.keras.losses.LogCosh(),
- 'RMSE': RMSE(), : RMSE is not defined as a loss, only a metric. Coded as loss here 
- 'KLD_RMSE':loss_kld(model.get_layer(name='encoder').get_weights(),
                                 model.get_layer(name='decoder').get_weights()),
                                 
The KLD RMSE applied a Kullback–Leibler divergence penatly to the RMSE loss. This is based on PELICAN from Pasquet et al (2019) where we assume that sparse lightcurves will have small weights because many weights will not be activated. We set the expected value of the weights in the encoder and decoder to be a small number and add a penatly as you move away from the small number. This generates 2 new hyperparameters: `alpha` : the percentage to weight the KLD and `rho` : the small number to compare the weights to.


The model is not very fast and needs to be optimized! 