import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import tensorflow_probability as tfp
import tensorflow.compat.v2 as tf
from sklearn.model_selection import train_test_split

tfk = tf.keras
tfkl=tf.keras.layers
tfpl= tfp.layers         # layers for tensor flow probability 
tfd = tfp.distributions

def set_gpu_limit(n):
    """
    This function sets the max num of GPU in G to minimize overuse GPU per session.
    
    args:
    n: a float, num of GPU in G.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*n)]) 

def make_tensor_dataset(dfs,batch_size,test_size=None,seed=None, needsplit = False):
    """
    This function generates tensorflow train and test dataset for NN.
    
    args:
    dfs: a list of pandas dataframes, used for training and testing.
    batch_size: training batch size for each shuffle 
    test_size: a float, percentage of the df for testing during fitting if needsplit is True
    seed: an integer for random shuffling during train test split if needsplit is True 
    
    returns:
    tensforflow dataset: train and test for modeling fitting/tuning 
    """
    if needsplit:
        train, test = train_test_split(dfs[0],test_size=test_size,random_state=seed)
    else:
        train, test = dfs
    train_set, dev_set = train.values, test.values 
    train_data = tf.data.Dataset.from_tensor_slices((train_set,
                 train_set)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(train_set.shape[0])
    dev_data = tf.data.Dataset.from_tensor_slices((dev_set,
                 dev_set)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(dev_set.shape[0])
    return train_data, dev_data 

def dense_layers(sizes,l1 = 10e-5):
    """
    This function creates tfk sequantial layers to customize the size of layers 
    
    args:
    sizes: a list of integer indicator hidden layer node number
    l1: optional term for custum set l1 regularizer for preventing overfitting 
    
    returns:
    tensforflow dense layers sequential layers
    """
    layers = tfk.Sequential([tfkl.Dense(s,activation=tf.nn.leaky_relu,
                                        activity_regularizer=tfk.regularizers.l1(l1))
                             for s in sizes])
    return layers

def make_autoencoder(sizes,input_size, l1 = 10e-5, ifvae = False, encoded_size = None,prior=None):
    if ifvae and encoded_size and prior:
        encoder = tfk.Sequential([
                  tfkl.InputLayer(input_shape = input_size), 
                  dense_layers(sizes,l1),
                  tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),activation=None),
                  tfpl.MultivariateNormalTriL(encoded_size,activity_regularizer=tfpl.KLDivergenceRegularizer(prior))
                  ],name = "encoder")
        decoder = tfk.Sequential([
                  tfkl.InputLayer(input_shape=[encoded_size]),
                  dense_layers(reversed(sizes),l1),
                  tfkl.Dense(tfpl.IndependentNormal.params_size(input_size),activation=None),
                  tfpl.IndependentNormal(input_size)
                  ], name = "decoder")
        model = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]),name="VAE")
        return model, encoder, decoder 
    else:
        encoder = tfk.Sequential([
                  tfkl.InputLayer(input_shape = input_size), 
                  dense_layers(sizes,l1)], name = "encoder")
        decoder = tfk.Sequential(
                  [tfkl.InputLayer(input_shape=sizes[-1]),
                  dense_layers(list(reversed(sizes))[1:],l1),
                  tfkl.Dense(units=input_size, activation=None)], name = "decoder")
        model = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs),name="encoder")
        return model
            

def plot_loss(history):
    """
    This function plots the mse value for train and test autoencoder model 
    
    Arg:
    history: a tensorflow.python.keras.callbacks.History object 
    
    Returns:
    A plot that shows 2 overlapping loss versus epoch images. red is for test and blue is for train 
    
    """
    history = history.history
    plt.plot(history["loss"], color="blue", label="train")
    plt.plot(history["val_loss"], color="red", label="test")
    plt.xlabel("epoch")
    plt.ylabel("mse")
    plt.title("model training results")
    plt.legend(loc="best")
    plt.show()   
    

def reconstruction_log_prob_sampled(X_test,y_test,X_input_sample_size,encoder, decoder,sampling_size=100):
    """
    This function generates log_prob score for each input sample with num of sampling_size per
    input data. 
    
    Arg:
    x_test: pandas dataframe input features
    y_test: pandas Series for label
    X_input_sample_size: an integer, how many datapoints to sample
    encoder: deep learning model 
    decoder: deep learning model 
    sampling_size: an integer, default is 100. 
    
    Returns:
    Average prob score for log_prob of each input. 
    """
    ind = X_test.index
    X_input_ind = np.random.choice(ind,replace=False, size =X_input_sample_size)
    X_input,y_input = X_test.loc[X_input_ind].values, y_test.loc[X_input_ind]
    Z = encoder(X_input)
    encoder_samples = Z.sample(sampling_size)  # generate 30 outputs from encoder per input 
    mean_score = np.mean(decoder(encoder_samples).log_prob(X_input), axis=0)
    return mean_score, y_input

def reconstruction_log_prob(x_input,stepsize,encoder, decoder,sampling_size=100):
    """
    This function generates log_prob score for each input sample with num of sampling_size per
    input data (for all samples incremental ways to allow more sampling size)
    
    Arg:
    x_input: numpy array, input features (30 dimension),
    step_size: incremental size for computing all datapoints 
    encoder: deep learning model 
    decoder: deep learning model 
    sampling_size: an integer, default is 100. 
    
    Returns:
    Average prob score for log_prob of each input. 
    """
    scores = []
    for i in range(0,len(x_input)//stepsize):
        x = x_input[i*stepsize:(i+1)*stepsize]
        Z = encoder(x)
        encoder_samples = Z.sample(sampling_size)  # generate 30 outputs from encoder per input 
        scores.extend(np.mean(decoder(encoder_samples).log_prob(x), axis=0))
    if (i+1)*stepsize < len(x_input):
        x = x_input[(i+1)*stepsize:]
        Z = encoder(x)
        encoder_samples = Z.sample(sampling_size)  # generate 30 outputs from encoder per input 
        scores.extend(np.mean(decoder(encoder_samples).log_prob(x), axis=0))
    return np.array(scores)