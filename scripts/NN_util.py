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

def make_tensor_dataset(dfs,buffer_size,batch_size,test_size=None,seed=None, needsplit = False):
    """
    This function generates tensorflow train and test dataset for NN.
    
    args:
    dfs: a list of pandas dataframes, used for training and testing.
    buffer_size: an integer for each shuffle 
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
                 train_set)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(buffer_size)
    dev_data = tf.data.Dataset.from_tensor_slices((dev_set,
                 dev_set)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(buffer_size)
    return train_data, dev_data 

def dense_layers(sizes):
    """
    This function creates tfk sequantial layers to customize the size of layers 
    
    args:
    sizes: a list of integer indicator hidden layer node number
    
    returns:
    tensforflow dense layers sequential layers
    """
    layers = tfk.Sequential([tfkl.Dense(s,activation=tf.nn_leaky_relu,
                                        activity_regularizer=tfk.regularizers.l1(10e-5))
                             for s in sizes])
    return layers


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
    

def reconstruction_log_prob(x_input,encoder, decoder,sampling_size=100):
    """
    This function generates log_prob score for each input sample with num of sampling_size per
    input data. 
    
    Arg:
    x_input: numpy array, input features (30 dimension)
    encoder: deep learning model 
    decoder: deep learning model 
    sampling_size: an integer, default is 100. 
    
    Returns:
    Average prob score for log_prob of each input. 
    """
    Z = encoder(x_input)
    encoder_samples = Z.sample(sampling_size)  # generate 30 outputs from encoder per input 
    return np.mean(decoder(encoder_samples).log_prob(x_input), axis=0)

# def neg_log_prob(neg_log,label):
#     """
#     This function plots the neg_log prob for both fraud and normal 
#     classes.
    
#     Args:
#     neg_log: - reconstruction_log_prob (this is the term we will use as pred prob score)
#     label: y label for test set 
    
#     Returns: A plot that shows the negative log prob of fraud versus normal 
#     transactions. 
#     """
#     plt.hist(neg_log[label==1],label="fraud",color="red",alpha = 0.5, log=True)
#     plt.hist(neg_log[label==0],label="normal",color="blue",alpha=0.5, log=True)
#     plt.title("reconstruction log prob")
#     plt.ylabel("frequence log")
#     plt.xlabel("logp(x_input|x_output)")
#     plt.show()    