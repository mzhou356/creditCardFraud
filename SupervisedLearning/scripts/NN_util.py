import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
from sklearn.model_selection import train_test_split

tfk = tf.keras
tfkl=tf.keras.layers

def set_gpu_limit(n):
    """
    This function sets the max num of GPU in G to minimize overuse GPU per session.
    
    args:
    n: a float, num of GPU in G.
    """
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*n)]) 
    
def make_tensor_dataset(df,label_name,batch_size,buffer_size,test_size,seed):
    """
    This function generates tensorflow train and test dataset for NN.
    
    args:
    df: a pandas dataframes, used for training and testing.
    label_name: String, name for the label column
    buffer_size: an integer for each shuffle 
    batch_size: training batch size for each shuffle 
    test_size: a float, percentage of the df for testing during fitting if needsplit is True
    seed: an integer for random shuffling during train test split if needsplit is True 
    
    returns:
    tensforflow dataset: train and test for modeling fitting/tuning 
    """
    train, test = train_test_split(df,test_size=test_size,random_state=seed)
    train_set, train_label = train.drop(label_name,axis=1).values, train[label_name].values
    dev_set, dev_label = test.drop(label_name,axis=1).values, test[label_name].values
    train_data = tf.data.Dataset.from_tensor_slices((train_set,
                 train_label)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(buffer_size)
    dev_data = tf.data.Dataset.from_tensor_slices((dev_set,
                 dev_label)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).shuffle(buffer_size)
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

def make_model(sizes,input_size, metrics, l1 = 10e-5, lr = 1e-4):
    """
    This function creates tensorflow binary classifier and use l1 to regularize the 
    activate output. 
    
    Args:
    sizes: a list of integer indicating num of hidden nodes
    input_size: num of input features, an integer
    l1:option input for activation regularizer to prevent overfitting of training data 
    lr: for optimizer adam, gradient descent step size 
    metrics: metrics for optimizing the model during tuning 
    """
    model = tfk.Sequential([
            tfkl.InputLayer(input_shape = input_size),
            dense_layers(sizes,l1),
            tfkl.Dense(units=1,activation="sigmoid")
            ], name = "NN_binary_classifier")
    model.compile(optimizer = tf.optimizers.Adam(learning_rate=lr), 
                  loss = "binary_crossentropy",
                  metrics = metrics)
    return model
            

def plot_loss(history):
    """
    This function plots the loss value for train and test 
    
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
    
def plot_metrics(history):
    """
    This function plots the metrics for train and test 
    
    Arg:
    history: a tensorflow.python.keras.callbacks.History object 
    
    Returns:
    A plot that shows 2 overlapping loss versus epoch images. red is for test and blue is for train 
    
    """
    result = history.history.copy()
    del result["loss"]
    del result["val_loss"]
    pd.DataFrame(result).plot()
    