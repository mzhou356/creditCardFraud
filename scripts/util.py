import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import tensorflow_probability as tfp
import tensorflow.compat.v2 as tf

tfk = tf.keras
tfkl=tf.keras.layers
tfpl= tfp.layers         # layers for tensor flow probability 
tfd = tfp.distributions

def create_model(input_size,encoded_size):
    """
    This function returns encoder, decoder, and VAE for variational autoencoder.
    
    Args:
    input_size: integer, number of input features. 
    encoded_size: dimension for the latent representation (Z).
    
    Returns:
    encoder, decoder, and VAE models 
    """
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                        reinterpreted_batch_ndims=1)
    encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape = input_size),  # 30 input features 
    tfkl.Dense(units=20, activation=tf.nn.leaky_relu, activity_regularizer=tfk.regularizers.l1(10e-5)),
    tfkl.Dense(units=10, activation=tf.nn.leaky_relu, activity_regularizer=tfk.regularizers.l1(10e-5)),
    tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),activation=None),
    tfpl.MultivariateNormalTriL(encoded_size,activity_regularizer=tfpl.KLDivergenceRegularizer(prior))
    ], name="encoder")
    decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[encoded_size]),
    tfkl.Dense(units=10, activation=tf.nn.leaky_relu, activity_regularizer=tfk.regularizers.l1(10e-5)),
    tfkl.Dense(units=20, activation=tf.nn.leaky_relu, activity_regularizer=tfk.regularizers.l1(10e-5)),
    tfkl.Dense(tfpl.IndependentNormal.params_size(input_size),activation=None),
    tfpl.IndependentNormal(input_size)
    ], name = "decoder")
    VAE = tfk.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs[0]),name="VAE")
    negloglik = lambda x_input, x_output: -x_output.log_prob(x_input)
    VAE.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), loss=negloglik)
    return encoder,decoder,VAE


def plot_relationship(df, label, feature_name):
    """
    This function plots the relationship between features and the 2 classes.
    
    Args:
    df: pandas dataframe, the dataframe that contains the dataset. 
    label: a type string, colname of the label 
    feature_name: a type string, colname of the dataset.
    
    Returns:
    A plot that shows 2 distribution maps, red is for fraud and green is for non fraud 
    """
    # make sure feature_name is type string 
    assert type(feature_name) == str and type(label)==str, "label and feature_name need to be type str"
    plt.hist(df[df[label]==1][feature_name], color = "red", label = "fraud",alpha=1,log=True)
    plt.hist(df[df[label]==0][feature_name], color = "green", label = "not fraud",alpha = 0.3, log=True)
    plt.legend(loc="best")
    plt.xlabel(f"{feature_name}")
    plt.ylabel("count (log)")
    plt.show()
    
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

def neg_log_prob(neg_log,label):
    """
    This function plots the neg_log prob for both fraud and normal 
    classes.
    
    Args:
    neg_log: - reconstruction_log_prob (this is the term we will use as pred prob score)
    label: y label for test set 
    
    Returns: A plot that shows the negative log prob of fraud versus normal 
    transactions. 
    """
    plt.hist(neg_log[label==1],label="fraud",color="red",alpha = 0.5, log=True)
    plt.hist(neg_log[label==0],label="normal",color="blue",alpha=0.5, log=True)
    plt.title("reconstruction log prob")
    plt.ylabel("frequence log")
    plt.xlabel("logp(x_input|x_output)")
    plt.show()
    
    
def plot_roc(label, neg_log):
    """
    This function plots the receiver operator curve. 
    
    Args:
    label:y label for test set 
    neg_log: - reconstruction_log_prob (this is the term we will use as pred prob score)
    
    Returns: ROC curve with area under the curve shown as a label. 
    """
    fpr, tpr, threshold = roc_curve(label,neg_log)
    auc = roc_auc_score(label,neg_log)
    plt.plot(fpr,tpr,label=f"auc={auc}")
    plt.legend(loc="best")
    plt.show()
    
def model_results(label,neg_log,threshold):
    """
    This function returns confusion matrix and classification report for holdout set. 
    
    Args:
    label:y label for test set 
    neg_log: - reconstruction_log_prob (this is the term we will use as pred prob score)
    threshold: a float. The cut off neg log prob score for classification. 
    
    Returns: confusion matrix and classification report for the specified neg log threshold.
    """
    results = pd.DataFrame({"label":label,"neg_log_prob":neg_log})
    results["pred_class"]=results.neg_log_prob.apply(lambda x: 1 if x>threshold else 0)
    print(confusion_matrix(results.label,results.pred_class))
    print(classification_report(results.label, results.pred_class))
    
    
# def log_scale_comparision(df,label,feature_name,show_original=False):
#     """
#     This function plots the feature space distribution between np.log scale and no log scale 
    
#     Args:
#     df: pandas dataframe, the dataframe that contains the dataset. 
#     label: a type string, colname of the label 
#     feature_name: a type string, colname of the dataset.
    
#     Returns:
#     A plot that shows 2 distribution maps, blue is for log and orange is not for log 
#     """
#     # make sure feature_name is type string 
#     assert type(feature_name) == str and type(label)==str, "label and feature_name need to be type str"
#     plt.hist(np.log(df[df[label]==0][feature_name]+0.00000000001), label='logscale',color = "blue",alpha = 1,log=show_original)
#     if show_original:
#         plt.hist(df[df[label]==0][feature_name],label="noscale",color='orange',alpha = 0.3,log=True)
#     plt.legend()
#     plt.title(feature_name)
#     plt.show()