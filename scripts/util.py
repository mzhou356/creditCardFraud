import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import make_scorer, f1_score, precision_score,recall_score,confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import tensorflow_probability as tfp
import tensorflow.compat.v2 as tf

tfk = tf.keras
tfkl=tf.keras.layers
tfpl= tfp.layers         # layers for tensor flow probability 
tfd = tfp.distributions

# define custom scores 
# multiple y_true and y_pred with -1 in confusion matrix to create the correct orientation for confusion matrix 
def make_custom_score(flip = True, AddCustomScore = False, **kwargs):
    """
    This function makes custom score for sklearn gridSearchCV and other evaluators 
    
    args: 
    flip: boolean, if flip y_true andy_pred with -1 to rotate the confusion matrix 
    AddCustomScore: boolean, if want to add additional custom score 
    *kwargs: flexible keyword argument, a dictionary with name as score name and value as make_scorer function 
    
    returns: 
    custom score as a dictionary
    """
    if flip:
        def tp(y_true,y_pred):return confusion_matrix(-y_true,-y_pred)[1,1]
        def fp(y_true,y_pred):return confusion_matrix(-y_true,-y_pred)[0,1]
        def fn(y_true,y_pred):return confusion_matrix(-y_true,-y_pred)[1,0]
    else:
        def tp(y_true,y_pred):return confusion_matrix(y_true,y_pred)[1,1]
        def fp(y_true,y_pred):return confusion_matrix(y_true,y_pred)[0,1]
        def fn(y_true,y_pred):return confusion_matrix(y_true,y_pred)[1,0]
    def f1_f(y_true,y_pred):return f1_score(y_true,y_pred, pos_label=-1)
    def prec_f(y_true,y_pred):return precision_score(y_true,y_pred, pos_label=-1)
    def recall_f(y_true,y_pred):return recall_score(y_true,y_pred, pos_label=-1)
    scoring = {'tp':make_scorer(tp),'fp':make_scorer(fp),
           'fn':make_scorer(fn), 'f1_f':make_scorer(f1_f),
           'prec_f':make_scorer(prec_f),'recall_f':make_scorer(recall_f)}
    if AddCustomScore:
        for scr_name,scr_func in kwargs.items():
            scoring[scr_name]=scr_func
    return scoring

def makeCustomSplits(training, label, n,seed,ratio):
    """
    This function makes custom train test splits for gridsearch or randomsearchCV for sklearn 
    
    args: 
    training: pandas df, contains training data 
    label: a string, colname for the Class label 
    n: an integer, how many cross fold splits 
    seed: an integer for random splitting 
    ratio: anomaly class prevalence level 
    
    returns: 
    generator that returns custom (train,test) for CV search in sklearn
    X_train as numpy array 
    y_train as numpy array
    """
    normX,fraudX = training[training[label]==0].drop(label,axis=1).values,\
                   training[training[label]==1].drop(label,axis=1).values
    normXSplits = KFold(n_splits=n,shuffle=True,random_state=seed).split(normX)
    X_train = np.concatenate([normX,fraudX],axis=0)
    nX,fX = normX.shape[0],fraudX.shape[0]
    y_train = np.concatenate([np.ones(nX),np.ones(fX)*-1],axis=0)
    # set similar prevalence for testing 
    fTest_ind = np.random.choice(np.arange(nX,fX+nX), size = int((nX//n+fX)*ratio), replace=False)
    cvSplits = ((train,np.concatenate([test,fTest_ind],axis=0)) for train, test in normXSplits)
    return cvSplits,X_train,y_train

def CVResultsOutput(CVresults, scorenames):
    """
    This function generates custom all evaluation outputs for all parameter combinations
    
    args:
    CVresults: gridsearch or randomsearch results, a dictionary format 
    scorenames: a list of evaluation metrics scorenames. 
    
    returns:
    pandas df with all hyperparameter and evaluation metrics.
    """
    df_output= pd.DataFrame(CVresults["params"])
    for name in scorenames:
        df_output[name]=CVresults[f"mean_test_{name}"]
    return df_output



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


def plot_relationship(norm_data = None, fraud_data = None, df=None, label=None, feature_name=None):
    """
    This function plots the relationship between features and the 2 classes.
    
    Args:
    norm_data: a list or an array or pandas series, data for normal returns 
    fraud_data: a list or an array or pandas series, data for fraud returns 
    df: pandas dataframe, the dataframe that contains the dataset. 
    label: a type string, colname of the label 
    feature_name: a type string, colname of the dataset.
    
    Returns:
    A plot that shows 2 distribution maps, red is for fraud and green is for non fraud 
    """
    # make sure feature_name is type string 
    if norm_data and fraud_data:
        plt.hist(fraud_data, color = "red", label = "fraud",alpha=1, log=True)
        plt.hist(norm_data, color = "green", label = "not fraud",alpha = 0.5,log=True)
    else:  
        plt.hist(df[df[label]==1][feature_name], color = "red", label = "fraud",alpha=1,log=True)
        plt.hist(df[df[label]==0][feature_name], color = "green", label = "not fraud",alpha = 0.3, log=True)
    plt.ylabel("count (log)")
    plt.legend(loc="best")
    plt.xlabel(f"{feature_name}")
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
    
    
def train_test_dfs(train,dev,test,label,test_size,seed):
    """
    This function splits data into train normal and test X and test y for model_results.
    
    Args:
    train, dev, test: pandas dataframe, training data 
    label: a string, column name for the label 
    test_size: a float, ratio of data for test 
    seed: an integer, random seed for splitting 
    
    Returns:
    training: a pandas df for training
    norm_data: a pandas df for normal returns only for training 
    test_data: a pandas df for testing features
    test_y: a pandas Series, label for test data 
    """
    train[label], dev[label] = 0, 0 
    data = pd.concat([train,dev,test])
    training, testing = train_test_split(data, test_size = test_size, random_state=seed)
    test_data,test_y = testing.drop(label,axis=1), testing.Class
    norm_data, training_data = training[training[label]==0], training
    norm_data = norm_data.drop(label, axis=1)
    return training_data,norm_data,test_data,test_y
    
    
    
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