import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve,average_precision_score
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
    
def plot_pr_re(label, score):
    """
    This function plots the precision recall curve.
    
    Args:
    label:y label for test set 
    score: prob score, for VAE:- reconstruction_log_prob (
           this is the term we will use as pred prob score)
    
    Returns: precsion recall curve
    """
    plt.figure(figsize=(6,6))
    precision, recall, threshold = precision_recall_curve(label,score)
    auc = average_precision_score(label,score)
    plt.plot(recall,precision,label=f"auc={auc}")
    plt.ylabel("precision")
    plt.xlabel("recall")
    plt.legend(loc="best")
    plt.show()

def model_results(label,prob_score,threshold=None,ifprint=False):
    """
    This function returns confusion matrix and classification report for holdout set. 
    
    Args:
    label:y label for test set 
    prob_score: - prediction_score in anomaly detection cases, a numpy aray
    threshold: a float. The cut off prob score for binary classification. 
    
    Returns: confusion matrix and classification report for the specified neg log threshold.
    """
    results = pd.DataFrame({"label":label,"anomaly_prob":prob_score})
    if ifprint and threshold:
        results["pred_class"]=results.anomaly_prob.apply(lambda x: 1 if x>=threshold else 0)
        print(confusion_matrix(results.label,results.pred_class))
        print(classification_report(results.label, results.pred_class))
    return results
    
    
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