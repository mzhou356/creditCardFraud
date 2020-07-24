import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
from sklearn.metrics import make_scorer, f1_score, precision_score,recall_score,precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV
plt.style.use('bmh')


def make_custom_score(AddCustomScore = False, **kwargs):
    """
    This function makes custom score for sklearn gridSearchCV and other evaluators 
    
    args: 
    AddCustomScore: boolean, if want to add additional custom score 
    *kwargs: flexible keyword argument, a dictionary with name as score name and value as make_scorer function 
    
    returns: 
    custom score as a dictionary
    """
    def tp(y_true,y_pred):return confusion_matrix(y_true,y_pred)[1,1]
    def fp(y_true,y_pred):return confusion_matrix(y_true,y_pred)[0,1]
    def fn(y_true,y_pred):return confusion_matrix(y_true,y_pred)[1,0]
    def f1_f(y_true,y_pred):return f1_score(y_true,y_pred, pos_label=1)
    def prec_f(y_true,y_pred):return precision_score(y_true,y_pred, pos_label=1)
    def recall_f(y_true,y_pred):return recall_score(y_true,y_pred, pos_label=1)
    scoring = {'tp':make_scorer(tp),'fp':make_scorer(fp),
           'fn':make_scorer(fn), 'f1_f':make_scorer(f1_f),
           'prec_f':make_scorer(prec_f),'recall_f':make_scorer(recall_f)}
    if AddCustomScore:
        for scr_name,scr_func in kwargs.items():
            scoring[scr_name]=scr_func
    return scoring

def plot_pr_re(label, score):
    """
    This function plots the receiver operator curve. 
    
    Args:
    label:y label for test set 
    score: prob_score
    
    Returns: precsion recall curve
    """
    plt.figure(figsize=(6,6))
    precision, recall, threshold = precision_recall_curve(label,score)
    plt.plot(recall,precision)
    plt.ylabel("precision")
    plt.xlabel("recall")
    plt.show()

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
    
def plot_umap(embed,title,target):
    """
    This function plots umap 2D embed.
    
    embed: umap embed, 2D 
    title: title for the graph 
    target: the classification label 
    
    plots a scatter plot of component 1 and component 2
    """
    plt.figure(figsize=(12,7),dpi=150)
    cdict = {0:"yellow",1:"purple"}
    n_ind = np.where(target==0)
    f_ind = np.where(target==1)
    plt.scatter(embed.T[0,n_ind],embed.T[1,n_ind],s=1,c=cdict[0],label="normal",cmap="viridis")
    plt.scatter(embed.T[0,f_ind],embed.T[1,f_ind],s=3,c=cdict[1],label="fraud",cmap="viridis")
    plt.title(title)
    plt.legend(fontsize=12,markerscale=3)
    plt.xlabel("$x_0$")
    plt.ylabel("$x_1$")
    plt.show()
    
def customFeatureElimination(estimator,train_X,train_y,test_X,test_y,n,max_iter,delta,verbose = True):
    """
    This function uses np.random to randomly remove n num of features until the score 
    no longer improve by a threshold. 
    
    estimator: xgboost model 
    train_X: training features, a pandas dataframe. 
    train_y: label for train, pandas Series or array
    test_X: test features, a pandas dataframe
    test_y: label for test, pandas Series or array
    n: num of features to keep 
    max_iter: maximum num of trials 
    verbose: show progress 
    delta: early stop if not improve after delta iterations 
    
    returns best metric as a float and a set of features as a list 
    """
    best_metric = 0.0;  # use precision recall score 
    features = [];
    best_iter = max_iter
    while max_iter > 0:
        max_iter -=1
        selected = train_X.sample(n=n,axis=1,replace=False)
        current_features = selected.columns 
        model = estimator.fit(selected, train_y, eval_metric="aucpr")
        test = test_X[current_features]
        current_metric = precision_score(test_y,model.predict(test))
        if current_metric > best_metric:
            features = current_features
            best_metric = current_metric 
            best_iter = max_iter
        if verbose:
            print(current_metric)
            print(best_metric)
            print(best_iter)
        if (best_iter-max_iter)>delta:
            print(f"early stop after {delta} steps!")
            break
    return best_metric, features

def FeatureSearch(estimator,train_X, train_y,test_X,test_y, max_n,min_n, step_n,max_iter, delta, v1=False,v2=False):
    """
    This function uses customFeatureElimination to search a set of features from max_n,min_n with stepsize of step_n.
    
    Args:
    estimator: xgboost model 
    train_X: training features, a pandas dataframe. 
    train_y: label for train, pandas Series or array
    test_X: test features, a pandas dataframe
    test_y: label for test, pandas Series or array
    max_n: max num of features to keep
    min_n: min num of features to keep 
    step_n: what stepsize to go from max_n to min_n
    max_iter: maximum num of trials 
    v1: show progress for n count 
    v2: show progress for customFeatureElimination
    delta: early stop if not improve after delta iterations 
    
    Returns:
    a pandas dataframe final result order by metrics column
    """
    metrics = []
    features = []
    num_features =np.arange(max_n,min_n+step_n,step_n)
    for n in num_features:
        if v1:
            print(f"Searching for {n} set of features...")
        best_metric, best_features = customFeatureElimination(estimator,train_X,train_y,
                                                    test_X,test_y,
                                                    n,max_iter=max_iter,delta=delta,verbose=v2)
        metrics.append(best_metric)
        features.append(best_features)
    final_result = pd.DataFrame({"n_features":num_features,
              "metrics":metrics,
              "features":features})
    return final_result.sort_values("metrics",ascending=False)
    
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
    train_df = train.copy()
    dev_df = dev.copy()
    train_df[label], dev_df[label] = 0, 0 
    data = pd.concat([train_df,dev_df,test])
    training, testing = train_test_split(data, test_size = test_size, random_state=seed)
    return training, testing

def plot_importance(importance_type, booster):
    """
    This function plots feature importance by importance type using xgboost. 
    
    Args:
    importance_type: a string, "weight","gain","cover", see the xgboost documentation for more info. 
    booster: trained xgboost model. 
    
    plots the importance horizontal barplot 
    """
    plt.rcParams["figure.figsize"]=(12,8)
    xgboost.plot_importance(booster, importance_type=importance_type)
    plt.show()