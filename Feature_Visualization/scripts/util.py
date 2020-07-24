import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="ticks")
import warnings
warnings.filterwarnings("ignore")
plt.style.use('bmh')

def combine_df(train,dev,test,label):
    """
    This function splits data into train normal and test X and test y for model_results.
    
    Args:
    train, dev, test: pandas dataframe, training data 
    label: a string, column name for the label 
    
    Returns:
    data: combined dataframe
    """
    train_df = train.copy()
    dev_df = dev.copy()
    train_df[label], dev_df[label] = 0, 0 
    data = pd.concat([train_df,dev_df,test])
    return data

def create_umapDF(data, label, sample_size,seed):
    """
    This function splits data into a UMAP feature and label with more equivalent num of 
    normal and fraud. 
    
    Args:
    data: pandas dataframe
    label: a string, the label column name 
    sample_size: an integer, num of samples for normal data
    seed: a random seed for sampling without replacement 
    
    Returns: 
    X: pandas dataframe, features 
    y: pandas series, label 
    """
    normal = data[data[label] == 0].sample(sample_size,replace=False,random_state=seed)
    fraud = data[data[label]==1]
    feature_df = pd.concat([normal,fraud],axis=0)
    X, y = feature_df.drop(label,axis=1), feature_df[label]
    return X,y
    
def plot_umap(embed,label,title):
    """
    This function plots umap 2D embed.
    
    Arg:
    embed: umap embed, 2D 
    label: a pandas series, label for the fraud and normal
    title: a string, indicates the title of the plot

    plots a scatter plot of component 1 and component 2
    """
    plt.figure(figsize=(10,7), dpi=120)
    plt.scatter(embed[label== 0,0], embed[label== 0,1], s=1, c='C0', cmap='fire', label="normal")
    plt.scatter(embed[label== 1,0], embed[label== 1,1], s=3, c='C1', cmap='fire', label="fraud")
    plt.title(title)
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    plt.legend(fontsize=16, markerscale=5)
    plt.show()

 