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
    plt.scatter(embed[n_ind,0],embed[n_ind,1],s=1,c=cdict[0],label="normal",cmap="viridis")
    plt.scatter(embed[f_ind,0],embed[f_ind,1],s=3,c=cdict[1],label="fraud",cmap="viridis")
    plt.title(title)
    plt.legend(fontsize=12,markerscale=3)
    plt.xlabel("$x_0$")
    plt.ylabel("$x_1$")
    plt.show()
    

 