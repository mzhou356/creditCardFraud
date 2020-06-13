import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd


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