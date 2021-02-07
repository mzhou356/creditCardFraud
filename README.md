# Credit Card Fraud
  - Testing various methods for imbalanced binary classification

### Data Set:  
* Dataset is from [kaggle]( https://www.kaggle.com/mlg-ulb/creditcardfraud?select=creditcard.csv)

### Data Process:
 - Simple data exploratory analyses 
 - Transformed time and amount feature columns
* Notebook, [Link](https://github.com/mzhou356/creditCardFraud/blob/master/processedData/notebook/RawDataExplorationAndDataProcessing.ipynb
)

### UMAP feature visualization:
  - Use UMAP to better understand the 30 features. 
* Notebook, [Link](https://github.com/mzhou356/creditCardFraud/blob/master/Feature_Visualization/notebooks/UMAP_visualization.ipynb
)

### Methods for unsupervised anomaly detection:  
* Autoencoder, [notebook link](https://github.com/mzhou356/creditCardFraud/blob/master/AnomalyDetection/notebooks/AutoEncoder.ipynb)
* Variational Autoencoder with Variational probability, [notebook link](https://github.com/mzhou356/creditCardFraud/blob/master/AnomalyDetection/notebooks/VariationalAutoEncoder.ipynb)
* Isolation forest, [notebook link](https://github.com/mzhou356/creditCardFraud/blob/master/AnomalyDetection/notebooks/IsolationForestforFraudDetection.ipynb)
* OneClass SVM, [notebook link](https://github.com/mzhou356/creditCardFraud/blob/master/AnomalyDetection/notebooks/oneClassSVM_anomalyDetection.ipynb)

### Methods for supervised machine learning without sampling strategy:
* Xgboost, [notebook link](https://github.com/mzhou356/creditCardFraud/blob/master/SupervisedLearning/notebooks/Xgboost_binary_classifier.ipynb
)
* Simple NN, [notebook link](https://github.com/mzhou356/creditCardFraud/blob/master/SupervisedLearning/notebooks/Supervised_NN_Binary_Classifier.ipynb)
* Logistic Regression using UMAP features, [notebook link](https://github.com/mzhou356/creditCardFraud/blob/master/SupervisedLearning/notebooks/Logistic_Regression_Classifier_UMAP_Transformation.ipynb)

### Summary:
  * Supervised machine learning methods performed better than unsupesrvised anomaly detection methods. 
  * Among supervised machine learning methods, supervised NN performed the best but XGBoost is very close. More data would improve NN performance.
