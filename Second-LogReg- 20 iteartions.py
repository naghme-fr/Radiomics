# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 18:43:18 2022

@author: Admin
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression  
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, roc_auc_score


df = pd.read_csv("ngtdm-selected.csv")
print(df.head())

df.label[df.label == 'epilepsy'] = 1
df.label[df.label == 'normal'] = 0
print(df.head())


Y = df["label"].values  
Y=Y.astype('int')


X = df.drop(labels = ["label"], axis=1) 
print(X.head())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

num_repeats = 20

precisions = []
recalls = []
aucs = []
accuracies = []
roc_aucs=[]


for _ in range(num_repeats):
    X_main, X_test, y_main, y_test = train_test_split(X, Y, test_size=0.2, random_state=None )
    X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=0.2, random_state=None )

    scaler=StandardScaler().fit(X_main)
    X_main_scaled=scaler.transform(X_main)

    scaler=StandardScaler().fit(X_train)
    X_train_scaled=scaler.transform(X_train)

    scaler=StandardScaler().fit(X_val)
    X_val_scaled=scaler.transform(X_val)

    scaler=StandardScaler().fit(X_test)
    X_test_scaled=scaler.transform(X_test)

    model = LogisticRegression() 
    model.fit(X_train_scaled, y_train)  

    prediction_test = model.predict(X_test_scaled)

  # Calculate performance metrics on the test set
    precision_test = precision_score(y_test, prediction_test)
    recall_test = recall_score(y_test, prediction_test)
    auc_test = roc_auc_score(y_test, prediction_test)
    acc_test = accuracy_score(y_test, prediction_test)
    fpr3, tpr3, threshold = metrics.roc_curve(y_test, prediction_test)
    roc_auc3 = metrics.auc(fpr3, tpr3)
    
    # Append the performance metrics to the lists
    precisions.append(precision_test)
    recalls.append(recall_test)
    aucs.append(auc_test)
    accuracies.append(acc_test)
    roc_aucs.append(roc_auc3)
   
#Test accuracy for various test sizes and see how it gets better with more training data
# Compute the mean and standard deviation of the performance metrics
mean_precision_tests = sum(precisions) / num_repeats
mean_recall_tests = sum(recalls) / num_repeats
mean_auc_tests = sum(aucs) / num_repeats
mean_acc_tests = sum(accuracies) / num_repeats
mean_roc_auc_tests = sum(roc_aucs) / num_repeats

std_precision_tests = np.std(precisions)
std_recall_tests = np.std(recalls)
std_auc_tests = np.std(aucs)
std_acc_tests = np.std(accuracies)
std_roc_auc_tests = np.std(roc_aucs)
# Print the mean and standard deviation of the performance metrics

print(f"Mean ACC: {mean_acc_tests}")
print(f"Standard Deviation AUC: {std_acc_tests}") 

print(f"Mean Precision: {mean_precision_tests}")
print(f"Standard Deviation Precision: {std_precision_tests}")

print(f"Mean Recall: {mean_recall_tests}")
print(f"Standard Deviation Recall: {std_recall_tests}")

print(f"Mean AUC: {mean_auc_tests}")
print(f"Standard Deviation AUC: {std_auc_tests}")



num_repeats = 20

# Initialize lists to store performance metrics
precisionss = []
recallss = []
aucss = []
accuraciess=[]
roc_aucss=[]

# Repeat the process for the desired number of times
for _ in range(num_repeats):
    X_main, X_test, y_main, y_test = train_test_split(X, Y, test_size=0.2, random_state=None )
    X_train, X_val, y_train, y_val = train_test_split(X_main, y_main, test_size=0.2, random_state=None )

    scaler=StandardScaler().fit(X_main)
    X_main_scaled=scaler.transform(X_main)

    scaler=StandardScaler().fit(X_train)
    X_train_scaled=scaler.transform(X_train)

    scaler=StandardScaler().fit(X_val)
    X_val_scaled=scaler.transform(X_val)

    scaler=StandardScaler().fit(X_test)
    X_test_scaled=scaler.transform(X_test)
    
    model = LogisticRegression()  
    model.fit(X_train_scaled, y_train)
    
    
    prediction_val = model.predict(X_val_scaled)
   
    precision_val = precision_score(y_val, prediction_val)
    recall_val = recall_score(y_val, prediction_val)
    auc_val = roc_auc_score(y_val, prediction_val)
    acc_val=accuracy_score(y_val, prediction_val)
    fpr4, tpr4, threshold = metrics.roc_curve(y_val, prediction_val)
    roc_auc4 = metrics.auc(fpr4, tpr4)
    

    # Append the performance metrics to the lists
    precisionss.append(precision_val)
    recallss.append(recall_val)
    aucss.append(auc_val)
    accuraciess.append(acc_val)
    roc_aucss.append(roc_auc4)
    
    
   
#Test accuracy for various test sizes and see how it gets better with more training data
# Compute the mean and standard deviation of the performance metrics
mean_precision_vals = sum(precisionss) / num_repeats
mean_recall_vals = sum(recallss) / num_repeats
mean_auc_vals = sum(aucss) / num_repeats
mean_acc_vals = sum(accuraciess) / num_repeats
mean_roc_auc_vals = sum(roc_aucss) / num_repeats


std_precision_vals = np.std(precisionss)
std_recall_vals = np.std(recallss)
std_auc_vals = np.std(aucss)
std_acc_vals = np.std(accuraciess)
std_roc_auc_vals = np.std(roc_aucss)

# Print the mean and standard deviation of the performance metrics

print(f"Mean ACC_val: {mean_acc_vals}")
print(f"Standard Deviation Acc_val: {std_acc_vals}")

print(f"Mean Precision_val: {mean_precision_vals}")
print(f"Standard Deviation Precision_val: {std_precision_vals}")

print(f"Mean Recall_val: {mean_recall_vals}")
print(f"Standard Deviation Recall_val: {std_recall_vals}")

print(f"Mean AUC_val: {mean_auc_vals}")
print(f"Standard Deviation AUC_val: {std_auc_vals}")

