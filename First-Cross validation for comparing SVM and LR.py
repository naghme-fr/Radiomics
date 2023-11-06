# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 00:11:08 2023

@author: Admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import  LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn import datasets, metrics, model_selection, svm



df = pd.read_csv(".csv")
print(df.head())


df.isnull().sum()
df = df.dropna()
df.isnull().sum()


df.label[df.label == 'epilepsy'] = 1
df.label[df.label == 'normal'] = 0
print(df.head())

y = df["label"].values 
y=y.astype('int')


X = df.drop(labels = ["label"], axis=1) 
print(X.head())

X.shape, y.shape


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler().fit(X)
X_scaled=scaler.transform(X)



sel = SelectFromModel(LogisticRegression(penalty = 'l2', C = 0.05, solver = 'liblinear'))
sel.fit(X_scaled, y)
sel.get_support()


finalselectedfeatures=X.columns[sel.get_support()]
finalselectedfeatures

X_l2 = sel.transform(X_scaled)

# one time with LR and one time with SVM for comparison
from sklearn.linear_model import LogisticRegression  
model_0 = LogisticRegression() 

from sklearn import svm
model_1 = svm.SVC(kernel='rbf',probability=(True))

from sklearn.model_selection import cross_validate
scores = cross_validate(model_0, X_l2, y, cv=10,
                        scoring=('accuracy', 'precision', 'recall', 'roc_auc'),
                        return_train_score=True)

print((scores['train_accuracy']))
print(np.mean(scores['train_accuracy']))
print(np.std(scores['train_accuracy']))

print((scores['train_precision']))
print(np.mean(scores['train_precision']))
print(np.std(scores['train_precision']))

print((scores['train_recall']))
print(np.mean(scores['train_recall']))
print(np.std(scores['train_recall']))

print((scores['train_roc_auc']))
print(np.mean(scores['train_roc_auc']))
print(np.std(scores['train_roc_auc']))

print((scores['test_accuracy']))
print(np.mean(scores['test_accuracy']))
print(np.std(scores['test_accuracy']))

print((scores['test_precision']))
print(np.mean(scores['test_precision']))
print(np.std(scores['test_precision']))

print((scores['test_recall']))
print(np.mean(scores['test_recall']))
print(np.std(scores['test_recall']))

print((scores['test_roc_auc']))
print(np.mean(scores['test_roc_auc']))
print(np.std(scores['test_roc_auc']))

