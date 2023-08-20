# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:42:53 2023

@author: Admin
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectFromModel

df = pd.read_csv("E:/99-CT/features/excels/final-2D Excels/WITH ADNI/ngtdm.csv")
print(df.head())

df.isnull().sum()

#df.drop(labels = ['age', 'deck'], axis = 1, inplace = True)
df = df.dropna()
df.isnull().sum()

#oonha ke mikhad ro jaygozin mikone
#data = df[['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'who', 'alone']].copy()
#data.head()
#data.isnull().sum()

# bad miad datahaye keifi ro kami mikone:
#sex = {'male': 0, 'female': 1}
#data['sex'] = data['sex'].map(sex)
#data.head()

df.label[df.label == 'epilepsy'] = 1
df.label[df.label == 'normal'] = 0
print(df.head())

Y = df["label"].values 
Y=Y.astype('int')

X = df.drop(labels = ["label"], axis=1) 
print(X.head())

X.shape, Y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

#Estimation of coefficients of Linear Regression ke zarayebe linear reg ra hesab mikone ba astaneye default ke mean az zareyb hast
sel = SelectFromModel(LinearRegression())
sel.fit(X_train, y_train)

sel.get_support()

sel.estimator_.coef_

mean = np.mean(np.abs(sel.estimator_.coef_))
mean

np.abs(sel.estimator_.coef_)

features = X_train.columns[sel.get_support()]
features

X_train_reg = sel.transform(X_train)
X_test_reg = sel.transform(X_test)
X_test_reg.shape

def run_randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    
%%time
run_randomForest(X_train_reg, X_test_reg, y_train, y_test)    

#hala ba dataye asli
%%time
run_randomForest(X_train, X_test, y_train, y_test)

#hala ba svm mizanam:
%%time 
from sklearn import svm
model = svm.SVC(kernel='rbf')
model.fit(X_train_reg, y_train)

#STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE
prediction_test = model.predict(X_test_reg)

from sklearn import metrics
#Print the prediction accuracy

print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test)) 

#Logistic Regression Coefficient with L1 Regularization:
sel = SelectFromModel(LogisticRegression(penalty = 'l1', C = 0.05, solver = 'liblinear'))
sel.fit(X_train, y_train)
sel.get_support()    

sel.estimator_.coef_

X_train_l1 = sel.transform(X_train)
X_test_l1 = sel.transform(X_test)

%%time
run_randomForest(X_train_l1, X_test_l1, y_train, y_test)


%%time 
from sklearn import svm
model = svm.SVC(kernel='rbf')
model.fit(X_train_l1, y_train)

#STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE
prediction_test = model.predict(X_test_l1)

from sklearn import metrics
#Print the prediction accuracy

print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test)) 


#ba asli
%%time 
from sklearn import svm
model = svm.SVC(kernel='rbf')
model.fit(X_train, y_train)

#STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE
prediction_test = model.predict(X_test)

from sklearn import metrics
#Print the prediction accuracy

print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test)) 

#L2 Regularization
sel = SelectFromModel(LogisticRegression(penalty = 'l2', C = 0.05, solver = 'liblinear'))
sel.fit(X_train, y_train)
sel.get_support()

sel.estimator_.coef_

X_train_l2 = sel.transform(X_train)
X_test_l2 = sel.transform(X_test)

%%time  
run_randomForest(X_train_l2, X_test_l2, y_train, y_test)


%%time 
from sklearn import svm
model = svm.SVC(kernel='rbf')
model.fit(X_train_l2, y_train)

#STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE
prediction_test = model.predict(X_test_l2)

from sklearn import metrics
#Print the prediction accuracy

print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test)) 

%%time 
from sklearn import svm
model = svm.SVC(kernel='rbf')
model.fit(X_train, y_train)

#STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE
prediction_test = model.predict(X_test)

from sklearn import metrics
#Print the prediction accuracy

print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test)) 