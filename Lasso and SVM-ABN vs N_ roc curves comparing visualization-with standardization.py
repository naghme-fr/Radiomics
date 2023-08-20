# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 00:11:08 2023

@author: Admin
"""
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectFromModel

df = pd.read_csv("E:/99-CT/features/excels/final-2D Excels/WITH ADNI/Whole selected features with lasso for SVM .csv")
print(df.head())


#sizes = df['label'].value_counts(sort = 1)

#plt.pie(sizes, shadow=True, autopct='%1.1f%%') 

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


#X=df[['original_firstorder_MeanAbsoluteDeviation']]
X = df.drop(labels = ["label"], axis=1) 
print(X.head())

X.shape, Y.shape

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

#X_train standardization
scaler=StandardScaler().fit(X_train)
#scaler.mean_
#scaler.scale_

X_train_scaled=scaler.transform(X_train)

#X_test standardization
scaler=StandardScaler().fit(X_test)
X_test_scaled=scaler.transform(X_test)

#np.savetxt("test.csv", X_test_scaled, delimiter=",")

sel = SelectFromModel(LogisticRegression(penalty = 'l2', C = 0.05, solver = 'liblinear'))
sel.fit(X_train_scaled, y_train)
sel.get_support()
sel.threshold_

sel.estimator_.coef_

finalselectedfeatures=X_train.columns[sel.get_support()]

X_train_l2 = sel.transform(X_train_scaled)
X_test_l2 = sel.transform(X_test_scaled)


%%time 
from sklearn import svm
model = svm.SVC(kernel='rbf') #poly was the best

#testing cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train_l2,y_train, cv=5)
scores

model.fit(X_train_l2, y_train)

#STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE
prediction_test = model.predict(X_test_l2)

from sklearn import metrics
#Print the prediction accuracy

print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test)) 


from sklearn.metrics import roc_auc_score
from math import sqrt

#Calculating confidence interval of ROC-AUC. https://gist.github.com/doraneko94/e24643136cfb8baf03ef8a314ab9615c
def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, upper)

roc_auc_ci(y_test, prediction_test)

# test drawing roc curve
fpr, tpr, threshold = metrics.roc_curve(y_test, prediction_test)
roc_auc = metrics.auc(fpr, tpr)
 

#from numpy import genfromtxt
#my_data = genfromtxt('E:/99-CT/features/excels/Visualization-NORM VS ABN/my_data.csv', delimiter=',')
#my_data=my_data.astype(int)

#predictiontest = genfromtxt('E:/99-CT/features/excels/Visualization-NORM VS ABN/prediction_test.csv', delimiter=',')
#predictiontest=predictiontest.astype(int)


my_data = np.array([1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           0,
                 1])

prediction=np.array([1,           1,           0,           0,           1,
                 1,           0,           1,           1,           1,
                 0,           0,           1,           0,           0,
                 1,           1,           0,           1,           0,
                 1])

roc_auc_ci(my_data, prediction)

fpr1, tpr1, threshold = metrics.roc_curve(my_data, prediction)
roc_auc1 = metrics.auc(fpr1, tpr1)
#time 
#from sklearn import svm
#model = svm.SVC(kernel='rbf')
#model.fit(X_train, y_train)

#STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE
#prediction_test = model.predict(X_test)

#from sklearn import metrics
#Print the prediction accuracy

#print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test)) 



import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr1, tpr1, 'b', label = 'Visualization AUC = %0.2f' % roc_auc1)
plt.plot(fpr, tpr, 'g', label = 'SVM_MODEL AUC= %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#JADI-COFUSION MATRIX
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, prediction_test, labels=[1,0]))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, prediction_test, labels=[1,0])
np.set_printoptions(precision=2)

print (classification_report(y_test, prediction_test))
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Epilepsy=1','Normal=0'],normalize= False,  title='Confusion matrix-SVM')

from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(my_data, prediction, labels=[1,0]))
# Compute confusion matrix
cnf_matrix = confusion_matrix(my_data, prediction, labels=[1,0])
np.set_printoptions(precision=2)

print (classification_report(my_data, prediction))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Epilepsy=1','Normal=0'],normalize= False,  title='Confusion matrix-Visualization')