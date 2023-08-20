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


df = pd.read_csv("E:/99-CT/features/excels/final-2D Excels/WITH ADNI/ngtdm-selected.csv")
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

#scaler=StandardScaler().fit(X)
#X_scaled=scaler.transform(X)

# Set the number of desired repetitions
num_repeats = 20

# Initialize lists to store performance metrics
precisions = []
recalls = []
aucs = []
accuracies = []
roc_aucs=[]

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
    
    model = LogisticRegression()  #Create an instance of the model.
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

#UNDERSTAND WHICH VARIABLES HAVE MOST INFLUENCE ON THE OUTCOME
# To get the weights of all the variables

print(model.coef_) #Print the coefficients for each independent variable. 
#But it is not clear which one corresponds to what.
#SO let us print both column values and coefficients. 
#.Series is a 1-D labeled array capable of holding any data type. 
#Default index would be 0,1,2,3... but let us overwrite them with column names for X (independent variables)
weights = pd.Series(model.coef_[0], index=X.columns.values)

print("Weights for each variables is a follows...")
print(weights)

#tabdil konam be CSV
import six
print('Calculated firstorder features: ')
for (key, val) in six.iteritems(weights):
  print('  ', key, ':', val)
df=pd.DataFrame((weights))
df.head()
df.to_csv('weights-Selected FOrder-LogReg.csv')


#visualization
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

my_data1 = np.array([1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           1,
                 1,           1,           1,           1,           0,
                 1])

prediction1=np.array([1,           1,           1,           0,           1,
                 1,           0,           1,           1,           1,
                 0,           0,           1,           0,           0,
                 1,           1,           0,           1,           0,
                 1])


fpr1, tpr1, threshold = metrics.roc_curve(my_data, prediction)
roc_auc1 = metrics.auc(fpr1, tpr1)

fpr2, tpr2, threshold = metrics.roc_curve(my_data1, prediction1)
roc_auc2 = metrics.auc(fpr2, tpr2)


# method I: plt
import matplotlib.pyplot as plt
plt.title("Receiver Operating Characteristic(NGTDM features)")
plt.plot(fpr3, tpr3, 'g', label = 'Test Data   = %0.4f' % mean_roc_auc_tests)
plt.plot(fpr4, tpr4, 'm', label = 'Validation Data  = %0.4f' % mean_roc_auc_vals)
plt.plot(fpr1, tpr1, 'b', label = 'Visualization1 = %0.4f' % roc_auc1)
plt.plot(fpr2, tpr2, 'k', label = 'Visualization2 = %0.4f' % roc_auc2)
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
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
print(confusion_matrix(my_data1, prediction1, labels=[1,0]))
# Compute confusion matrix
cnf_matrix = confusion_matrix(my_data1, prediction1, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['epilepsy=1','epilepsy=0'],normalize= False,  title='Confusion matrix-Visualization 1')


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


roc_auc_ci(my_data, prediction)