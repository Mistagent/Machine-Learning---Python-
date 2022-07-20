# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:23:27 2022

@author: Connor
"""

from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import itertools
from sklearn.metrics import confusion_matrix,precision_recall_curve
from itertools import cycle

#Here I am defining a function to find the best parameter that will allow recall
def gridsearch_print(model, param_grid,para,X_train_undersample,y_train_undersample):
    clf = GridSearchCV(model, param_grid, cv=5, scoring='recall')
    clf.fit(X_train_undersample,y_train_undersample.values.ravel())
    
    print ("Best paras on undersamp train set:")
    print (clf.best_params_)
    
    return clf.best_params_[para]

def get_confusion_matrix(model, X_train_undersample, y_train_undersample, X_test_undersample, y_test_undersample):
    model.fit(X_train_undersample, y_train_undersample.values.ravel())  
    y_pred_undersample = model.predict(X_test_undersample.values)
    #Compute confusion matrix
    cf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
    np.set_printoptions(precision=2)
    plt.figure() #Plot non-normalized confusion matrix
    plot_confusion_matrix(cf_matrix, classes=[0,1], title='Confusion matrix')
    plt.show()
    
    y_pred = model.predict(X_test.values) #Computes confusion matrix 
    cf_matrix_test = confusion_matrix(y_test,y_pred)
    
    np.set_printoptions(precision=2)
    plt.figure() #Plot non-normalized confusion matrix
    plot_confusion_matrix(cf_matrix_test, classes=[0,1], title='Confusion matrix for und samp prediction')
    plt.show()


def recall_matrix(model, X_test_undersample, y_test_undersample):
    plt.figure(figsize=(10,10))
    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    y_pred_undersample_proba = model.predict_proba(X_test_undersample.values)
    
    j = 1
    for i in thresholds:
        y_test_predictions_high_recall = y_pred_undersample_proba[:,1] > i
        
        plt.subplot(3,3,j)
        j += 1 #Computes confusion matrix
        cf_matrix = confusion_matrix(y_test_undersample, y_test_predictions_high_recall)
        np.set_printoptions(precision=2) #Plots the non normalized confusion matrix
        print("Recall metric in the undersample testing dataset for threshold {}: {}".format(i, cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])))
        
        class_names = [0,1]
        plot_confusion_matrix(cf_matrix, classes=class_names,title='Threshold >= %s' %i)
        
def plot_confusion_matrix(cm, classes, normalize=False,title='Confusion matrix', cmap=plt.cm.Blues): #Function used to print and plot confusion matrix.
    plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    if normalize: 
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #Prints the nomalized matrix
    else:
         1. #Prints the confusion matrix
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.ylim([1.5, -0.5])


def plot_precision_curve(model,X_test_undersample,y_test_undersample): #Plots the precision recall curve for undersample data set
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue', 'black'])
    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    plt.figure(figsize=(5,5))
    
    y_pred_undersample_proba = model.predict_proba(X_test_undersample.values)
        
    for i, color in zip(thresholds,colors):
        y_test_predictions_prob = y_pred_undersample_proba[:,1] > i 
        precision, recall, thresholds = precision_recall_curve(y_test_undersample,y_test_predictions_prob)
        #Plotting precision-recall curve
        plt.plot(recall, precision, color=color, label='Threshold: %s'%i)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example')
        plt.legend(loc="lower left")

data = pd.read_csv("creditcard.csv")
X = data.iloc[:, data.columns != 'Class'] #Splitting the dataset into train set and test
y = data.iloc[:, data.columns == 'Class']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

#Number of datapoints in the class fraud
number_records_fraud = len(data[data.Class == 1])
# Pick the indices of the fraud and normal classes
fraud_indices = np.array(data[data.Class == 1].index)
normal_indices = data[data.Class == 0].index
#Out of the indicies, it randomly selects the number 
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)
#Append 2 indicies 
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])
#Under sample dataset
under_sample_data = data.iloc[under_sample_indices,:]
X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']
#Undersample X and y
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,y_undersample,test_size =0.3,random_state = 0)
#Setting the parameter grid
param_grid_knn = {'n_neighbors':[1, 2, 3, 4, 5]}

knn = neighbors.KNeighborsClassifier() #Defines the model 
#Uses function to find the best params
best_params_knn = gridsearch_print(knn,param_grid_knn, 'n_neighbors', X_train_undersample, y_train_undersample)
#Find the best params of decision free
param_grid_decisiontree={'max_depth':[3, 4, 5, 6, 7]}

dt = DecisionTreeClassifier()
best_params_dt = gridsearch_print(dt, param_grid_decisiontree, 'max_depth', X_train_undersample, y_train_undersample)
#Best param of RF (random forest)
param_grid_randomforest={'n_estimators': [5, 10, 20, 50]}
rf = RandomForestClassifier()
best_params_rf = gridsearch_print(rf,param_grid_randomforest,'n_estimators',X_train_undersample,y_train_undersample)

"""
est parameter to build the clasifier with the undersample training dataset and 
predict classes via undersample test dataset and test dataset
"""
knn2 = neighbors.KNeighborsClassifier(n_neighbors=best_params_knn) # define model
get_confusion_matrix(knn2,X_train_undersample,y_train_undersample,X_test_undersample,y_test_undersample)
# plot recall matric with different threshold for the undersample dataset
recall_matrix(knn2,X_test_undersample,y_test_undersample)
# plot precision-recall curve for the undersample dataset
plot_precision_curve(knn2,X_test_undersample,y_test_undersample)

dt2 = DecisionTreeClassifier(max_depth=best_params_dt)
get_confusion_matrix(dt2,X_train_undersample,y_train_undersample,X_test_undersample,y_test_undersample)
recall_matrix(dt2,X_test_undersample,y_test_undersample)
# plot precision-recall curve for the undersample dataset
plot_precision_curve(dt2,X_test_undersample,y_test_undersample)

rf2 = RandomForestClassifier(n_estimators=best_params_rf)
get_confusion_matrix(rf2,X_train_undersample,y_train_undersample,X_test_undersample,y_test_undersample)
recall_matrix(rf2,X_test_undersample,y_test_undersample)
# plot precision-recall curve for the undersample dataset
plot_precision_curve(rf2,X_test_undersample,y_test_undersample)
 

