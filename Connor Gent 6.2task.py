# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:23:36 2022

@author: Connor
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

Data = pd.read_csv('payment_fraud.csv')
Data = pd.get_dummies(Data, columns=['paymentMethod'])


def Generate_Coefficient_Magnitude(Data):
    fix, ax = plt.subplots(figsize=(7,7), dpi=100)
    
    x = pd.DataFrame(Data,columns=['accountAgeDays','numItems','localTime','paymentMethodAgeDays','paymentMethod_creditcard','paymentMethod_paypal','paymentMethod_storecredit'])
    y = pd.DataFrame(Data,columns=['label'])
    
    X_train, X_test, y_train, y_test = train_test_split(x,y, stratify=y, train_size=0.33)
    
    logreg1 = LogisticRegression(C=1).fit(X_train, y_train)
    print("Training set score: {:.3f}".format(logreg1.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(logreg1.score(X_test, y_test)))
    
    logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
    print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))
    
    logreg10 = LogisticRegression(C=10).fit(X_train, y_train)
    print("Training set score: {:.3f}".format(logreg10.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(logreg10.score(X_test, y_test)))
    
    logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
    print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))
    
    logreg0001 = LogisticRegression(C=0.001).fit(X_train, y_train)
    print("Training set score: {:.3f}".format(logreg0001.score(X_train, y_train)))
    print("Test set score: {:.3f}".format(logreg0001.score(X_test, y_test)))
    
    plt.plot(logreg1.coef_.T, '^', label="C=1")
    plt.plot(logreg100.coef_.T, 'v', label="C=100")
    plt.plot(logreg10.coef_.T, '*', label="C=10")
    plt.plot(logreg001.coef_.T, 'o', label="C=0.01")
    plt.plot(logreg0001.coef_.T, 's', label="C=0.001")
    
    plt.xticks(range(x.shape[1]), x.columns, rotation=90)
    
    xlims = plt.xlim()
    plt.hlines(0, xlims[0], xlims[1])
    plt.xlim(xlims)
    plt.ylim(-5, 5)
    plt.xlabel("Feature")
    plt.ylabel("Coefficient magnitude")
    plt.legend()
    
def Generate_Regression_Tree(Data):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    
    Data = pd.read_csv('payment_fraud.csv')
    Data = pd.get_dummies(Data, columns=['paymentMethod'])
    
    x = pd.DataFrame(Data,columns=['accountAgeDays','numItems','localTime','paymentMethodAgeDays','paymentMethod_creditcard', 'paymentMethod_paypal','paymentMethod_storecredit'])
    y = pd.DataFrame(Data,columns=['label'])
    X_train, X_test, y_train, y_test = train_test_split(x,y, stratify=y, train_size = 0.33)
    
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    
    print("Accuracy on training set: {:.3f}".format(tree.score(X_train,y_train)))
    print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
    
    print("Feature importances:")
    print(tree.feature_importances_)
    
    def plot_feature_importances_cancer(model):
        n_features = x.shape[1]
        plt.barh(np.arange(n_features), model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), x.columns)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.ylim(-1, n_features)
        
    plot_feature_importances_cancer(tree)
    
Generate_Coefficient_Magnitude(Data)
Generate_Regression_Tree(Data)
