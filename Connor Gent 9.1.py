# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:09:44 2022

@author: Connor
"""

#Imports that are required to complete this task

from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.datasets import load_digits

#Loading the digit data and allocateing X,Y 
digits = load_digits()
x = digits.data 
y = digits.target 

#Defining the param grid and its data which can then be used to print later 
param_grid = {'C':[0.001,0.01,0.1,1,10,100],
              'gamma':[0.001,0.01,0.1,1,10,100]}

#Manipulation of the data 
grid_search = GridSearchCV(SVC(), param_grid, cv = 5, return_train_score = True) #returning train score if its true

#Spliting the dataset 
X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=0)

#Using the grid search inorder to fit the data 
grid_search.fit(X_train, y_train)

def Gen_Output(pg, gs):
    print('Parameter Grid: ') #Prints the data in the param grind 
    print(pg) 
    print("Test set score: {:.2f}".format(gs.score(X_test, y_test))) #Formatting the data 
    print("Best Parameters: {}".format(gs.best_params_))
    print("Best Cross-validation score: {:.2f}".format(gs.best_score_))
    print("Best estimator:\n{}".format(gs.best_estimator_))
    
Gen_Output(param_grid, grid_search)

