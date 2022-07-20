# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:08:03 2022

@author: Connor GENT


"""
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from pandas import DataFrame

data = pd.read_csv("spambase.data")
X = data.iloc[:,0:57]
y = data.iloc[:,57]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

max_depths = [10,12,15]
min_sample_splits = [5,20,30]
tree_results=[]
for max_dep in max_depths:
    for min_samples_split in min_sample_splits:
        dt = DecisionTreeClassifier (max_depth=max_dep, min_samples_split=min_samples_split)
        dt.fit(X_train, y_train)
        tree_results.append({
            'Decision tree max_depth':max_dep,
           'min_samples_split':min_samples_split,
           'train_score':dt.score(X_train,y_train),
           'test_score':dt.score(X_test,y_test)
           
           })
df=DataFrame(tree_results)
print(df)

n_estimators = [50,100,300]
min_samples_leaves = [1,2,3]
rf_results=[]
for n_estimator in n_estimators:
    for min_samples_leaf in min_samples_leaves:
        rf = RandomForestClassifier(n_estimators=n_estimator, min_samples_leaf=min_samples_leaf)
        rf.fit(X_train,y_train)
        rf_results.append({
            'Random forest n_estimator':n_estimator,
            'min_samples_leaf':min_samples_leaf,
            'train_score':rf.score(X_train,y_train),
            'test_score':rf.score(X_test,y_test)
            
        })
df2=DataFrame(rf_results)
print(df2)

C=[5000,7000,9000]
gammas=[0.000001,0.00001,0.0001]
svm_results=[]
for c in C:
    for ga in gammas:
        svc = SVC(C=c,gamma=ga)
        svc.fit(X_train, y_train)
        svm_results.append({
            'svm C':c,
            'gamma':ga,
            'train_score':svc.score(X_train,y_train),
            'test_score':svc.score(X_test,y_test)
        
        })
df3=DataFrame(svm_results)
print(df3)
