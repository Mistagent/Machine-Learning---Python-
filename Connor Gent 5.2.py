# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:16:31 2022

@author: Connor
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
   
import seaborn as sns 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('admission_predict.csv')
df.head()
df.describe()



X = df['GRE_SCORE'].values

y = df['CHANCE_OF_ADMIT'].values.tolist()

 
split = 0.75 
split_idx = int(np.round(split * len(df)))
split_idx

x = np.c_['']

X_test = X['GRE_']


lr = linear_model.LinearRegression()

lr.fit(X_test,y_test)

y_pred = lr.predict(X_test)

accuracy = r2_score(y_test,y_pred)
print('accuracy =',accuracy)

X = df['CGPA'].values
y = df['CHANCE_OF_ADMIT'].tolist()



lr = linear_model.LinearRegression()

lr.fit(X_test,y_test)

y_predict = lr.predict(X_test)

accuracy = r2_score(y_test,y_predict)
print('accuracy =',accuracy)

residuals = np.subtract(y_pred, X_test)

residuals_ = np.subtract(y_predict, X_test)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,10),dpi=100)
ax[i][j].scatter(X_test,y_test)
ax[0][1].set_xlabel('GRE SCORE')
ax[0][1].set_ylabel('Chance of admit')
ax[0][1].set_title('Linear regression of GRE score and chance of admit')
plt.show()

sns.regplot(X_test,y=residuals,ax=ax[0,1])
ax[0, 1].set_xlabel('GRE score ')
ax[0, 1].set_ylabel('Chance of admit')
ax[0, 1].set_title('GRE SCORE vs CHANCE of admit:true value and residual')
plt.show()

sns.regplot(x=X_test,y=y_predict,ax=ax[1,0])
ax[1,0].set_xlabel('CGPA')
ax[1,0].set_ylabel('Chance of admit')
ax[1,0].set_title('Linear regression of CGPA and Chance of admit')
plt.show()

sns.regplot(x=X_test,y=residuals_, ax=ax[1,1])
ax[1,1].set_xlabel('CGPA')
ax[1,1].set_ylabel('Chance of admit')
ax[1,1].set_title('CGPA vs chance of admit:true value and residual')
plt.show()