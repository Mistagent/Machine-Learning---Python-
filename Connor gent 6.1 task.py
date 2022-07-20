# -*- coding: utf-8 -*-
"""
Created on Mon May 16 08:36:30 2022

@author: Connor
"""

#Connor Gent task 6.1 

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from matplotlib.colors import ListedColormap
import seaborn as sns 
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("task6_1_dataset.csv", index_col=0) #Reading file from the CSV
print(df.head())

sns.set_theme()
X = df[["x1", "x2"]] #getting and acquiring the data points from the dataframe
y = df["y"] #Getting the labels

clf = KNeighborsClassifier(n_neighbors=1) #Training the classifier 
clf.fit(X,y)

test_point = [[-6, 6]] #Defining the test points shown on the task sheet 
prediction = clf.predict(test_point)[0] # Making the prediction 

colors = ["green", "blue", "magenta"] #Color list used for the graph 

fig,ax = plt.subplots(figsize = (7,5), dpi=100) #Defining plot axes 
sns.scatterplot(ax=ax,x=X["x1"], y=X["x2"], hue=y,alpha = 0.6,s=75,  
                palette=colors,legend=False,edgecolor = None) #Plotting using scatter plot. 

ax.scatter(test_point[0][0], test_point[0][1],s=200,marker='x',alpha =1, c=colors[int(prediction)]) #Plotting the test results that have been defined
ax.annotate("(-6,6), test point",(-6,6),color='red',fontsize=10)
ax.set_title(f"3-class classification (k=1)\n the test point is predicted as {colors[int(prediction)]}") #settin the title

plt.show()


X = np.array(df[["x1", "x2"]]) #Data and test points as numpy arrays 
y = np.array(df["y"])

clf = KNeighborsClassifier(n_neighbors=15) #Training the classifer with n-15 this time 
clf.fit(X,y) #Fitting the data 

test_point = [[-1, 5]] # Defining the test points 
prediction = clf.predict(test_point)[0]
colors = ["green", "blue", "magenta"]
cmap_Light = ListedColormap(['#AAFFAA','#AAAAFF','#FFAAAA']) #Color map for decision regions 

h=0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 #Creating grid. 
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape) #Reshaping the output of the grind 

fig,ax = plt.subplots(figsize = (7,5),dpi=100)
plt.contourf(xx, yy, Z, cmap=cmap_Light)

sns.scatterplot(ax=ax, x=X[:,0],y=X[:,1], hue=y,alpha =0.6, s=75,
                palette=colors,legend=False,edgecolor= None)
ax.scatter(test_point[0][0], test_point[0][1], s=200,marker='x',alpha=1, c=colors[int(prediction)])
ax.annotate("(-1,5), test point", (-1,5),color='red', fontsize=10)
ax.set_title(f"3-class classification (k=15)\n the test prediction as {colors[int(prediction)]}")
plt.show()