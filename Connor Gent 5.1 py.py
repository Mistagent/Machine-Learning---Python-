# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 12:51:10 2022

@author: Connor
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv('data_correlation.csv') #Loading the data into the pandas dataframe. 

corr_a = {} #Here we are correlating the coefficients. We can find the correlation coefficients between 'a', 'b', 'c' and d. 
#This can be done in a single loop and saved in a dictionary. 
#Peasons co-efficient is a term with index [0,1] or [1,0]
for var in ['b', 'c', 'd']:
    corr_a[var] = np.corrcoef(df['a'], df[var])
    
for key, val in corr_a.items(): #Now the correlations have been defined, I used the data to print the correlation value 
#Again I used a loop to iterate the dictionary and visualize everything required for this task. 
    print('a and ' + key + ' pearson_r: ' + str(val[0,1]))
    print('a and ' + key + ' corrcoef: ' + str(val)) #Printed the correlations and pearson_r

#Here I'm seeing if iitmes in the dictiornary correlation value is greater than 5.    
    x = df['a'] 
    y = df[key]
    if abs(val[0,1]) > 0.5:
        color = 'red' if val[0,1] > 0.5 else 'green'
        line_coef = np.polyfit(x, y, 1) #Polyfit is used if there is a strong correlation. 
        xx = np.arange(0, 50, 0.1)
        yy = line_coef[0]*xx + line_coef[1]
        plt.plot(xx, yy, color, lw = 2)
    #Assign the color of the line to red if the correlation value is positive 
    #Green is assigned if it is negative 

    plt.scatter(x, y)
    plt.xlabel('a')
    plt.ylabel(key)
    plt.show()
        