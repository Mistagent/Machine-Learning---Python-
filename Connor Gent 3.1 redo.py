# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:58:03 2022

@author: Connor
"""

import numpy as np 
import pandas 

file = pandas.read_csv('result_withoutTotal.csv', header = None, skiprows = 1, dtype = float);

aTotal = np.zeros(len(file[file.shape[1]-1])); #The np.zeros is used to create an aray. 
#file.shape[1] returns column number of the file. These returns do not include the index 
#File[1] is the assin1 column 
#The code stores a column in an array
i = 0;

for x in file[file.shape[1]-1]:
    aTotal[i] = x; 
    i += 1;
#Nd arrau does not have an attirbute 'index' - therefore it transfers into list; list.index(value) this returns the values corresponding index
index = aTotal.tolist().index(max(aTotal));
print("Total Max:", max(aTotal), ", min:", min(aTotal), ", average: ", sum(aTotal)/len(file[1]));


#array high stores scores that are contained in aassin1 to exam with the defined index 
array = np.zeros(len(file[1]));
arrayHigh = np.zeros(file.shape[1]-1);

m = 0; # from assin1 to exam

for i in range(1, file.shape[1] -1):
    g = 0;
    for x in file[i]:
        array[g] = x;
        g += 1;
    arrayHigh[m] = array[index];
    m += 1; 
    if i < 5: 
        print("assin", i, "max:", max(array), ", min:", min(array), ", average:", sum(array)/len(file[i]));
    else:
        print("Exam max:", max(array), ", min:", min(array), ", average:", sum(array)/len(file[1]));
print("Student with highest total:\n", "ID:", index + 1, "assin1:", arrayHigh[0], " assin2:", arrayHigh[1], "assign3:", arrayHigh[2], " assign4:",arrayHigh[3], "exam:", arrayHigh[4], "total:", max(aTotal));