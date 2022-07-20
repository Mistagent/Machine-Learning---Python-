# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:10:42 2022

@author: Connor
"""

#importing the dataset

import pandas as pd

import numpy as np

df1 = pd.read_csv('result_withoutTotal.csv') #DF1 reading the data contained in the csv file.

#creating the list for total , final and grad


fin = list() #this is the creation of the final list 

tot =list() #creates the total list 

grad=list() #creates the grade list 

for row in df1.iterrows(): #for loop

    a= list(row) #a is the list created here. 

    total = ((5/100)*(a[1].Ass1+a[1].Ass3))+((15/100)*(a[1].Ass2+a[1].Ass4))+((60/100)*a[1].Exam) #Added to the total

    if(total>100): #If total is greater than 100

        total =100 #total equals 100 

    #finding the fianl 

    if(total>=50 and a[1].Exam>=48): #if statement 

        final= round(total) 

    elif(total<=50 and a[1].Exam<=48): #Else if statement 

        final= round(total)

    else:

        final = 44

    #finding the grade 

    if(final<=49.45): 

        grade='N'

    elif(final>49.45 and final <=59.45):

        grade= 'P'

    elif(final>59.45 and final <=69.45):

        grade= 'C'

    elif(final >69.45 and final<=79.45):

        grade= 'D'

    else:

        grade= 'HD'

    #appending all values to specific list

    tot.append(total) #tot appended to the total list 

    fin.append(final) #fin appeneded to the final list 

    grad.append(grade) #grad appened to the grade list 

df1['Total']=tot

df1['Final']=fin

df1['Grade']=grad

#making id as a index 

df1.set_index(['ID'],inplace=True) 

#Writing Dataframe to csv 

df1.to_csv('result_withoutTotal.csv', header=True,index=True) 

df2 = df1[df1['Final']<48] #df2 created

print("Students with marks <48")

print(df2) #Prints updated table 

print("Students with marks >100")

df2 = df1[df1['Final']>=100] 

print(df2) #Prints 2nd table of grade results 

df2.to_csv('Failed_hurdle.csv') #failed hurting results are updated. 