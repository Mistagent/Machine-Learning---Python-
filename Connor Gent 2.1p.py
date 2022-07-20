# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:38:08 2022

@author: Connor
"""
def taskIntro():
    print("\n For task 2.1 I must submit a code that does not use Recursion function....")
def taskDetails():
    print("\n This code below is my attempt at not using recursion function to complete the required task")

print('*******************************************')

taskIntro()
taskDetails()

def identFactorial(x):
    result = 1 
    
    while x > 1: #Loop begins if result is greater than 1 
        result = result * x # result is multipled by N 
        x -= 1 #Decreasing the int number by 1 
        
    return result 

n = int(input("Please enter a non-negative number:"))

while n < 0: #Loop occurs if result is a negative number
    print("Your input was a negative number you must input a non-negative number")
    n = int(input("Please enter a non-negative number:")) # asks the user for their input again 

print("Factorial of", n , ":" , identFactorial(n)) 
#input n and the factorial formula are combined to produce the result entered by the user. 