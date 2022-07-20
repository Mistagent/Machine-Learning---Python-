# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:57:12 2022

@author: Connor
"""

def my_intro():
    print("Hey my names Connor Gent Its nice to meet you")
def welcome():
    print("\nWelcome to Deakin pass task 1.2... Let us begin")

'***************Task 1.2**************'
my_intro()
welcome()


'*************************************'
    

m = int(input("Please enter an integer number: ")) #Taking user input here 
while m > 0: #While loop is user here if the number is greater than 0 
    print("You have entered", m)
    break

while m < 0: # while loop is created if the input is less than 0 
        print('Sorry Please enter a number greater than 0') 
        m = int(input("Please enter an integer:" )) # reasks the user for a number 

    
star = "*" #Star variable is created
for i in range(m): #for loop that repeats the range defined by the user input. 
    print(star*m) #Star variable is multiplied by the m variable which value is based on user input 
