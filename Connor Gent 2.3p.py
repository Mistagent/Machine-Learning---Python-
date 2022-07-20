# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:00:15 2022

@author: Connor
"""

import re 



regex = '^[A-Z0-9]+[\._]?[A-Z0-9]+[@]\w+[.]\w{2,3}$' #


def check_email_address(address):
    
    is_valid = re.search(regex, address)
    
    if is_valid:
        return True 
    else:
        print ('Oh no Your email is invalid please try again')
        return False 

address = str(input('Please enter your email address: '))

while not check_email_address(address):
    address = input ('Please enter your email address: ')
else:
    print ('EMAIL:', address)