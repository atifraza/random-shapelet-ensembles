# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:02:02 2016

@author: atif
"""
min_len = int(input("Enter the minimum length of shapelet candidates: "))
max_len = int(input("Enter the maximum length of shapelet candidates: "))
ds_size = int(input("Enter the time series dataset size: "))
ts_length = int(input("Enter the time series length: "))
stepSize = int(input("Enter step size: "))
count=0
for l in range(min_len, max_len+1, stepSize):
    for t in range(0, ds_size):
        count += (ts_length-l+1)

print "Number of shapelet candidates = ", count