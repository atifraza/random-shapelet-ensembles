# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:02:02 2016

@author: atif
"""
import math

ds_size = int(input("Enter the train set size: "))
ts_length = int(input("Enter the time series length: "))
min_len = int(math.ceil(ts_length/4.)) #int(input("Enter the minimum length of shapelet candidates: "))
max_len = int(math.floor(ts_length*2/3.)) #int(input("Enter the maximum length of shapelet candidates: "))
stepSize = 1 #int(input("Enter step size: "))
count=0
for l in range(min_len, max_len+1, stepSize):
    count += (ts_length-l+1)*ds_size
#    for t in range(0, ds_size):
#        count += (ts_length-l+1)

print "Number of shapelet candidates = ", count
