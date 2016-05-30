# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:02:02 2016

@author: atif
"""

ds_size = int(input("Enter the train set size: "))
ts_length = int(input("Enter the time series length: "))
min_len = int(input("Enter the minimum length of shapelet candidates: ")) #int(math.ceil(ts_length/4.))
max_len = int(input("Enter the maximum length of shapelet candidates: ")) #int(math.floor(ts_length*2/3.))
stepSize = 1 #int(input("Enter step size: "))
count=0
for l in range(min_len, max_len+1, stepSize):
    count += (ts_length-l+1)*ds_size
#    for t in range(0, ds_size):
#        count += (ts_length-l+1)

print "Number of shapelet candidates = ", count

sampled_count = int(count*0.01)

for stepSize in range(1,100):
    newcount=0
    for l in range(min_len, max_len+1, stepSize):
        newcount += (ts_length-l+1)*ds_size
    if newcount < sampled_count:
        break
    
print "Effective step size for 10% sampling: ", stepSize
