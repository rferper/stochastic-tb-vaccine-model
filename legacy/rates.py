# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:24:09 2019

@author: rfern
"""

import numpy
l=numpy.random.uniform(0,1,	424523)
l.sort()
diff_list = [] 
for x, y in zip(l[0::], l[1::]): 
    diff_list.append(y-x) 
print(numpy.mean(diff_list))