# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 16:59:19 2015

@author: Christian
"""

import numpy as np

big = 50000

def transform(target):
    
    target = np.array(target)    
    
    smallIndex = target <= 1
    bigIndex = target >= big
    middleIndex = ~(smallIndex | bigIndex)
    
    target[smallIndex] = 1
    target[bigIndex] = 0
    target[middleIndex] = 1 - np.log(target[middleIndex]) / np.log(big)
    
    return target


def invTransform(target):
    
    target = np.exp((1 - target) * np.log(big))
    
    return target
