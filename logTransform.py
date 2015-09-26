# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 16:59:19 2015

@author: Christian
"""

import numpy as np

def logTransform(meas):
    
    big = 50000
    smallIndex = meas <= 1
    bigIndex = meas >= big
    middleIndex = ~(smallIndex | bigIndex)
    
    meas[smallIndex] = 1
    meas[bigIndex] = 0
    meas[middleIndex] = 1 - np.log(meas[middleIndex]) / np.log(big)
    
    return meas
