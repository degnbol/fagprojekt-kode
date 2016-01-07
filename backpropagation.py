# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:15:18 2015

@author: Christian
"""
import numpy as np
from threshold import diffSmoothThreshold

def backward(layer, nextWeight, nextDelta):
    """
    if there were more layers and nextWeight became quadratic this function would
    need to be controled with matrix types instead of array types. Since we only
    have one way of multiplying where the dimensions fits, we don't need to worry
    about all that. Numpy takes care of it for us.
    """  
    
    return diffSmoothThreshold(layer) * nextWeight * nextDelta
    
def updateWeight(previousLayer, weight, nextDelta, learningRate):

    # np.dot only works if they are matrix types because they are column vectors
    nextDelta = np.mat(nextDelta).T
    previousLayer = np.mat(previousLayer)        
    dEdWeight = np.array(nextDelta.dot(previousLayer))
    # convert back to use arrays in general    
    
    return weight - learningRate * dEdWeight