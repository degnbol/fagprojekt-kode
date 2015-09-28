# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 18:57:30 2015

@author: Christian
"""

import numpy as np
from activation import activation

def forward(inputLayer, weightMatrix1, weightMatrix2):
            
    # beregn foreløbig hidden layer vha. w matrix og binær sekvens
    hiddenLayer = np.dot(weightMatrix1, inputLayer)
    
    # brug blød threshold på hidden layer: y = 1/(1 + exp(-x))
    hiddenLayer = activation(hiddenLayer)
    
    # bias
    hiddenLayer = np.append(hiddenLayer, 1) 
    
    # beregn resultat med hidden layer og w matrix 2 (som blot er liggende vektor)
    outputLayer = np.dot(weightMatrix2, hiddenLayer)
    
    # brug blød threshold på resultat
    outputLayer = activation(outputLayer)    
    
    return outputLayer