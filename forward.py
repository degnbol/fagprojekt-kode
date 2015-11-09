# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 18:57:30 2015

@author: Christian
"""

import numpy as np
from threshold import smoothThreshold

def forward(inputLayer, weight1, weight2):
    
    # beregn foreløbig hidden layer vha. w matrix og binær sekvens
    preHiddenLayer = np.dot(weight1, inputLayer)
    
    # brug blød threshold på hidden layer: y = 1/(1 + exp(-x))
    hiddenLayer = smoothThreshold(preHiddenLayer)
    
    # bias
    hiddenLayer = np.append(hiddenLayer, 1) 
    
    # beregn resultat med hidden layer og w matrix 2 (som blot er liggende vektor)
    preOutputLayer = np.dot(weight2, hiddenLayer)
    
    # brug blød threshold på resultat
    outputLayer = smoothThreshold(preOutputLayer)    
    
    return hiddenLayer, outputLayer