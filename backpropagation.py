# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:15:18 2015

@author: Christian
"""

import numpy as np
from threshold import diffSmoothThreshold


def backward(hiddenLayer, weightMatrix2, outputLayer, meas):
    
    # beregn delta for output laget. Se formel i mathtype dok.
    # formel kommer fra at aflede fejlen i forhold til signalet.
    # at det er signalet, kommer fra at kædereglen for diff gør det
    # muligt at regne fejlen afledt i forhold til vægtene.
    outputDelta = -2 * meas * diffSmoothThreshold(outputLayer)
    
    # vi går tilbage et skridt med formel fra caltech
    hiddenDelta = diffSmoothThreshold(hiddenLayer) * weightMatrix2 * outputDelta
    
    return hiddenDelta, outputDelta
    
    
    
def updateWeight(inputLayer, weightMatrix1, hiddenLayer, weightMatrix2, hiddenDelta, outputDelta, learningRate):
    
    # there is only used the deltas from the hidden layer 1:8 since the bias is not made from the input layer
    # it has to be converted to the matrix format to make sure it's a vertical vector and not just an array
    hiddenDelta = np.matrix(hiddenDelta[0, 0:8]).T
    
    weightMatrix1 -= learningRate * hiddenDelta * inputLayer
    weightMatrix2 -= learningRate * outputDelta * hiddenLayer
    
    return weightMatrix1, weightMatrix2