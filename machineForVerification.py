# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:33:39 2015

@author: Christian, Marianne, Kathrine & Cecilia
"""

# importer ting
import os
import sys
import numpy as np

# ændr sti så vi kan finde hjælpefunktioner
projectPath = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(projectPath)

# importer hjælpefunktioner
import fileUtils
import logTransform
import sequenceUtils
from forward import forward
import backpropagation


# set seed to be able to reproduce results
np.random.seed(1234)


def train(inputLayer, target, weight1, weight2, hiddenNodes, epochs, learningRate):
    
    print("Starting training.")   
    
    for epoch in range(epochs):
        # train on training set
        
        error = np.zeros(1)
        
        for index in range(1):
           
            # run the forward function
            hiddenLayer, outputLayer = forward(inputLayer, weight1, weight2)

            print("Hidden layer:")
            print(hiddenLayer)
            print("Output layer:")
            print(outputLayer)
 
            # save the error
            error[index] = 1/2 * (outputLayer - target)**2
            print("Error: {:.8f}.".format(error[index]))
            errorDelta = outputLayer - target
            
            # backpropagation
            outputDelta = backpropagation.backward(outputLayer, 1, errorDelta)
            
            print("Output delta: {:.8f}.".format(outputDelta))            
            
            weight2 = backpropagation.updateWeight(hiddenLayer, weight2, outputDelta, learningRate)

            print("weight 2:")
            print(weight2)        
              
            hiddenDelta = backpropagation.backward(hiddenLayer, weight2, outputDelta)
            
            # bias is not a part of calculating the weights for the input
            hiddenDelta = hiddenDelta[0,0:hiddenNodes]

            print("hidden delta:")
            print(hiddenDelta)
            
            weight1 = backpropagation.updateWeight(inputLayer, weight1, hiddenDelta, learningRate)
            
            print("weight 1:")
            print(weight1)         

        
        
    print("Training complete.")
    
    return weight1, weight2
     


def predict(inputLayer, weight1, weight2):
    
    print("Starting prediction.")   
    
        
    error = np.zeros(1)
    
    for index in range(1):
       
        # run the forward function
        hiddenLayer, outputLayer = forward(inputLayer, weight1, weight2)

        print("Hidden layer:")
        print(hiddenLayer)
        print("Output layer:")
        print(outputLayer)
 
        # save the error
        error[index] = 1/2 * (outputLayer - target)**2
        print("Error: {:.8f}.".format(error[index]))
  
        
    print("Prediction complete.")
     





inputLayer = np.array([1, 1, 1])
target = 0
weight1 = np.array([[1,1,0.5],[-1,1,0.75]])
weight2 = np.array([-1,1,1])
hiddenNodes = 2
learningRate = 0.5
weight1, weight2 = train(inputLayer, target, weight1, weight2, hiddenNodes, 1, learningRate)

predict(inputLayer, weight1, weight2)



