# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:33:39 2015

@author: Christian, Marianne, Kathrine & Cecilia
"""

# importer ting
import os
import sys
import numpy as np
import argparse

# ændr sti så vi kan finde hjælpefunktioner
projectPath = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(projectPath)

# importer hjælpefunktioner
import fileUtils
from logTransform import logTransform
import sequenceUtils
from weight import weight
from forward import forward
import backpropagation

# argument parsing
descriptionOfArguments = 'ANN that finds correlations between peptides and their measurements.'
helpForPath = 'a path to a file with peptide data'
helpForWeightPath = 'two paths that weights are saved to when training or loaded from when predicting (default is "weight1" and "weight2")'
helpForTrain = 'train the machine with the data (default: prediction on the data)'
helpForHiddenNodes = 'number of hidden nodes when training (default is 8)'
helpForIterations = 'number of iterations when training (default is 10)'
helpForLearningRate = 'the step size of changes when training (default is 0.01)'

parser = argparse.ArgumentParser(description = descriptionOfArguments)
parser.add_argument('path', help = helpForPath)
parser.add_argument('--weightPath', nargs = 2, default = ['weight1', 'weight2'], help = helpForWeightPath)
parser.add_argument('-t', '--train', action = 'store_true', help = helpForTrain)
parser.add_argument('--hiddenNodes', default = 8, type = int, help = helpForHiddenNodes)
parser.add_argument('--iterations', default = 10, type = int, help = helpForIterations)
parser.add_argument('--learningRate', default = 0.001, type = float, help = helpForLearningRate)
args = parser.parse_args()

# tager input path for stien til en fil med HLA info
def train(path, weightPath1, weightPath2, hiddenNodes, iterations, learningRate):
    
    # indlæs sekvenser og hvor godt de binder til mhc proteinet
    sequence, target = fileUtils.readHLA(path)
    
    # log transformer measurements så de er pænere tal
    target = logTransform(target)
    
    # længden af sekvensbiderne og antallet er mulige amino syrer. Der er 20 normale.
    sequenceLength = 9
    numOfAminoAcids = 20
    
    # lav vægt matrix med tilfældige værdier
    weight1 = weight(hiddenNodes, numOfAminoAcids * sequenceLength + 1) # plus 1 for bias
    weight2 = weight(1, hiddenNodes + 1) # plus 1 for bias
    
    for iteration in range(iterations):
        
        # lav scrampled rækkefølge af sekvenserne
        indexes = np.arange(len(sequence))
        np.random.shuffle(indexes)

        error = np.zeros(len(sequence))
        
        for index in indexes:
            
            # convert peptide sequence to quasi-binary
            inputLayer = sequenceUtils.createInputLayer(sequence[index])
            
            # run the forward function. returns the hidden layer, the output layer before and after activation
            preHiddenLayer, hiddenLayer, preOutputLayer, outputLayer = forward(inputLayer, weight1, weight2)

            # save the error
            error[index] = 1/2 * (outputLayer - target[index])**2
            errorDelta = outputLayer - target[index]
        
            # backpropagation
            outputDelta = backpropagation.backward(hiddenLayer, 1, errorDelta)
            
            weight2 = backpropagation.updateWeight(hiddenLayer, weight2, outputDelta, learningRate)
            
            hiddenDelta = backpropagation.backward(inputLayer, weight2, outputDelta)
            
            weight1 = backpropagation.updateWeight(inputLayer, weight1, hiddenDelta, learningRate)
            
        
        print(error.mean())
        
    # gem vægt matricer
    fileUtils.saveMatrix(weight1, weightPath1)
    fileUtils.saveMatrix(weight2, weightPath2)
    
    
    
# tager input path for stien til en fasta fil
def predict(path, weightPath1, weightPath2): 
    
    # read files
    proteins = fileUtils.readFasta(path)
    weight1 = fileUtils.loadMatrix(weightPath1)
    weight2 = fileUtils.loadMatrix(weightPath2)
    
    for protein in proteins:  
        
        sequences = sequenceUtils.openReadingFrames(protein)        
        
        for sequence in sequences:
            
             # lav sekvens om til binær
            inputLayer = sequenceUtils.createInputLayer(sequence)
            
            # kør forward funktion med vægt matricer
            outputLayer = forward(inputLayer, weight1, weight2)[1]
        
            print(sequence, outputLayer)



if(args.train):
    train(args.path, args.weightPath[0], args.weightPath[1], args.hiddenNodes, args.iterations, args.learningRate)
else:
    predict(args.path, args.weightPath[0], args.weightPath[1])


'''


lav cross validation

1 fil -> 5 træningssæt og tilsvarende 5 test sæt




'''