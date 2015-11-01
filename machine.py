# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:33:39 2015

@author: Christian, Marianne, Kathrine & Cecilia
"""

# import packages
import os
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plot

# change path so we can find help functions
projectPath = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(projectPath)

# import help functions
import fileUtils
from logTransform import logTransform
import sequenceUtils
from weight import weight
from forward import forward
import backpropagation

# argument parsing
descriptionOfArguments = 'ANN that finds correlations between peptides and their measurements.'
helpForPath = 'a path to a file with peptide data'
helpForWeightPath = 'two paths that weights are saved to when training or loaded from when predicting (default is "weight1.npy" and "weight2.npy")'
helpForTrain = 'train the machine with the data (default: prediction on the data)'
helpForHiddenNodes = 'number of hidden nodes when training (default is 8)'
helpForEpocs = 'number of iterations through the sets when training (default is 10)'
helpForLearningRate = 'the step size of changes when training (default is 0.01)'
helpForProceed = 'continue on the weight matrices in the specified files (default is starting with random weights)'

parser = argparse.ArgumentParser(description = descriptionOfArguments)
parser.add_argument('path', help = helpForPath)
parser.add_argument('--weightPath', nargs = 2, default = ['weight1.npy', 'weight2.npy'], help = helpForWeightPath)
parser.add_argument('-t', '--train', action = 'store_true', help = helpForTrain)
parser.add_argument('--hiddenNodes', default = 8, type = int, help = helpForHiddenNodes)
parser.add_argument('--epocs', default = 10, type = int, help = helpForEpocs)
parser.add_argument('--learningRate', default = 0.001, type = float, help = helpForLearningRate)
parser.add_argument('--proceed', action = 'store_true', help = helpForProceed)
args = parser.parse_args()
#args = parser.parse_args("mhcSequences.txt -t --epocs 100 --learningRate 0.00005 --proceed".split())

# takes the path for the HLA file as input 
def train(path, weightPath1, weightPath2, hiddenNodes, epocs, learningRate, proceed):
    
    # read sequences and their measured binding affinities
    allSequences, allTargets = fileUtils.readHLA(path)
    
    # log transformer measurements så de er pænere tal
    allTargets = logTransform(allTargets)
    
    # divide the data into training set and validation set
    numOfSequences = len(allSequences)
    indexes = np.arange(numOfSequences)
    np.random.shuffle(indexes)
    numOfTrain = (int) (numOfSequences * 0.8) # 80 % is for training
    trainSequence = allSequences[indexes[0:numOfTrain]]
    trainTarget = allTargets[indexes[0:numOfTrain]]
    valSequence = allSequences[indexes[numOfTrain:numOfSequences]]
    valTarget = allTargets[indexes[numOfTrain:numOfSequences]]
    
    trainError = np.zeros(epocs)
    valError = np.zeros(epocs)
    
    # længden af sekvensbiderne og antallet er mulige aminosyrer. Der er 20 normale.
    mer = 9
    numOfAminoAcids = 20
    
    # create weight matrix with random values or load the files
    if(proceed):
        weight1 = np.load(weightPath1)
        weight2 = np.load(weightPath2)
    else:
        weight1 = weight(hiddenNodes, numOfAminoAcids * mer + 1) # plus 1 for bias
        weight2 = weight(1, hiddenNodes + 1) # plus 1 for bias    
    
    weights1 = []    
    weights2 = []    
    weights1.append(weight1)
    weights2.append(weight2)
    
    print("Starting training.")
    print("Errors from training set:")   
    
    # train on training set
    for epoc in range(epocs):
        
        # make scrampled order of sequences
        indexes = np.arange(numOfTrain)
        np.random.shuffle(indexes)

        error = np.zeros(numOfTrain)
        
        for index in indexes:
            
            # convert peptide sequence to quasi-binary
            inputLayer = sequenceUtils.createInputLayer(trainSequence[index])
            
            # run the forward function
            hiddenLayer, outputLayer = forward(inputLayer, weight1, weight2)

            # save the error
            error[index] = 1/2 * (outputLayer - trainTarget[index])**2

            # backpropagation            
            errorDelta = outputLayer - trainTarget[index]
            
            outputDelta = backpropagation.backward(hiddenLayer, 1, errorDelta)
            
            weight2 = backpropagation.updateWeight(hiddenLayer, weight2, outputDelta, learningRate)
            
            hiddenDelta = backpropagation.backward(inputLayer, weight2, outputDelta)
            
            weight1 = backpropagation.updateWeight(inputLayer, weight1, hiddenDelta, learningRate)
            
            
        # save weights
        weights1.append(weight1)
        weights2.append(weight2)        
        
        trainError[epoc] = error.mean()
        
        if(epoc % 10 == 0):           
            percent = (int) (epoc/epocs*100)
            print("Error: {}. {}% complete.".format(trainError[epoc], percent))
        
        
    print("Training set complete.")
    print("Errors from validation set:")
    
    #validate on validation set
    for epoc in range(epocs):
        
        error = np.zeros(numOfSequences - numOfTrain)
        
        for index in range(numOfSequences - numOfTrain):
            
            # convert peptide sequence to quasi-binary
            inputLayer = sequenceUtils.createInputLayer(valSequence[index])
            
            # run the forward function
            hiddenLayer, outputLayer = forward(inputLayer, weights1[epoc], weights2[epoc])

            # save the error
            error[index] = 1/2 * (outputLayer - valTarget[index])**2

            
        valError[epoc] = error.mean()
        
        if(epoc % 10 == 0):           
            percent = (int) (epoc/epocs*100)
            print("Error: {}. {}% complete.".format(valError[epoc], percent))
    
    
    print("Validation set complete.")
    
    # plot
    plot.plot(trainError, label = "Training set")
    plot.plot(valError, label = "Validation set")
    plot.legend()
    plot.xlabel("epoc")
    plot.ylabel("error")
    plot.show()
        
    # save the best weight matrices
    best = (int) (np.where(valError == min(valError))[0])
    print("The minimum error of the validation set is at epoc {}".format(best))
    np.save(weightPath1, weights1[best])
    np.save(weightPath2, weights2[best])
    
    
    
# tager input path for stien til en fasta fil
def predict(path, weightPath1, weightPath2):
    
    # read files
    proteins = fileUtils.readFasta(path)
    weight1 = np.load(weightPath1)
    weight2 = np.load(weightPath2)
    
    for protein in proteins:  
        
        sequences = sequenceUtils.openReadingFrames(protein)        
        
        for sequence in sequences:
            
             # lav sekvens om til binær
            inputLayer = sequenceUtils.createInputLayer(sequence)
            
            # kør forward funktion med vægt matricer
            outputLayer = forward(inputLayer, weight1, weight2)[1]
        
            print(sequence, outputLayer)



if(args.train):
    train(args.path, args.weightPath[0], args.weightPath[1], args.hiddenNodes, args.epocs, args.learningRate, args.proceed)
else:
    predict(args.path, args.weightPath[0], args.weightPath[1])

