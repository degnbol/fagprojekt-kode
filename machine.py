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
import matplotlib.pyplot as plot

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
helpForWeightPath = 'two paths that weights are saved to when training or loaded from when predicting (default is "weight1.npy" and "weight2.npy")'
helpForTrain = 'train the machine with the data (default: prediction on the data)'
helpForHiddenNodes = 'number of hidden nodes when training (default is 8)'
helpForEpochs = 'number of iterations through the sets when training (default is 10)'
helpForLearningRate = 'the step size of changes when training (default is 0.01)'
helpForProceed = 'continue on the weight matrices in the specified files (default is starting with random weights)'

parser = argparse.ArgumentParser(description = descriptionOfArguments)
parser.add_argument('path', help = helpForPath)
parser.add_argument('--weightPath', nargs = 2, default = ['weight1.npy', 'weight2.npy'], help = helpForWeightPath)
parser.add_argument('-t', '--train', action = 'store_true', help = helpForTrain)
parser.add_argument('--hiddenNodes', default = 8, type = int, help = helpForHiddenNodes)
parser.add_argument('--epochs', default = 10, type = int, help = helpForEpochs)
parser.add_argument('--learningRate', default = 0.001, type = float, help = helpForLearningRate)
parser.add_argument('--proceed', action = 'store_true', help = helpForProceed)
args = parser.parse_args("mhcSequences.txt -t --epochs 500 --learningRate 0.01".split())


# set seed to be able to reproduce results
np.random.seeed(1234)


# tager input path for stien til en fil med HLA info
def train(path, weightPath1, weightPath2, hiddenNodes, epochs, learningRate, proceed):
    
    # read sequences and their measured binding affinities
    allSequences, allTargets = fileUtils.readHLA(path)
    
    # log transformer measurements så de er pænere tal
    allTargets = logTransform(allTargets)
      
    
    # divide the data into training set, validation set and evaluation set
    numOfSequences = len(allSequences)    
    indexes = np.arange(numOfSequences)
    np.random.shuffle(indexes)
    numOfTrain = (int) (numOfSequences * 0.7) # 70 % is for training
    trainSequence = allSequences[indexes[0:numOfTrain]]
    trainTarget = allTargets[indexes[0:numOfTrain]]    
    numOfVal = (int) (numOfSequences * 0.2) # 20 % is for vaidation
    valSequence = allSequences[indexes[numOfTrain:(numOfTrain + numOfVal)]]
    valTarget = allTargets[indexes[numOfTrain:(numOfTrain + numOfVal)]]
    evalSequence = allSequences[indexes[(numOfTrain + numOfVal):numOfSequences]]
    evalTarget = allTargets[indexes[(numOfTrain + numOfVal):numOfSequences]]
    evalPrediction = np.zeros(len(evalSequence))
    
    trainError = np.zeros(epochs)   
    valError = np.zeros(epochs)
    
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
    
    bestWeight1 = weight1
    bestWeight2 = weight2
    bestError = 999 # just a large number so any validation will be better
    
    print("Starting training and validation.")   
    
    for epoch in range(epochs):
        
        # train on training set
        
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
            errorDelta = outputLayer - trainTarget[index]
            
            # backpropagation
            outputDelta = backpropagation.backward(outputLayer, 1, errorDelta)
            
            weight2 = backpropagation.updateWeight(hiddenLayer, weight2, outputDelta, learningRate)
              
            hiddenDelta = backpropagation.backward(hiddenLayer, weight2, outputDelta)
            
            # bias is not a part of calculating the weights for the input
            hiddenDelta = hiddenDelta[0,0:hiddenNodes]
            
            weight1 = backpropagation.updateWeight(inputLayer, weight1, hiddenDelta, learningRate)


        trainError[epoch] = error.mean()
        
        
        
        # validation
        
        error = np.zeros(numOfVal)
        
        for index in range(numOfVal):
            
            # convert peptide sequence to quasi-binary
            inputLayer = sequenceUtils.createInputLayer(valSequence[index])
            
            # run the forward function
            hiddenLayer, outputLayer = forward(inputLayer, weight1, weight2)

            # save the error
            error[index] = 1/2 * (outputLayer - valTarget[index])**2

            
        valError[epoch] = error.mean()
        

        # find the best weight matrices so far
        if(valError[epoch] < bestError):
            bestWeight1 = weight1
            bestWeight2 = weight2
            bestError = valError[epoch]
        
        
        if(epoch % 10 == 0):           
            percent = (int) (epoch/epochs*100)
            print("Error: {:.8f}. {:2}% complete.".format(valError[epoch], percent))
        
        if(epoch % 10 == 0):           
            percent = (int) (epoch/epochs*100)
            print("Error: {:.8f}. {:2}% complete.".format(trainError[epoch], percent))
        
        
    print("Training and validation complete.")
    
    
    # plot error
    plot.plot(trainError, label = "Training set")
    plot.plot(valError, label = "Validation set")
    plot.legend()
    plot.xlabel("epoch")
    plot.ylabel("error")
    plot.title("Validation")
    plot.show()
    plot.savefig('validation.png', bbox_inches='tight')
        
    # save the bet weight matrices
    best = (int) (np.where(valError == min(valError))[0])
    print("The minimum error of the validation set is at epoch {}".format(best))
    weight1 = weights1[best]
    weight2 = weights2[best]
    np.save(weightPath1, weight1)
    np.save(weightPath2, weight2)
    
    #evaluation   
    print("Predicting on evaluation set.")

    for index in range(len(evalSequence)):
        
        # convert peptide sequence to quasi-binary
        inputLayer = sequenceUtils.createInputLayer(evalSequence[index])
        
        # run the forward function
        hiddenLayer, outputLayer = forward(inputLayer, weight1, weight2)
        
        evalPrediction[index] = outputLayer


    # plot comparison of prediction and target for evaluation set
    plot.plot(evalTarget, evalPrediction, '.')
    plot.xlabel("target")
    plot.ylabel("prediction")
    plot.title("Evaluation")
    plot.show()
    plot.savefig('evaluation.png', bbox_inches='tight')
    

 
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
    train(args.path, args.weightPath[0], args.weightPath[1], args.hiddenNodes, args.epochs, args.learningRate, args.proceed)
else:
    predict(args.path, args.weightPath[0], args.weightPath[1])

