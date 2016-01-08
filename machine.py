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
import matplotlib.pyplot as pyplot

# change path to find other functions
projectPath = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(projectPath)

# importer hjælpefunktioner
import fileUtils
import logTransform
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
helpForEpochs = 'number of iterations through the sets when training (default is 500)'
helpForLearningRate = 'the step size of changes when training (default is 0.01)'
helpForProceed = 'continue on the weight matrices in the specified files (default is starting with random weights)'
helpForMethod = 'the method for loading data in prediction. Default is fasta. Other options are raw and hla.'

parser = argparse.ArgumentParser(description = descriptionOfArguments)
parser.add_argument('path', help = helpForPath)
parser.add_argument('--weightPath', nargs = 2, default = ['weight1.npy', 'weight2.npy'], help = helpForWeightPath)
parser.add_argument('-t', '--train', action = 'store_true', help = helpForTrain)
parser.add_argument('--hiddenNodes', default = 8, type = int, help = helpForHiddenNodes)
parser.add_argument('--epochs', default = 500, type = int, help = helpForEpochs)
parser.add_argument('--learningRate', default = 0.01, type = float, help = helpForLearningRate)
parser.add_argument('--proceed', action = 'store_true', help = helpForProceed)
parser.add_argument('--method', default = 'fasta', help = helpForMethod)

# train with this line uncommented
#args = parser.parse_args("data/mhcSequences.txt -t".split())

# predict with this line uncommented
#args = parser.parse_args("data/hivCodingSequences.txt".split())

# have the program be run from terminal with this line uncommented
args = parser.parse_args()


# set seed to be able to reproduce results
np.random.seed(1234)


# tager input path for stien til en fil med HLA info
def train(path, weightPath1, weightPath2, hiddenNodes, epochs, learningRate, proceed):
    
    # read sequences and their measured binding affinities
    allSequences, allTargets = fileUtils.readHLA(path)  
    
    # log transform the data to fit between 0 and 1
    allTargets = logTransform.transform(allTargets)    
    
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
    bestEpoch = 0
    
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
            bestEpoch = epoch
        
        
        if(epoch % 10 == 0):           
            percent = (int) (epoch/epochs*100)
            print("Training error: {:.8f}. Validation error: {:.8f}. {:2}% complete."
            .format(trainError[epoch], valError[epoch], percent))
        
        
    print("Training and validation complete.")
    
    
    # plot error
    pyplot.plot(trainError, label = "Training set")
    pyplot.plot(valError, label = "Validation set")
    pyplot.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    pyplot.xlabel("epoch")
    pyplot.ylabel("error")
    pyplot.title("Validation")
    pyplot.savefig('validation.png', bbox_inches='tight')    
    pyplot.show()
            
    # save the best weight matrices
    np.save(weightPath1, bestWeight1)
    np.save(weightPath2, bestWeight2)
    print("The minimum error of the validation set is at epoch {}. The validation error is {}."
    .format(bestEpoch, bestError))
    
    #evaluation   
    print("Predicting on evaluation set.")

    for index in range(len(evalSequence)):
        
        # convert peptide sequence to quasi-binary
        inputLayer = sequenceUtils.createInputLayer(evalSequence[index])
        
        # run the forward function
        hiddenLayer, outputLayer = forward(inputLayer, bestWeight1, bestWeight2)
        
        evalPrediction[index] = outputLayer


    # plot comparison of prediction and target for evaluation set
    pyplot.plot(evalTarget, evalPrediction, '.')
    pyplot.xlabel("target")
    pyplot.ylabel("prediction")
    pyplot.title("Evaluation")
    pyplot.savefig('evaluationLog.png', bbox_inches='tight')    
    pyplot.show()
    
    # how correlated is it?
    corr = np.corrcoef(evalTarget, evalPrediction)[1,0]
    print("The Pearson correlation coefficient is {}.".format(corr))
    
    # plot comparison again, now inverse log transfomed back but with a logarithmic scale
    evalPrediction = logTransform.invTransform(evalPrediction)
    evalTarget = logTransform.invTransform(evalTarget)
    pyplot.axes().set_xscale('log')
    pyplot.axes().set_yscale('log')
    pyplot.plot(evalTarget, evalPrediction, '.')
    pyplot.xlabel("target")
    pyplot.ylabel("prediction")
    pyplot.title("Evaluation")
    pyplot.savefig('evaluation.png', bbox_inches='tight')    
    pyplot.show()
    

 
# tager input path for stien til en fasta fil
def predict(path, weightPath1, weightPath2, method):
    
    predictionPath = "predictions.txt"
    
    predictions = []    
    limit = 500
    
    # load
    weight1 = np.load(weightPath1)
    weight2 = np.load(weightPath2)
    
    method = method.lower()
    if(method == 'fasta'):
        proteins = fileUtils.readFasta(path)
        
        for proteinId in range(len(proteins)):  
            
            sequences = sequenceUtils.openReadingFrames(proteins[proteinId])        
            
            for pos in range(len(sequences)):
                
                 # lav sekvens om til binær
                inputLayer = sequenceUtils.createInputLayer(sequences[pos])
                
                # forward
                outputLayer = forward(inputLayer, weight1, weight2)[1]
                outputLayer = logTransform.invTransform(outputLayer)                
                
                if(outputLayer <= limit):
                    # plus one, since both are zero indexed
                    predictions.append([proteinId + 1, pos + 1])
    
    
                
    np.savetxt(predictionPath, np.array(predictions), fmt = '%d', delimiter = '\t')
    print("There is {} predicted epitopes.".format(len(predictions)))   



if(args.train):
    train(args.path, args.weightPath[0], args.weightPath[1], args.hiddenNodes, args.epochs, args.learningRate, args.proceed)
else:
    predict(args.path, args.weightPath[0], args.weightPath[1], args.method)

