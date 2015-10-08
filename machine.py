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
from logTransform import logTransform
import sequenceUtils
from weight import weight
from forward import forward
import backpropagation



# tager input path for stien til en fil med HLA info
def train(path, hiddenNodes, iterations, learningRate):
    
    # indlæs sekvenser og hvor godt de binder til mhc proteinet
    sequence, meas = fileUtils.readHLA(path)
    
    # log transformer measurements så de er pænere tal
    meas = logTransform(meas)
    
    # længden af sekvensbiderne og antallet er mulige amino syrer. Der er 20 normale.
    sequenceLength = 9
    numOfAminoAcids = 20
    
    # lav vægt matrix med tilfældige værdier
    weightMatrix1 = weight(hiddenNodes, numOfAminoAcids * sequenceLength + 1) # plus 1 for bias
    weightMatrix2 = weight(1, hiddenNodes + 1) # plus 1 for bias
    
    # lav scrampled rækkefølge af sekvenserne
    indexes = np.arange(len(sequence))
    np.random.shuffle(indexes)

    for i in range(iterations):
        
        # next index
        index = indexes[i % len(indexes)]
        
        # find nuværende sekvens. Med modulus sørges der for at vi bliver indenfor mængden
        currentSequence = sequence[index]
        
        # lav sekvens om til binær
        inputLayer = sequenceUtils.createInputLayer(currentSequence)
        
        # kør forward funktion med vægt matricer
        hiddenLayer, outputLayer = forward(inputLayer, weightMatrix1, weightMatrix2)
        
        # backpropergation
        hiddenDelta, outputDelta = backpropagation.backward(hiddenLayer, weightMatrix2, outputLayer, meas[index])
        
        weightMatrix1, weightMatrix2 = backpropagation.updateWeight(inputLayer, weightMatrix1, hiddenLayer, weightMatrix2, hiddenDelta, outputDelta, learningRate)
    
    # gem vægt matricer
    fileUtils.saveMatrix(weightMatrix1, 'weightMatrix1')
    fileUtils.saveMatrix(weightMatrix2, 'weightMatrix2')
    
    
    
# tager input path for stien til en fasta fil
def predict(path, weightMatrix1, weightMatrix2): 
    
    # indlæs sekvenser
    proteins = fileUtils.readFasta(path)
    
    for protein in proteins:  
        
        sequences = sequenceUtils.openReadingFrames(protein)        
        
        for sequence in sequences:
            
             # lav sekvens om til binær
            inputLayer = sequenceUtils.createInputLayer(sequence)
            
            # kør forward funktion med vægt matricer
            outputLayer = forward(inputLayer, weightMatrix1, weightMatrix2)[1]
        
            print(sequence, outputLayer)

