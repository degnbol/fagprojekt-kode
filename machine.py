# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:33:39 2015

@author: Christian
"""

# importer ting
import os
import sys
import numpy as np

# ændr sti så vi kan finde hjælpefunktioner
projectPath = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(projectPath)

# importer hjælpefunktioner
import fileHandling
from logTransform import logTransform
from createInputLayer import createInputLayer
from weight import weight
from forward import forward

def train(path, hiddenNodes, iterations, learningRate):
    
    # indlæs sekvenser og hvor godt de binder til mhc proteinet
    sequence, meas = fileHandling.readHLA(path)
    
    # log transformer measurements så de er pænere tal
    meas = logTransform(meas)
    
    # længden af sekvensbiderne og antallet er mulige amino syrer. Der er 20 normale.
    sequenceLength = 9
    numOfAminoAcids = 20
    
    # lav vægt matrix med tilfældige værdier
    weightMatrix1 = weight(hiddenNodes, numOfAminoAcids * sequenceLength)
    weightMatrix2 = weight(1, hiddenNodes + 1) # plus 1 for bias
    
    # lav scrampled rækkefølge af sekvenserne
    indexes = np.arange(len(sequence))
    indexes = np.random.shuffle(indexes)

    for i in range(iterations):    
        
        # find nuværende sekvens
        currentSequence = sequence[indexes[i]]
        
        # lav sekvens om til binær
        inputLayer = createInputLayer(currentSequence)
        
        # kør forward funktion med tilfældige vægt matricer
        outputLayer = forward(inputLayer, weightMatrix1, weightMatrix2)
        
        # backpropergation
        
    
    # gem vægt matricer
    fileHandling.saveMatrix(weightMatrix1, 'weightMatrix1')
    fileHandling.saveMatrix(weightMatrix2, 'weightMatrix2')
    
    
def predict(path, weightMatrix1, weightMatrix2):
    
    """

    ////////////
    forudsigelsesfunktion. brug read fasta og gør meget det samme som i train


    """    
    