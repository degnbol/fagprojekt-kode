# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 01:13:10 2015

@author: Christian
"""

import numpy as np

def createInputLayer(sequence):
    
    peptideAlphabet = np.array(['A','C','D','E','F','G','H',
    'I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
    
    sequenceLen = len(sequence)
    alphabetLen = len(peptideAlphabet)    
    
    inputLayer = np.zeros(sequenceLen * alphabetLen + 1)
    
    for i in range(sequenceLen):
        binary = peptideAlphabet == sequence[i]
        inputLayer[i*alphabetLen:(i+1)*alphabetLen] = binary
    
    # bias    
    inputLayer[len(inputLayer)-1] = 1
    
    return inputLayer
    

# lav alle mulige kombinationer af 9 lange sekvenser fra en større sekvens
def openReadingFrames(sequence):
        
    frameLength = 9
    openFrames = []

    for i in range(len(sequence) - frameLength):
        openFrames.append(sequence[i:i+frameLength])

    return openFrames