# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 01:13:10 2015

@author: Christian
"""

import numpy as np

def peptideToBinary(sequence):
    
    peptideAlphabet = np.array(['A','C','D','E','F','G','H',
    'I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
    
    sequenceLen = len(sequence)
    alphabetLen = len(peptideAlphabet)    
    
    binarySequence = np.zeros(sequenceLen * alphabetLen, int)
    
    for i in range(sequenceLen):
        binary = peptideAlphabet == sequence[i]
        binarySequence[i*alphabetLen:(i+1)*alphabetLen] = binary
    
    return binarySequence
    
