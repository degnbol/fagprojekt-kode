# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 15:41:26 2015

@author: Christian
"""

def readHLA(path):
    
    sequence = []
    meas = []
    
    with open(path) as data:
        
        for line in data:
            l = line.split()
            if (l[0] == 'human'
            and l[1] == 'HLA-A-0201'
            and l[2] == '9'):
                sequence.append(l[4])
                meas.append(float(l[6]))
    
    return sequence, meas


def readFasta(path):
    
    sequence = []
    
    """
    
    /////// læs fasta fil fra path. Brug sekvenser mellem linjerne der starter
    med >
    bruge alle læsesrammer med 9 aminosyrer fra denne samlede sekvens
    
    """
    
    return sequence
    
    
def saveMatrix(matrix, name):
    """
    
    /////// gem matrix i fil med navnet name på simpleste måde
    
    """