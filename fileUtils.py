# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 15:41:26 2015

@author: Christian
"""

import numpy as np

def readHLA(path):
    
    sequence = []
    meas = []
    
    with open(path) as data:
        
        for line in data:
            l = line.split()
            
            # tilføj info fra linjer, der lever op til kriterier
            if (l[0] == 'human'
            and l[1] == 'HLA-A-0201'
            and l[2] == '9'):
                sequence.append(l[4])
                meas.append(float(l[6]))
    
    return np.array(sequence), np.array(meas)



def readFasta(path):
    
    protein = []
    
    with open(path) as data:
           
        for line in data:
            if(line[0] == '>'):
                # gør klar til et nyt protein
                protein.append('')
            else:
                # gem sekvensdata på seneste protein
                protein[len(protein)-1] += line.strip()
    
    return protein



def readPeptides(path, mer):
    
    sequence = []
    
    with open(path) as data:
        

        for line in data:
            
            line = line.upper()
            
            if (length(line) == 9
            sequence.append(line)
            
        else:
            
            
            )

    return sequence
    