# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 15:41:26 2015

@author: Christian
"""

import numpy as np

def readHumanData(path):
    
    sequence = []
    meas = np.empty(0)

    
    with open(path) as data:
        
        data.readline()        
        
        for line in data:
            l = line.split()
            if (l[0] == 'human'
            and l[1] == 'HLA-A-0201'
            and l[2] == '9'):
                sequence.append(l[4])
                meas = np.append(meas, l[6])
   
   
    return sequence, meas

