# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:33:39 2015

@author: Christian
"""

# importer ting
import os
import sys
import matplotlib.pyplot as plot

# ændr sti så vi kan finde hjælpefunktioner
projectPath = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(projectPath)

# importer hjælpefunktioner
from readHLA import readHLA
from logTransform import logTransform
from createInputLayer import createInputLayer
from forward import forward

# indlæs sekvenser og hvor godt de binder til mhc proteinet
sequence, meas = readHLA('data.txt')

# log transformer measurements så de er pænere tal
meas = logTransform(meas)

# bare for sjov, så man kan se fordelingen af binding affinities
plot.hist(meas, bins = 30)

# lav en tilfældig sekvens om til binær
inputLayer = createInputLayer(sequence[0])

# vælg antal nodes i hidden layer
numOfHiddenNodes = 8

# lav vægt matrix med tilfældige værdier

weightMatrix1 = [] # = random(numOfHiddenNodes, len(inputLayer))
weightMatrix2 = [] # = random(1, numOfHiddenNodes + 1) # plus 1 for bias

# kør forward funktion med tilfældige vægt matricer
outputLayer = forward(inputLayer, weightMatrix1, weightMatrix2)

print(outputLayer)