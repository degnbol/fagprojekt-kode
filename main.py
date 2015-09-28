# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:33:39 2015

@author: Christian
"""

import os
import sys
import matplotlib.pyplot as plot

projectPath = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(projectPath)

from readHLA import readHLA
from logTransform import logTransform
from peptideToBinary import peptideToBinary

sequence, meas = readHLA('data.txt')
meas = logTransform(meas)

plot.hist(meas)

inputLayer = peptideToBinary(sequence[0])


# lav vægt matrix med tilfældige værdier

# beregn foreløbig hidden layer vha. w matrix og binær sekvens

# brug blød threshold på hidden layer: y = 1/(1 + exp(-x))

# beregn resultat med hidden layer og w matrix 2 (som blot er liggende vektor)

# brug blød threshold på resultat

