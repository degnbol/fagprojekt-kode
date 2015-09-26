# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:33:39 2015

@author: Christian
"""

import os
import matplotlib.pyplot as plot

os.chdir('/Users/Christian/OneDrive/Fagprojekt/fagprojekt-kode')
from readHLA import readHLA
from logTransform import logTransform

sequence, meas = readHLA('data.txt')
meas = logTransform(meas)

plot.hist(meas)