# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 16:59:19 2015

@author: Christian
"""

def logTransform(meas):
    
    return(meas[meas < 500])