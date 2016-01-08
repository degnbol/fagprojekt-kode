# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:33:39 2015

@author: Christian, Marianne, Kathrine & Cecilia
"""

# importer ting
import os
import sys
import numpy as np

# change path to find other functions
projectPath = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(projectPath)

# importer hjælpefunktioner
import fileUtils
import logTransform
import sequenceUtils
from forward import forward


# set seed to be able to reproduce results
np.random.seed(1234)

limit = 500
syfLimit = 21
names = np.array(["gag", "pol", "vif", "vpr", "tat", "rev", "vpu", "env", "nef"])

# mhc epitopes
mhcSequences, mhcAffinities = fileUtils.readHLA("data/mhcSequences.txt")
mhcEpitopes = mhcSequences[mhcAffinities <= limit]

# complete hiv
hivProteins = fileUtils.readFasta("data/hivCodingSequences.txt")


# SMMPMBEC
smm0 = fileUtils.readColumn("data/smmpmbec.csv", 0, True)
smm1 = fileUtils.readColumn("data/smmpmbec.csv", 1, True)
smm2 = fileUtils.readColumn("data/smmpmbec.csv", 2, True)
smm3 = fileUtils.readColumn("data/smmpmbec.csv", 3, True)
index = smm3 <= limit
smm = [smm2[index], np.repeat(0, sum(index)), smm1[index]]
# replace names wih numbers
for name in names:
    # plus one to make it one indexed
    smm[1][smm0[index] == name] = np.where(name == names)[0] +1


# NetMHC
netmhc0 = fileUtils.readColumn("data/netmhc.csv", 0, True)
netmhc1 = fileUtils.readColumn("data/netmhc.csv", 1, True)
netmhc2 = fileUtils.readColumn("data/netmhc.csv", 2, True)
index = netmhc2 <= limit
netmhc = [np.repeat("SEQUENCE", sum(index)), np.repeat(0, sum(index)), netmhc1[index]]
# replace names wih numbers
for name in names:
    # plus one to make it one indexed
    netmhc[1][netmhc0[index] == name] = np.where(name == names)[0] +1
for i in range(len(netmhc[0])):
    # this is the position where the sequence is found
    name = names[int(netmhc[1][i]-1)]
    reference = np.logical_and(smm0 == name, smm1 == netmhc[2][i])
    netmhc[0][i] = smm2[reference][0]

# SYFPEITHI
syf0 = fileUtils.readColumn("data/syfpeithi.csv", 0, True)
syf1 = fileUtils.readColumn("data/syfpeithi.csv", 1, True)
syf2 = fileUtils.readColumn("data/syfpeithi.csv", 2, True)
syf3 = fileUtils.readColumn("data/syfpeithi.csv", 3, True)
index = syf3 >= syfLimit
syf = [syf2[index], np.repeat(0, sum(index)), syf1[index]]
# replace names wih numbers
for name in names:
    # plus one to make it one indexed
    syf[1][syf0[index] == name] = np.where(name == names)[0] +1

# epitopes
hivEpitopes = fileUtils.readPeptides("data/hivEpitopes.csv", 9)
# used to get the info on the hiv epitopes
epitopes = [[], [], []]
   


# machine
# all sequences
machine0 = []
# all affinity predictions
machine1 = []
# predicted epitopes
machine = [[], [], []]
weight1 = np.load("weight1.npy")
weight2 = np.load("weight2.npy")


# predict

allSequences = [[], [], [], [], [], [], [], [], []]

for proteinId in range(len(hivProteins)):  
    
    allSequences[proteinId] = sequenceUtils.openReadingFrames(hivProteins[proteinId])        
    
    for pos in range(len(allSequences[proteinId])):
        
        ## machine
        # save sequence
        machine0.append(allSequences[proteinId][pos])        
        
        # lav sekvens om til binær
        inputLayer = sequenceUtils.createInputLayer(allSequences[proteinId][pos])
        
        # forward
        outputLayer = forward(inputLayer, weight1, weight2)[1]
        outputLayer = logTransform.invTransform(outputLayer)                

        # save prediction
        machine1.append(outputLayer)        
        
        # save epitope
        if(outputLayer <= limit):
            # plus one, since both are zero indexed
            machine[0].append(allSequences[proteinId][pos])
            machine[1].append(proteinId + 1)
            machine[2].append(pos + 1)


        ## epitopes
        if(np.any(allSequences[proteinId][pos] == np.array(hivEpitopes))):
            epitopes[0].append(allSequences[proteinId][pos])
            epitopes[1].append(proteinId + 1)
            epitopes[2].append(pos + 1)
            
            
 
## first outlier
           
noPrediction = epitopes[0]
noPrediction = np.setdiff1d(noPrediction, machine[0])
noPrediction = np.setdiff1d(noPrediction, netmhc[0])
noPrediction = np.setdiff1d(noPrediction, smm[0])
noPrediction = np.setdiff1d(noPrediction, syf[0])
#print(noPrediction)
noInGag = np.intersect1d(allSequences[0], noPrediction)
#print(noInGag)
#print(np.where(noInGag == np.array(machine0)))
#print(machine1[19])
#print(np.where(noInGag == np.array(allSequences[0])))
#print(np.array(allSequences[0][19]))
noInNetmhc = np.logical_and(netmhc0 == 'gag', netmhc1 == 19+1)
#print(netmhc2[noInNetmhc])
#print(smm3[smm2 == noInGag])
#print(syf3[syf2 == noInGag])


## second outlier

wrongPrediction = machine[0]
wrongPrediction = np.setdiff1d(wrongPrediction, epitopes[0])
wrongPrediction = np.setdiff1d(wrongPrediction, netmhc[0])
wrongPrediction = np.setdiff1d(wrongPrediction, smm[0])
wrongPrediction = np.setdiff1d(wrongPrediction, syf[0])
#print(wrongPrediction)
wrongInPol = np.intersect1d(allSequences[1], wrongPrediction)
#print(wrongInPol)
#print(np.where(wrongInPol[0] == np.array(allSequences[1])))
#print(np.where(wrongInPol[1] == np.array(allSequences[1])))
wrongFirst = allSequences[1][58]
#print(wrongFirst)
wrongInMachine = np.where(wrongFirst == np.array(machine0))
#print(wrongInMachine)
#print(machine1[wrongInMachine[0][0]])
#print(np.where(wrongFirst == np.array(allSequences[1])))
#print(np.array(allSequences[1][58]))
noInNetmhc = np.logical_and(netmhc0 == 'gag', netmhc1 == 58+1)
#print(netmhc2[noInNetmhc])
#print(smm3[smm2 == wrongFirst])
#print(syf3[syf2 == wrongFirst])



# mhc epitopes in HIV that hasn't already been found
hiv = []
for i in range(len(names)):
    hiv = hiv + allSequences[i]
newEpitopes = np.intersect1d(mhcEpitopes, np.setdiff1d(hiv, hivEpitopes))
#print(newEpitopes)
newPredictions = np.intersect1d(newEpitopes, machine[0])
#print(newPredictions)












