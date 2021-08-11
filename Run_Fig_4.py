#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 4. 
Fraction of nPE neurons depends on SOM and VIP neuron inputs.
"""

# %% Import  & settings

import numpy as np
from Model_Network import Neurons, Network, Stimulation, Simulation, SaveNetworkPara 
from Model_Network import SaveData, Learning, RunStaticNetwork, RunPlasticNetwork
from Plot_NetResults import HeatmapTest

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% Figure 4

"""
Please note:    
    To reproduce all panels of the figure, 
    vary parameter 'num_SOMs_visual' (see below).
"""

folder = 'Fig_4'
num_SOMs_visual = 10 # in percent (in paper: 10, 50 and 90)
np.random.seed(186)

### Define neuron and network parameters
NeuPar = Neurons()

wPP, wPE, wDS = dtype(-0.5), dtype(2.5), dtype(-5.0)
NetPar = Network(NeuPar, wPP=wPP, wPE=wPE, wDS=wDS)

### Define simulation parameters & recording options
SimPar_test = Simulation()
SimPar = Simulation(dt=0.2)
SavePar = SaveData()

### Define plasticity parameter
LearnPar = Learning(NeuPar)


# #####################
# ##### Learning ######
# #####################

### Define input parameters
stim_max, SD = dtype(100), dtype(5)
Xternal = dtype([400,2,2,2,0])
num_stim = np.int32(3600)
VS = num_SOMs_visual/100
VV = np.round(1 - VS,1)

StimPar = Stimulation(NeuPar, NetPar, SD, num_stim=num_stim, stim_max=stim_max, 
                      VS=VS, VV=VV, flg_test=0, Xternal=Xternal)

### Run simulation
fln = 'After_' + str(num_SOMs_visual)
RunPlasticNetwork(NeuPar, NetPar, StimPar, SimPar, LearnPar, SavePar, folder, fln)
SaveNetworkPara(NeuPar, NetPar, StimPar, LearnPar, folder, fln)


#####################
# After plasticity ##
#####################

### Define input parameters
StimPar_test = Stimulation(NeuPar, NetPar, SD, None, stim_max, Xternal=Xternal, MM_factor = dtype(0.5))
StimPar_test.neurons_visual = StimPar.neurons_visual
StimPar_test.neurons_motor = StimPar.neurons_motor

### Run simulations
fln = 'After_Test_' + str(num_SOMs_visual)
RunStaticNetwork(NeuPar, NetPar, StimPar_test, SimPar_test, folder, fln)

### Analyse & plot network
HeatmapTest(NeuPar, NeuPar.NCells[0], folder, fln, MM_factor=dtype(0.5))

