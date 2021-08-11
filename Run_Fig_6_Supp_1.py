#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 6–Figure supplement 1. 
Learning nPE neurons by biologically plausible learning rules 
in networks without visual input at the soma of PCs.
"""

# %% Import  & settings

import numpy as np
from Model_Network import Neurons, Network, Stimulation, Simulation, SaveNetworkPara 
from Model_Network import SaveData, Learning, RunStaticNetwork, RunPlasticNetwork
from Plot_NetResults import HeatmapTest, Plot_Currents2PC, Plot_Convergence

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% Figure 6 - Supplement Figure 1

folder = 'Fig_6_Supp_1'

### Define neuron and network parameters
NeuPar = Neurons()

wPP, wPE, wDS = dtype(-1.5), dtype(1.2), dtype(-5.0)
NetPar = Network(NeuPar, wPP=wPP, wPE=wPE, wDS=wDS)

### Define simulation parameters & recording options
SimPar_test = Simulation()
SimPar = Simulation(dt=0.2)
SavePar = SaveData()

### Define plasticity parameter
LearnPar = Learning(NeuPar, flagRule=0)


#####################
# Before plasticity #
#####################

## Define input parameters
stim_max, SD = dtype(100), dtype(5)
Xternal = dtype([400,2,2,2,0])
num_stim = np.int32(10)

StimPar_test = Stimulation(NeuPar, NetPar, SD, None, stim_max, VE=0, Xternal=Xternal)
StimPar_test_long = Stimulation(NeuPar, NetPar, SD, num_stim, stim_max, VE=0, flg_test=0, Xternal=Xternal)

### Run simulations
RunStaticNetwork(NeuPar, NetPar, StimPar_test_long, SimPar_test, folder, 'Before_Balance')
RunStaticNetwork(NeuPar, NetPar, StimPar_test, SimPar_test, folder, 'Before_Test')
SaveNetworkPara(NeuPar, NetPar, StimPar_test_long, None, folder,'Before')

### Analyse & plot network
Plot_Currents2PC(NeuPar, NetPar, StimPar_test_long, SimPar_test, folder, 'Before_Balance')
HeatmapTest(NeuPar, NeuPar.NCells[0], folder, 'Before_Test')


# #####################
# ##### Learning ######
# #####################

### Define input parameters
num_stim = np.int32(5400)

StimPar = Stimulation(NeuPar, NetPar, SD, num_stim=num_stim, stim_max=stim_max, flg_test=0, VE=0, Xternal=Xternal)

### Run simulation
RunPlasticNetwork(NeuPar, NetPar, StimPar, SimPar, LearnPar, SavePar, folder)
SaveNetworkPara(NeuPar, NetPar, StimPar, LearnPar, folder,'After')

### Analyse & plot network
Plot_Convergence(NeuPar, NetPar, StimPar, LearnPar, stim_max, folder)


#####################
# After plasticity ##
#####################

### Define input parameters
num_stim = np.int32(10)

StimPar_test = Stimulation(NeuPar, NetPar, SD, None, stim_max, VE=0, Xternal=Xternal)
StimPar_test_long = Stimulation(NeuPar, NetPar, SD, num_stim, stim_max, VE=0, flg_test=0, Xternal=Xternal)

### Run simulations
RunStaticNetwork(NeuPar, NetPar, StimPar_test_long, SimPar_test, folder, 'After_Balance')
RunStaticNetwork(NeuPar, NetPar, StimPar_test, SimPar_test, folder, 'After_Test')

### Analyse & plot network
Plot_Currents2PC(NeuPar, NetPar, StimPar_test_long, SimPar_test, folder, 'After_Balance')
HeatmapTest(NeuPar, NeuPar.NCells[0], folder, 'After_Test')
