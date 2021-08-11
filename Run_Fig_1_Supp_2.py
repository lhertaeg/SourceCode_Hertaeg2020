#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 1–Figure supplement 2. 
VIP->PV synapses are not required for the formation of nPE neurons.
"""

# %% Import  & settings

import numpy as np
from Model_Network import Neurons, Network, Stimulation, Simulation, SaveNetworkPara 
from Model_Network import SaveData, Learning, RunStaticNetwork, RunPlasticNetwork
from Plot_NetResults import HeatmapTest

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% Figure 1 - Supplement Figure 2

"""
Please note:    
    To reproduce panel c and f of the figure, 
    vary parameter 'PV_input_type' (see below).
"""

folder = 'Fig_1_Supp_2'
PV_input_type = 1 # 0: motor, 1: visual
PV_input_ABC = ['M','V']

### Define neuron and network parameters
NeuPar = Neurons()

wPP, wPE, wDS = dtype(-0.5), dtype(2.5), dtype(-5.0)
NetPar = Network(NeuPar, wPP=wPP, wPE=wPE, wDS=wDS, pPV=dtype(0))

### Define simulation parameters & recording options
SimPar_test = Simulation()
SimPar = Simulation(dt=0.2)
SavePar = SaveData()

### Define plasticity parameter
LearnPar = Learning(NeuPar, pPV=dtype(0))


# #####################
# ##### Learning ######
# #####################

### Define input parameters
stim_max, SD = dtype(100), dtype(5)
Xternal = dtype([400,2,2,2,0])
num_stim = np.int32(3600)

StimPar = Stimulation(NeuPar, NetPar, SD, num_stim=num_stim, stim_max=stim_max, flg_test=0,
                      Xternal=Xternal, VP = int(PV_input_type), MP = int(1-PV_input_type))

### Run simulation
RunPlasticNetwork(NeuPar, NetPar, StimPar, SimPar, LearnPar, SavePar, folder, PV_input_ABC[PV_input_type])
SaveNetworkPara(NeuPar, NetPar, StimPar, LearnPar, folder, PV_input_ABC[PV_input_type] + '_After')


#####################
# After plasticity ##
#####################

### Define input parameters
num_stim = np.int32(10)

StimPar_test = Stimulation(NeuPar, NetPar, SD, None, stim_max, Xternal=Xternal,
                           VP = int(PV_input_type), MP = int(1-PV_input_type))

### Run simulations
RunStaticNetwork(NeuPar, NetPar, StimPar_test, SimPar_test, folder, PV_input_ABC[PV_input_type] + '_After_Test')

### Analyse & plot network
HeatmapTest(NeuPar, NeuPar.NCells[0], folder, PV_input_ABC[PV_input_type] + '_After_Test')


