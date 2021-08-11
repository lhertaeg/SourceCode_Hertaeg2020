#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5–Figure supplement 1. 
Coupled-trained networks can produce nPE neurons 
that decrease their activity in playback phase.
"""

# %% Import  & settings

import numpy as np
from Model_Network import Neurons, Network, Stimulation, Simulation, SaveNetworkPara 
from Model_Network import SaveData, Learning, RunStaticNetwork, RunPlasticNetwork
from Plot_NetResults import HeatmapTest, Plot_PopulationRate

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% Figure 5 - Supplement Figure 1

"""
Please note:    
    To reproduce all panels of the figure, 
    vary parameter 'FlagDendRec' (see below).
"""

folder = 'Fig_5_Supp_1'
FlagDendRec = 1 # 0 = no rectification, 1 = rectification
FlagDendRec_ABC = ['Dend_wo_rect_','Dend_w_rect_']

### Define neuron and network parameters
NeuPar = Neurons(FlagDendRec = np.int32(FlagDendRec))

wPP, wPE, wDS = dtype(-0.5), dtype(2.5), dtype(-5.0)
if FlagDendRec==1:
    wPV = dtype(-0.3)
elif FlagDendRec==0:
    wPV = dtype(-0.18)

NetPar = Network(NeuPar, wPP=wPP, wPE=wPE, wDS=wDS, wPV=wPV)

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
if FlagDendRec==0:
    Xternal[-1] = 40
num_stim = np.int32(3600)

StimPar = Stimulation(NeuPar, NetPar, SD, num_stim=num_stim, stim_max=stim_max, 
                      flg_test=0, Xternal=Xternal, CT = True)

### Run simulation
RunPlasticNetwork(NeuPar, NetPar, StimPar, SimPar, LearnPar, SavePar, folder, FlagDendRec_ABC[FlagDendRec])
SaveNetworkPara(NeuPar, NetPar, StimPar, LearnPar, folder, FlagDendRec_ABC[FlagDendRec] + 'After')


#####################
# After plasticity ##
#####################

### Define input parameters
num_stim = np.int32(10)

StimPar_test = Stimulation(NeuPar, NetPar, SD, None, stim_max, Xternal=Xternal)

### Run simulations
RunStaticNetwork(NeuPar, NetPar, StimPar_test, SimPar_test, folder, FlagDendRec_ABC[FlagDendRec] + 'After')

### Analyse & plot network
HeatmapTest(NeuPar, NeuPar.NCells[0], folder, FlagDendRec_ABC[FlagDendRec] + 'After')
Plot_PopulationRate(NeuPar, folder, FlagDendRec_ABC[FlagDendRec] + 'After', ini=1)

