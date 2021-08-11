#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Appendix 2 Figure 2. 
Balancing excitation and inhibition gives rise to positive prediction-error neurons.
"""

# %% Import  & settings

import numpy as np
from Model_Network import Neurons, Network, Stimulation, Simulation, SaveNetworkPara 
from Model_Network import SaveData, Learning, RunStaticNetwork, RunPlasticNetwork
from Plot_NetResults import HeatmapTest, Plot_Currents2PC

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% Appendix 2 Figure 2

folder = 'Appx_2_Fig_2'

### Define neuron and network parameters
NeuPar = Neurons()

wSV = dtype(-0.8)
wPP, wPE, wDS = dtype(-0.5), dtype(2.5), dtype(-5.0)
NetPar = Network(NeuPar, wPP=wPP, wPE=wPE, wDS=wDS, wSV=wSV)

### Define simulation parameters & recording options
SimPar_test = Simulation()
SimPar = Simulation(dt=0.2)
SavePar = SaveData()

### Define plasticity parameter
LearnPar = Learning(NeuPar)


#####################
# Before plasticity #
#####################

### Define input parameters
stim_max, SD = dtype(100), dtype(5)
Xternal = dtype([400,2,2,2,0])
num_stim = np.int32(10)
VS, VV, VP, MP = np.int32(0), np.int32(1), np.int32(0), np.int32(1)

StimPar_test = Stimulation(NeuPar, NetPar, SD, None, stim_max, VS=VS, VV=VV, VP=VP, MP=MP,
                           Xternal=Xternal, pPE_flag = True, MM_factor = dtype(0.5))
StimPar_test_long = Stimulation(NeuPar, NetPar, SD, num_stim, stim_max, VS=VS, VV=VV, VP=VP, MP=MP,
                                flg_test=0, Xternal=Xternal, pPE_flag = True)

### Run simulations
RunStaticNetwork(NeuPar, NetPar, StimPar_test_long, SimPar_test, folder, 'Before_Balance')
RunStaticNetwork(NeuPar, NetPar, StimPar_test, SimPar_test, folder, 'Before_Test')
SaveNetworkPara(NeuPar, NetPar, StimPar_test_long, None, folder,'Before')

### Analyse & plot network
Plot_Currents2PC(NeuPar, NetPar, StimPar_test_long, SimPar_test, folder, 'Before_Balance')
HeatmapTest(NeuPar, NeuPar.NCells[0], folder, 'Before_Test', MM_factor=dtype(0.5))


# #####################
# ##### Learning ######
# #####################

### Define input parameters
num_stim = np.int32(3600)

StimPar = Stimulation(NeuPar, NetPar, SD, num_stim=num_stim, stim_max=stim_max, flg_test=0, 
                      VS=VS, VV=VV, VP=VP, MP=MP, Xternal=Xternal, pPE_flag = True)

### Run simulation
RunPlasticNetwork(NeuPar, NetPar, StimPar, SimPar, LearnPar, SavePar, folder)
SaveNetworkPara(NeuPar, NetPar, StimPar, LearnPar, folder, 'After')


#####################
# After plasticity ##
#####################

### Define input parameters
num_stim = np.int32(10)

StimPar_test = Stimulation(NeuPar, NetPar, SD, None, stim_max, VS=VS, VV=VV, VP=VP, MP=MP,
                           Xternal=Xternal, pPE_flag = True, MM_factor = dtype(0.5))
StimPar_test_long = Stimulation(NeuPar, NetPar, SD, num_stim, stim_max, VS=VS, VV=VV, VP=VP, MP=MP,
                                flg_test=0, Xternal=Xternal, pPE_flag = True)

### Run simulations
RunStaticNetwork(NeuPar, NetPar, StimPar_test_long, SimPar_test, folder, 'After_Balance')
RunStaticNetwork(NeuPar, NetPar, StimPar_test, SimPar_test, folder, 'After_Test')
SaveNetworkPara(NeuPar, NetPar, StimPar_test_long, None, folder,'After')

### Analyse & plot network
Plot_Currents2PC(NeuPar, NetPar, StimPar_test_long, SimPar_test, folder, 'After_Balance')
HeatmapTest(NeuPar, NeuPar.NCells[0], folder, 'After_Test', MM_factor=dtype(0.5))

