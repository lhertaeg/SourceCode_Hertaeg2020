#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5. 
Experience-dependence of nPE and PV neurons.
"""

# %% Import  & settings

import numpy as np
from Model_Network import Neurons, Network, Stimulation, Simulation, SaveNetworkPara 
from Model_Network import SaveData, Learning, RunStaticNetwork, RunPlasticNetwork
from Plot_NetResults import BarPlot_Comparison_Training, Plot_MM_Comparison_Training 
from Plot_NetResults import Plot_Comparison_QTvsCT

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32

# %% Figure 5

folder = 'Fig_5'

# #####################################################################
# ##### Compare Quasi-natural training with Random gain training ######
# #####################################################################

VS = 0.9
VV = 0.5

### Define neuron and network parameters
NeuPar = Neurons()

wPP, wPE, wDS = dtype(-0.5), dtype(2.5), dtype(-5.0)

np.random.seed(186)
NetPar_QT = Network(NeuPar, wPP=wPP, wPE=wPE, wDS=wDS)

np.random.seed(186)
NetPar_RT = Network(NeuPar, wPP=wPP, wPE=wPE, wDS=wDS)

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
num_stim = np.int32(5400) 

StimPar_QT = Stimulation(NeuPar, NetPar_QT, SD, num_stim=num_stim, stim_max=stim_max, 
                      VS=VS, VV=VV, flg_test=0, Xternal=Xternal)

StimPar_RT = Stimulation(NeuPar, NetPar_RT, SD, num_stim=num_stim, stim_max=stim_max, 
                      VS=VS, VV=VV, flg_test=0, Xternal=Xternal, QT=False)

StimPar_RT.neurons_visual = StimPar_QT.neurons_visual
StimPar_RT.neurons_motor = StimPar_QT.neurons_motor

### Run simulations
fln_QT = 'QT'
RunPlasticNetwork(NeuPar, NetPar_QT, StimPar_QT, SimPar, LearnPar, SavePar, folder, fln_QT)
SaveNetworkPara(NeuPar, NetPar_QT, StimPar_QT, LearnPar, folder, fln_QT)

fln_RT = 'RT'
RunPlasticNetwork(NeuPar, NetPar_RT, StimPar_RT, SimPar, LearnPar, SavePar, folder, fln_RT)
SaveNetworkPara(NeuPar, NetPar_RT, StimPar_RT, LearnPar, folder, fln_RT)


#####################
# After plasticity ##
#####################

### Define input parameters
StimPar_test_QT = Stimulation(NeuPar, NetPar_QT, SD, None, stim_max, Xternal=Xternal)
StimPar_test_QT.neurons_visual = StimPar_QT.neurons_visual
StimPar_test_QT.neurons_motor = StimPar_QT.neurons_motor

StimPar_test_RT = Stimulation(NeuPar, NetPar_RT, SD, None, stim_max, Xternal=Xternal)
StimPar_test_RT.neurons_visual = StimPar_RT.neurons_visual
StimPar_test_RT.neurons_motor = StimPar_RT.neurons_motor

### Run simulations
RunStaticNetwork(NeuPar, NetPar_QT, StimPar_test_QT, SimPar_test, folder, fln_QT)
RunStaticNetwork(NeuPar, NetPar_RT, StimPar_test_RT, SimPar_test, folder, fln_RT)

### Analyse & plot network
BarPlot_Comparison_Training(NeuPar, fln_QT, fln_RT, folder)
Plot_MM_Comparison_Training(NeuPar, fln_QT, fln_RT, folder)



# #################################################################
# ##### Compare Quasi-natural training with coupled training ######
# #################################################################

### Define neuron and network parameters
NeuPar = Neurons()

wPV = dtype(-0.3)
NetPar = Network(NeuPar, wPP=wPP, wPE=wPE, wDS=wDS, wPV=wPV)

### Define plasticity parameter
LearnPar = Learning(NeuPar, pPV=dtype(0))


# #####################
# ##### Learning ######
# #####################

### Define input parameters
num_stim = np.int32(3600)
fln_CT = 'CT'

StimPar = Stimulation(NeuPar, NetPar, SD, num_stim=num_stim, stim_max=stim_max, 
                      flg_test=0, Xternal=Xternal, CT = True)

### Run simulation
RunPlasticNetwork(NeuPar, NetPar, StimPar, SimPar, LearnPar, SavePar, folder, fln_CT)
SaveNetworkPara(NeuPar, NetPar, StimPar, LearnPar, folder, fln_CT)


#####################
# After plasticity ##
#####################

### Define input parameters
num_stim = np.int32(10)

StimPar_test = Stimulation(NeuPar, NetPar, SD, None, stim_max, Xternal=Xternal)

### Run simulations
RunStaticNetwork(NeuPar, NetPar, StimPar_test, SimPar_test, folder, fln_CT)

### Analyse network and compare to QT
Plot_Comparison_QTvsCT(NeuPar, folder, fln_QT, fln_CT, G = None)
