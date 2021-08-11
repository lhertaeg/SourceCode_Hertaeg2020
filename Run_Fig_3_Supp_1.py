#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 3–Figure supplement 1. 
Net currents in PCs after in/activation of PV, SOM or VIP neurons 
elucidate prediction-error circuits.
"""

# %% Import  & settings

import numpy as np
from Model_Network import Neurons, Network, Stimulation, Simulation, RunStaticNetwork
from Plot_NetResults import Plot_Current2PC_OptoStim

import warnings
warnings.filterwarnings("ignore")

dtype = np.float32()

# %% Figure 3 - Supp 1

"""
Please note:    
    To reproduce all panels of the figure, 
    vary parameter 'Panel' (see below).
"""

folder = 'Fig_3_Supp_1'

### Choose panel
Panel = 4 # [a, b, c, d] = [1, 2, 3, 4]

PanelABC = ['a', 'b', 'c', 'd']
IN_type = ['PV', 'SOM', 'VIP']

### Define neuron and network parameters
NCells = np.array([70,10,10,10])
Nb = np.cumsum(NCells)

if Panel==1: # a
    VE, VP, MP = 1, 1, 0
    wSV, wVS, wPP, wEP = -0.6, -0.5, -0.1, -40.0
elif Panel==2: # b
    VE, VP, MP = 1, 0, 1
    wSV, wVS, wPP, wEP = -0.6, -0.5, -0.1, -40.0
elif Panel==3: # c
    VE, VP, MP = 0, 1, 0
    wSV, wVS, wPP, wEP = -0.6, -0.5, -1.5, -40.0
elif Panel==4: # d
    VE, VP, MP = 0, 0, 1
    wSV, wVS, wPP, wEP = -0.6, -0.5, -1.5, -40.0

wPS = -(VP + abs(wVS)*MP - (1-wPP)/(-0.07*wEP) * VE) # gain = 0.07
wPV = -(abs(wSV)*abs(wPS) + (1-wSV*wVS)*MP)

NeuPar = Neurons()
NetPar = Network(NeuPar, wPP=wPP, wPS=wPS, wPV=wPV, wEP=wEP, flag_hetero=0)

### Define input parameters
stim_max, SD = 50.0, 10.0
r0 = np.array([1,2,2,4])
StimExtra_all =  np.arange(-8, 8, 2, dtype=dtype)

### Define simulation parameters
SimPar = Simulation()

### Run simulations
for i in range(3):
    
    ListAllFiles = []
    
    for j in range(len(StimExtra_all)):

        StimExtra = StimExtra_all[j]
        fln = PanelABC[Panel-1] + '_' + IN_type[i]  + '_StimExtra_' + str(StimExtra)
        ListAllFiles.append(fln)
        
        StimPar = Stimulation(NeuPar, NetPar, SD, None, stim_max, r0 = r0, VE=VE, VP=VP, MP=MP)
        StimPar.inp_ext_soma[Nb[0]:Nb[1]] += StimExtra * (i==0)
        StimPar.inp_ext_soma[Nb[1]:Nb[2]] += StimExtra * (i==1)
        StimPar.inp_ext_soma[Nb[2]:] += StimExtra * (i==2)
        
        RunStaticNetwork(NeuPar, NetPar, StimPar, SimPar, folder, fln)
    
    # Analyse and plot simulation/network results  
    fln = PanelABC[Panel-1] + '_' + IN_type[i]  
    Plot_Current2PC_OptoStim(NeuPar, NetPar, StimPar, SimPar, ListAllFiles, StimExtra_all, folder, fln, LegendFlag=False)
