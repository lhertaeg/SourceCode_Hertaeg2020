#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 3. 
Simulated optogenetic manipulations of PV, SOM and VIP neurons 
disambiguate prediction-error circuits.
"""

# %% Import  & settings

import numpy as np
from Model_Network import Neurons, Network, Stimulation, Simulation, RunStaticNetwork
from Plot_NetResults import Heatmap_IN_manipulation

import warnings
warnings.filterwarnings("ignore")

# %% Figure 3

"""
Please note:    
    To reproduce all panels of the figure, 
    vary parameters 'Panel' and 'ManipulationType' (see below).
"""

folder = 'Fig_3'

### Choose panel
Panel = 4 # [a, b, c, d] = [1, 2, 3, 4]
ManipulationType = 1 # [act, inact] = [1, 2]

PanelABC = ['a', 'b', 'c', 'd']
ManipulationABC = ['act', 'inact']
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

### Define simulation parameters
SimPar = Simulation()
StimExtra = 5 if ManipulationType==1 else -8

### Define input parameters
stim_max, SD = 50.0, 10.0
r0 = np.array([1,2,2,4])

### Run simulations
for i in range(3):
    
    fln = PanelABC[Panel-1] + '_' + ManipulationABC[ManipulationType-1] + '_' + IN_type[i]
    StimPar = Stimulation(NeuPar, NetPar, SD, None, stim_max, r0 = r0, VE=VE, VP=VP, MP=MP)
    
    StimPar.inp_ext_soma[Nb[0]:Nb[1]] += StimExtra * (i==0)
    StimPar.inp_ext_soma[Nb[1]:Nb[2]] += StimExtra * (i==1)
    StimPar.inp_ext_soma[Nb[2]:] += StimExtra * (i==2)
    
    RunStaticNetwork(NeuPar, NetPar, StimPar, SimPar, folder, fln)
    
### Analyse & plot network
Heatmap_IN_manipulation(NeuPar, Panel, ManipulationType, folder)