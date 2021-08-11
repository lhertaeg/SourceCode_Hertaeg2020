#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Routines to analyse network
"""

# %% import packages

import numpy as np
dtype = np.float32()

# %% functions    

def MeanCurrent2PC_Phases(NeuPar, NetPar, StimPar, SimPar, folder, fln):
    
    CaSpike = NeuPar.CaSpike
    CaThreshold = NeuPar.CaThreshold
    lambda_D = NeuPar.lambda_D
    lambda_E = NeuPar.lambda_E    
    NCells = NeuPar.NCells
    NC = np.cumsum(NCells)
    N = np.int32(sum(NCells))
    
    wDE = NetPar.wDE
    wEP = NetPar.wEP
    wDS = NetPar.wDS 
    
    stim_visual = StimPar.stim_visual
    stim_motor = StimPar.stim_motor
    inp_ext_soma = np.mean(StimPar.inp_ext_soma[:NC[0]])
    inp_ext_dend = np.mean(StimPar.inp_ext_dend)
    scale = np.mean(StimPar.neurons_visual[:NC[0]])
    
    dt = SimPar.dt
    stim_length = SimPar.stim_length
    num_time_steps = np.int32(stim_length/dt)
    
    # load data
    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticNetwork_' + fln + '.dat',delimiter=' ')
    t, R = arr[:,0], arr[:,1:N+1]
    
    # population responses
    rE = np.mean(R[:,:NC[0]],1)
    rP = np.mean(R[:,NC[0]:NC[1]],1)
    rS = np.mean(R[:,NC[1]:NC[2]],1)

    # visual and moto-related inputs
    V = np.repeat(stim_visual,num_time_steps)
    M = np.repeat(stim_motor,num_time_steps)
    
    # Compute mean weights
    wDE_mean = np.mean(np.sum(wDE,1))
    wDS_mean = np.mean(np.sum(wDS,1))
    wEP_mean = np.mean(np.sum(wEP,1))
 
    # Compute "currents"
    IS  = V*scale + inp_ext_soma + wEP_mean*rP 
    ID = M + inp_ext_dend + wDE_mean*rE + wDS_mean*rS 
    I0 = lambda_E * IS + (1-lambda_D) * ID
    IC = CaSpike * 0.5 * (np.sign(I0 - CaThreshold) + 1)
    ID = ID + IC  
    
    if NeuPar.FlagDendRec==1:
        ID[ID<0.0] = 0.0
      
    sNet = (1-lambda_E) * IS + lambda_D * ID
    sNet_Phases = np.zeros(4, dtype=dtype)
    sNet_Phases[0] = np.mean(sNet[(t>1500.0) & (t<2000.0)])
    sNet_Phases[1] = np.mean(sNet[(t>3500.0) & (t<4000.0)])
    sNet_Phases[2] = np.mean(sNet[(t>5500.0) & (t<6000.0)])
    sNet_Phases[3] = np.mean(sNet[((t>500.0) & (t<1000.0)) | ((t>2500.0) & (t<3000.0)) | ((t>4500.0) & (t<5000.0))])
    
    return sNet_Phases


def pathway_strengths(NeuPar, NetPar, VE, VP, MP, pPE_flag = False):
    
    gain = NeuPar.gain
    
    wEP_mean = np.mean(np.sum(NetPar.wEP,1))
    wSV_mean = np.mean(np.sum(NetPar.wSV,1))
    wVS_mean = np.mean(np.sum(NetPar.wVS,1))
    wPP_mean = np.mean(np.sum(NetPar.wPP,1))
    
    if not pPE_flag:
        wPS_mean = -(VP + abs(wVS_mean) * MP - (1 - wPP_mean)/(-gain * wEP_mean) * VE)
        wPV_mean = -(abs(wSV_mean) * abs(wPS_mean) + (1 - wSV_mean * wVS_mean) * MP)
    else:
        wPV_mean = -(VP + abs(wSV_mean) * MP - (1 - wPP_mean)/(-gain * wEP_mean) * VE)
        wPS_mean = -(abs(wVS_mean) * abs(wPV_mean) + (1 - wSV_mean * wVS_mean) * MP) 

    wEP_mean *= 0.07    
    INP = abs((1 - wSV_mean * wVS_mean) * (1-wPP_mean))
    PE = abs(wEP_mean * (1 - wSV_mean * wVS_mean))
    SPE = abs(wEP_mean * wPS_mean)
    VPE = abs(wEP_mean * wPV_mean)
    VSPE = abs(wEP_mean * wPS_mean * wSV_mean)
    SVPE = abs(wEP_mean * wPV_mean * wVS_mean)
        
    Path_SPE = np.array([SPE*(not pPE_flag), SPE*(pPE_flag)])
    Path_VPE = np.array([VPE*(pPE_flag), VPE*(not pPE_flag)])
    Path_SVPE = np.array([SVPE*(not pPE_flag), SVPE*(pPE_flag)])
    Path_VSPE = np.array([VSPE*(pPE_flag), VSPE*(not pPE_flag)])
    Path_INP = np.array([INP * VE, 0]) 
    Path_PE = np.array([PE * VP, PE * MP]) 
        
    return (Path_SPE, Path_VPE, Path_SVPE, 
            Path_VSPE, Path_INP, Path_PE)
    

def weights_steady_state_theo(NeuPar, NetPar, StimPar, LearnPar, stim_max):
    
    CaThreshold = NeuPar.CaThreshold
    NCells = NeuPar.NCells
    gain = NeuPar.gain
    lambda_D = NeuPar.lambda_D
    lambda_E = NeuPar.lambda_E
    NCells = NeuPar.NCells
    NC = np.cumsum(NCells)
    CaSpike = NeuPar.CaSpike * gain
    threshold = NeuPar.threshold * gain
    
    stim_max_Hz = gain * stim_max
    inp_ext_E = np.mean(StimPar.inp_ext_soma[:NC[0]]) * gain
    inp_ext_D = np.mean(StimPar.inp_ext_dend) * gain
    inp_ext_P = np.mean(StimPar.inp_ext_soma[NC[0]:NC[1]])
    inp_ext_S = np.mean(StimPar.inp_ext_soma[NC[1]:NC[2]])
    inp_ext_V = np.mean(StimPar.inp_ext_soma[NC[2]:])
    
    wSV_mean = np.abs(np.mean(np.sum(NetPar.wSV,1)))
    wVS_mean = np.abs(np.mean(np.sum(NetPar.wVS,1)))
    wSE_mean = np.abs(np.mean(np.sum(NetPar.wSE,1)))
    wVE_mean = np.abs(np.mean(np.sum(NetPar.wVE,1)))
    wDE_mean = np.abs(np.mean(np.sum(NetPar.wDE,1))) * gain
    wPE_mean = np.abs(np.mean(np.sum(NetPar.wPE,1)))
    wPP_mean = np.abs(np.mean(np.sum(NetPar.wPP,1)))

    rho_E = np.mean(LearnPar.rho_E)
    rho_P = np.mean(LearnPar.rho_P)
    rS = (inp_ext_S + stim_max_Hz - wSV_mean * (inp_ext_V + stim_max_Hz) +
         (wSE_mean - wSV_mean * wVE_mean) * rho_E) / (1 - wSV_mean * wVS_mean)
    rV = (inp_ext_V + stim_max_Hz - wVS_mean * (inp_ext_S + stim_max_Hz) + 
         (wVE_mean - wVS_mean * wSE_mean) * rho_E) / (1 - wSV_mean * wVS_mean) 
   
    wDS_mean = ((inp_ext_D + stim_max_Hz) + wDE_mean * rho_E) / rS / gain
    wDS_Ca_mean = ((inp_ext_D + stim_max_Hz) + wDE_mean * rho_E + CaSpike) / rS / gain
    
    if LearnPar.eta[1,0]==0:
        alpha = inp_ext_E + stim_max_Hz - (rho_E + threshold) / (1 - lambda_E)
        beta = (1 - (rS + wSV_mean*rV) / alpha)
        wPS_mean = (1 - (inp_ext_P + stim_max_Hz + wPE_mean * rho_E) / alpha) / beta
        wPV_mean = wSV_mean * wPS_mean
        wEP_mean = alpha * (1 + wPP_mean) / (inp_ext_P + stim_max_Hz + wPE_mean * rho_E 
                                             - wPV_mean * rV - wPS_mean * rS)/gain
        
        rP = (inp_ext_P + stim_max_Hz + wPE_mean * rho_E 
          - wPS_mean * rS - wPV_mean * rV) / (1 + wPP_mean)
        
        current_soma = inp_ext_E - wEP_mean * rP
    else:
        alpha = inp_ext_E - (rho_E + threshold) / (1 - lambda_E)
        wPS_mean = 1
        wPV_mean = wSV_mean * wPS_mean
        wEP_mean = alpha / rho_P / gain
        wPE_mean = ((1 + wPP_mean) * rho_P - inp_ext_P - stim_max_Hz + wPV_mean * rV + wPS_mean * rS) / rho_E
        
        current_soma = inp_ext_E - wEP_mean * rho_P 
    
    current_dend = inp_ext_D + stim_max_Hz - wDS_mean * rS + wDE_mean * rho_E
    current_test = lambda_E * current_soma + (1-lambda_D) * current_dend
    
    if current_test >= CaThreshold:
        wDS_mean = wDS_Ca_mean
    
    C = NetPar.NCon
    w_theo = [-wEP_mean, -wDS_mean, wPE_mean, -wPS_mean, -wPV_mean] 
    NCon = [C[0,1], C[0,2], C[1,0], C[1,2], C[1,3]]
    
    return w_theo, NCon
    
    

def Currents2PC_vs_time(NeuPar, NetPar, StimPar, SimPar, folder: str, fln: str = ''):
    
    CaSpike = NeuPar.CaSpike
    CaThreshold = NeuPar.CaThreshold
    lambda_D = NeuPar.lambda_D
    lambda_E = NeuPar.lambda_E    
    NCells = NeuPar.NCells
    NC = np.cumsum(NCells)
    N = np.int32(sum(NCells))
    
    wDE = NetPar.wDE
    wEP = NetPar.wEP
    wDS = NetPar.wDS 
    
    stim_visual = StimPar.stim_visual
    stim_motor = StimPar.stim_motor
    inp_ext_soma = np.mean(StimPar.inp_ext_soma[:NC[0]])
    inp_ext_dend = np.mean(StimPar.inp_ext_dend)
    scale = np.mean(StimPar.neurons_visual[:NC[0]])
    
    dt = SimPar.dt
    stim_length = SimPar.stim_length
    num_time_steps = np.int32(stim_length/dt)
    
    # load data
    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticNetwork_' + fln + '.dat',delimiter=' ')
    t, R = arr[:,0], arr[:,1:N+1]
    
    # population responses
    rE = np.mean(R[:,:NC[0]],1)
    rP = np.mean(R[:,NC[0]:NC[1]],1)
    rS = np.mean(R[:,NC[1]:NC[2]],1)
    
    rE0 = np.mean(R[(t>500.0) & (t<1000.0),:NC[0]])
    rP0 = np.mean(R[(t>500.0) & (t<1000.0),NC[0]:NC[1]])
    rS0 = np.mean(R[(t>500.0) & (t<1000.0),NC[1]:NC[2]])

    # visual and moto-related inputs
    V = np.repeat(stim_visual,num_time_steps)
    M = np.repeat(stim_motor,num_time_steps)
    
    # Compute mean weights
    wDE_mean = np.mean(np.sum(wDE,1))
    wDS_mean = np.mean(np.sum(wDS,1))
    wEP_mean = np.mean(np.sum(wEP,1))
 
    # Compute "currents"
    IS  = V*scale + inp_ext_soma + wEP_mean*rP 
    ID = M + inp_ext_dend + wDE_mean*rE + wDS_mean*rS 
    I0 = lambda_E * IS + (1-lambda_D) * ID
    IC = CaSpike * 0.5 * (np.sign(I0 - CaThreshold) + 1)
    ID = ID + IC  
    
    IS_BL = inp_ext_soma + wEP_mean*rP0
    ID_BL = inp_ext_dend + wDE_mean*rE0 + wDS_mean*rS0
    I0_BL = lambda_E * IS_BL + (1-lambda_D) * ID_BL
    IC_BL = CaSpike * 0.5 * (np.sign(I0_BL - CaThreshold) + 1)
    ID_BL = ID_BL + IC_BL
    
    if NeuPar.FlagDendRec==1:
        ID[ID<0.0] = 0.0
        ID_BL = ID_BL*(ID_BL>0.0)
    
    sExc = (1-lambda_E)*V*scale + lambda_D*(ID*(ID>=0.0) - ID_BL*(ID_BL>=0.0))
    sInh = wEP_mean*(1-lambda_E)*(rP-rP0) + lambda_D*(ID*(ID<0.0) - ID_BL*(ID_BL<0.0))
    
    return t, sExc, sInh, rE


def activity_realtive_to_base(NeuPar, folder: str, fln: str = ''):
    
    NCells = NeuPar.NCells
    N = np.int32(sum(NCells))
    NC = np.cumsum(NCells)
    
    # load data
    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticNetwork_' + fln + '.dat',delimiter=' ')
    t, R = arr[:,0], arr[:,1:N+1]
    
    # dR/R for all neurons as function of time
    Rate_BL = np.mean(R[(t>500.0) & (t<1000.0),:],0)
    dR_over_R_neurons = 100.0*(R - Rate_BL)/Rate_BL
    dR_over_R_neurons_phases = np.zeros((N,3))
    dR_over_R_neurons_phases[:,0] = np.mean(dR_over_R_neurons[(t>1000.0) & (t<2000.0),:],0)
    dR_over_R_neurons_phases[:,1] = np.mean(dR_over_R_neurons[(t>3000.0) & (t<4000.0),:],0)
    dR_over_R_neurons_phases[:,2] = np.mean(dR_over_R_neurons[(t>5000.0) & (t<6000.0),:],0)
    
    # dR/R for each population
    dR_over_R_pop = np.zeros((len(t),4))
    dR_over_R_pop[:,0] = np.mean(dR_over_R_neurons[:,:NC[0]],1)
    dR_over_R_pop[:,1] = np.mean(dR_over_R_neurons[:,NC[0]:NC[1]],1)
    dR_over_R_pop[:,2] = np.mean(dR_over_R_neurons[:,NC[1]:NC[2]],1)
    dR_over_R_pop[:,3] = np.mean(dR_over_R_neurons[:,NC[2]:],1)
    
    dR_over_R_pop_phases  = np.zeros((4,3))
    dR_over_R_pop_phases[:,0] = np.mean(dR_over_R_pop[(t>1000.0) & (t<2000.0),:],0)
    dR_over_R_pop_phases[:,1] = np.mean(dR_over_R_pop[(t>3000.0) & (t<4000.0),:],0)
    dR_over_R_pop_phases[:,2] = np.mean(dR_over_R_pop[(t>5000.0) & (t<6000.0),:],0)
    
    # dR/R for nPE neurons only
    MM_Sign, idx_MM = indices_MM(dR_over_R_neurons_phases)
    dR_over_R_MM = dR_over_R_neurons[:,idx_MM]
    dR_over_R_MM_phases = np.zeros((len(idx_MM),3))
    dR_over_R_MM_phases[:,0] = np.mean(dR_over_R_MM[(t>1000.0) & (t<2000.0),:],0)
    dR_over_R_MM_phases[:,1] = np.mean(dR_over_R_MM[(t>3000.0) & (t<4000.0),:],0)
    dR_over_R_MM_phases[:,2] = np.mean(dR_over_R_MM[(t>5000.0) & (t<6000.0),:],0)
    
    return (t, dR_over_R_neurons, dR_over_R_neurons_phases, 
            dR_over_R_pop, dR_over_R_pop_phases, 
            dR_over_R_MM, dR_over_R_MM_phases, idx_MM, MM_Sign)   


def indices_MM(dR_over_R_neurons_phases):
    
    tol, resp_tol = 10, 20

    MM_Sign = 1*((abs(dR_over_R_neurons_phases[:,0]) < tol) & 
                 (abs(dR_over_R_neurons_phases[:,2]) < tol) & 
                 (dR_over_R_neurons_phases[:,1]>resp_tol))
    
    return MM_Sign, np.where(MM_Sign)[0]