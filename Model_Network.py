#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classes and functions to create and run network
"""

# %% import packages & settings

import numpy as np
import tables
import os
from numba import njit
from typing import NamedTuple, NewType
import pickle

dtype = np.float32

RateVec = NewType("RateVec", np.ndarray)
WeightVec = NewType("WeightVec", np.ndarray)

# %% classes

class Neurons(NamedTuple):
        
        NCells: list = np.array([70,10,10,10], dtype=np.int32)
        FlagDendRec: int = np.int32(1)
        tau_inv_E: dtype = dtype(1.0/60.0)
        tau_inv_I: dtype = dtype(1.0/2.0)
        gain: dtype = dtype(0.07)
        threshold: dtype = dtype(200)
        lambda_E: dtype = dtype(0.31)
        lambda_D: dtype = dtype(0.27)
        CaSpike: dtype = dtype(100.0)
        CaThreshold: dtype = dtype(400.0)               
          
    
class Network:
    def __init__(
            self, Neurons, pPV: dtype = dtype(0.5),
            wED: dtype = dtype(6), wEP: dtype = dtype(-25), wDS: dtype = dtype(-50), wPE: dtype = dtype(1.5), 
            wPP: dtype = dtype(-1.5), wPS: dtype = dtype(-0.3), wPV: dtype = dtype(-0.6),
            wSV: dtype = dtype(-0.6), wVS: dtype = dtype(-0.5), flag_hetero: int = np.int32(1)):
        
            NCells = Neurons.NCells
            conn_prob = np.array([[0.1,0.6,0.55,0.0],[0.45,0.5,0.6,pPV],
                                  [0.35,0.0,0.0,0.5],[0.1,0.0,0.45,0.0]], dtype=dtype)
            weights_mean = np.array([[wED,wEP,wDS,0],[wPE,wPP,wPS,wPV],
                                      [1.0,0.0,0.0,wSV],[1.0,0.0,wVS,0.0]], dtype=dtype)
            
            weight_name = np.array([['wDE','wEP','wDS','wDV'],['wPE','wPP','wPS','wPV'],
                                    ['wSE','wSP','wSS','wSV'],['wVE','wVP','wVS','wVV']])
            temp_name = np.array([['TDE','TEP','TDS','TDV'],['TPE','TPP','TPS','TPV'],
                                    ['TSE','TSP','TSS','TSV'],['TVE','TVP','TVS','TVV']])
            
            NCon = np.round(conn_prob * NCells).astype(np.int32)
            self.NCon = NCon
            
            for i in range(16):
                m,n = np.unravel_index(i,(4,4))
                Mtx = np.zeros((NCells[m],NCells[n]), dtype=dtype)
                if NCon[m,n]>0:
                    if m==n:
                        for l in range(NCells[m]):
                            r = np.array([0] * (NCells[n]-1-NCon[m,n]) + [1] * NCon[m,n], dtype=dtype)
                            np.random.shuffle(r)
                            r = np.insert(r,l,0)         
                            Mtx[l,:] = r            
                    else:
                        for l in range(NCells[m]):
                            r = np.array([0] * (NCells[n]-NCon[m,n]) + [1] * NCon[m,n], dtype=dtype)
                            np.random.shuffle(r)      
                            Mtx[l,:] = r 
                    if flag_hetero==1:
                        Mtx[:] *= dtype(np.random.uniform(0.5*weights_mean[m,n],1.5*weights_mean[m,n],size=(NCells[m],NCells[n])))/NCon[m,n]
                    elif flag_hetero==0:
                        Mtx[:] *= weights_mean[m,n]/NCon[m,n]
                exec('self.' + weight_name[m][n] + ' = Mtx')
                exec('self.' + temp_name[m][n] + ' = (Mtx!=dtype(0))')   
            
   
class Stimulation:
    def __init__(
        self, Neurons, Network, SD: dtype, num_stim: int, stim_max: dtype, flg_test: int = 1,
        VE: int = 1, VP: int = 1, VS: int = 1, VV: int = 0, MP: int = 0, r0: list = None, 
        Xternal: list = None, QT: bool = True, CT: bool = False, MM_factor = 0, VE_scale: dtype = 1,
        pPE_flag: bool = False):
        
        NCells = Neurons.NCells
        Nb = np.cumsum(NCells, dtype=np.int32) 
        
        if flg_test==1:
            stim_visual = np.array([0, stim_max, 0, stim_max * MM_factor, 0, stim_max, 0], dtype=dtype)
            stim_motor = np.array([0, stim_max, 0, stim_max, 0, stim_max * MM_factor, 0], dtype=dtype)
        else:
            stim_visual = np.tile(np.array([0,1],dtype=dtype),num_stim)
            stim_visual[:] = stim_visual * dtype(np.round(np.random.uniform(0.0,stim_max,size=2*num_stim)))
            if QT:
                stim_motor = stim_visual.copy()
                if (not CT and not pPE_flag):
                    stim_motor[:] = stim_motor * np.random.choice((0,1),size=2*num_stim,p=[0.5,0.5])
                elif (not CT and pPE_flag):
                    stim_visual[:] = stim_visual * np.random.choice((0,1),size=2*num_stim,p=[0.5,0.5])
            else:
                stim_motor = np.tile(np.array([0,1],dtype=dtype),num_stim)
                stim_motor[:] = stim_motor * dtype(np.round(np.random.uniform(0.0,stim_max,size=2*num_stim)))
                   
        NE, NP, NS, NV = NCells
        num_VE, num_VS, num_VV  = int(VE*NE), int(VS*NS), int(VV*NV) 
        num_VP, num_MP = int(VP*NP), int(MP*NP)
    
        vis_E = np.array([1] * num_VE + [0] * (NE - num_VE), dtype=dtype)
        vis_P = np.array([1] * num_VP + [0] * (NP - num_VP), dtype=dtype)
        vis_S = np.array([1] * num_VS + [0] * (NS - num_VS), dtype=dtype)
        vis_V = np.array([1] * num_VV + [0] * (NV - num_VV), dtype=dtype)
        mot_P = np.array([1] * num_MP + [0] * (NP - num_MP), dtype=dtype)
        
        if (VE_scale!=1.0):
            vis_E *= dtype(VE_scale)
    
        np.random.shuffle(vis_E), np.random.shuffle(vis_P), np.random.shuffle(vis_S)
        np.random.shuffle(vis_V), np.random.shuffle(mot_P)
    
        neurons_visual = np.zeros(sum(NCells), dtype=dtype)
        neurons_visual[:Nb[0]] = vis_E
        neurons_visual[Nb[0]:Nb[1]] = vis_P
        neurons_visual[Nb[1]:Nb[2]] = vis_S
        neurons_visual[Nb[2]:] = vis_V
        
        neurons_motor = np.zeros(sum(NCells), dtype=dtype)
        neurons_motor[:Nb[0]] = 1
        neurons_motor[Nb[1]:Nb[2]] = 1 - neurons_visual[Nb[1]:Nb[2]]
        neurons_motor[Nb[2]:] = 1 - neurons_visual[Nb[2]:]
        if MP==0:
            neurons_motor[Nb[0]:Nb[1]] = 1 - neurons_visual[Nb[0]:Nb[1]]
        else:
            neurons_motor[Nb[0]:Nb[1]] = mot_P
        
        inp_ext_soma, inp_ext_dend = BackgroundInput(Neurons, Network, r0, Xternal)
 
        self.SD: dtype = SD
        self.stim_visual: dtype = stim_visual
        self.stim_motor: dtype = stim_motor
        self.neurons_visual: list = neurons_visual
        self.neurons_motor: list = neurons_motor
        self.inp_ext_soma: dtype = inp_ext_soma
        self.inp_ext_dend: dtype = inp_ext_dend
        self.CT: bool = CT
        
   
class Simulation(NamedTuple): 
        
        dt: dtype = dtype(0.1)
        stim_length: dtype = dtype(1000.0)


class Learning: 
    def __init__(self, Neurons, flagRule: int = 1, decay: dtype = 0.1, pPV: dtype = dtype(0.5)):
        
        NCells = Neurons.NCells
        rho_E = np.zeros(NCells[0], dtype=dtype)
        rho_P = np.zeros(NCells[1], dtype=dtype)
        
        if flagRule==0:
            eta = np.array([[0,1e-6,1e-7,0],[1e-6,0,1e-7,1e-7],[0,0,0,0],[0,0,0,0]], dtype=dtype)
            rho_E[:] = dtype(2.0)
            rho_P[:] = dtype(2.5)
        else:
            eta = np.array([[0,1e-4,1e-6,0],[0,0,1e-6,1e-6],[0,0,0,0],[0,0,0,0]], dtype=dtype)
            rho_E[:] = dtype(1.25)
        
        if pPV==0:
            eta[1,3] = dtype(0)
            
        self.eta: dtype = eta
        self.rho_E: dtype = rho_E
        self.rho_P: dtype = rho_P
        self.decay: dtype = dtype(decay)
        self.flagRule: int = np.int32(flagRule)
 
    
class SaveData(NamedTuple):
        
        ijk: list = np.array([0,0,0,0],dtype=np.int32)
        nstep: int = np.int32(100)

   
# %% functions

@njit(cache=True)
def drdt(tau_inv_E, tau_inv_I, lambda_E, lambda_D, 
         gain, threshold, CaSpike, CaThreshold, FlagDendRec, 
         wEP, wDS, wDE, 
         wPE, wPP, wPS, wPV, 
         wSE, wSP, wSS, wSV, 
         wVE, wVP, wVS, wVV,
         rE, rP, rS, rV,
         StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend): 
    
    input_soma = StimSoma_E + wEP @ rP 
    input_dend = StimDend + wDS @ rS + wDE @ rE
    input_dend +=  CaSpike * dtype(0.5) * (np.sign(lambda_E*input_soma 
                                            + (dtype(1)-lambda_D)*input_dend - CaThreshold) + dtype(1))
    input_dend[...] = np.where(input_dend>=0, input_dend, input_dend*(1-FlagDendRec))    
    
    drE = tau_inv_E * (-rE + gain * ((1-lambda_E)*input_soma + lambda_D*input_dend - threshold))
    drP = tau_inv_I * (-rP + wPE @ rE + wPP @ rP + wPS @ rS + wPV @ rV + StimSoma_P)
    drS = tau_inv_I * (-rS + wSE @ rE + wSP @ rP + wSS @ rS + wSV @ rV + StimSoma_S)
    drV = tau_inv_I * (-rV + wVE @ rE + wVP @ rP + wVS @ rS + wVV @ rV + StimSoma_V)
    
    return drE, drP, drS, drV, input_dend


def RateDynamics(tau_inv_E, tau_inv_I, lambda_E, lambda_D, 
                 gain, threshold, CaSpike, CaThreshold, FlagDendRec, 
                 wEP, wDS, wDE, 
                 wPE, wPP, wPS, wPV, 
                 wSE, wSP, wSS, wSV, 
                 wVE, wVP, wVS, wVV,
                 rE, rP, rS, rV,
                 StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend,
                 dt):
    
    rE0 = rE.copy()
    rP0 = rP.copy()
    rS0 = rS.copy()
    rV0 = rV.copy()
    
    drE1, drP1, drS1, drV1, inp_dend1 = drdt(tau_inv_E, tau_inv_I, lambda_E, lambda_D, 
                                             gain, threshold, CaSpike, CaThreshold, FlagDendRec, 
                                             wEP, wDS, wDE, 
                                             wPE, wPP, wPS, wPV, 
                                             wSE, wSP, wSS, wSV, 
                                             wVE, wVP, wVS, wVV,
                                             rE0, rP0, rS0, rV0,
                                             StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend)
    rE0[:] += dt * drE1
    rP0[:] += dt * drP1
    rS0[:] += dt * drS1
    rV0[:] += dt * drV1
    
    drE2, drP2, drS2, drV2, inp_dend2 = drdt(tau_inv_E, tau_inv_I, lambda_E, lambda_D, 
                                             gain, threshold, CaSpike, CaThreshold, FlagDendRec, 
                                             wEP, wDS, wDE, 
                                             wPE, wPP, wPS, wPV, 
                                             wSE, wSP, wSS, wSV, 
                                             wVE, wVP, wVS, wVV,
                                             rE0, rP0, rS0, rV0,
                                             StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend)
    rE[:] += dt/2 * (drE1 + drE2)
    rP[:] += dt/2 * (drP1 + drP2) 
    rS[:] += dt/2 * (drS1 + drS2) 
    rV[:] += dt/2 * (drV1 + drV2) 
    
    rE[rE<0] = 0
    rP[rP<0] = 0
    rS[rS<0] = 0
    rV[rV<0] = 0

    return (inp_dend1+inp_dend2)/2 


@njit(cache=True)
def WeightDynamics(wEP, wDS, wPE, wPS, wPV,
                   TEP, TDS, TPE, TPS, TPV,
                   eta, rho_E, rho_P, decay, flagRule,
                   rE, rP, rS, rV, activity_dend, FlagDendRec, CT):
       
    mean_weights = np.ones(5, dtype=dtype)
    mean_weights.fill(np.nan)
    diff_rE = (rE - rho_E)
    diff_rD = (activity_dend - decay)
    if ((FlagDendRec==0) and not CT):
        diff_rD = dtype(0.5) * (np.tanh(activity_dend/dtype(50)) + 1) * activity_dend
    diff_rP = (rP - rho_P)

    if flagRule==1:
        UEP = np.outer(diff_rE, np.ones_like(rP, dtype=dtype))
        UEP[...] = np.where(TEP, UEP, np.nan)
        mean_EP = np.zeros(len(rP), dtype=dtype)
        for i in range(len(rP)):
            mean_EP[i] = np.nanmean(UEP[:,i])
    elif flagRule==2:
        UPE = np.outer(np.ones_like(rP, dtype=dtype), diff_rE)
        UPE[...] = np.where(TPE, UPE, np.nan)
        UPE *= wPE 
        mean_PE = np.zeros(len(rP), dtype=dtype)
        for i in range(len(rP)):
            mean_PE[i] = np.nanmean(UPE[i,:])
    
    wEP -= eta[0,1] * np.outer(diff_rE,rP)
    wEP *= TEP
    wEP[...] = np.where(wEP<=0,wEP,dtype(0))        
    mean_weights[0] = np.nanmean(np.where(TEP, wEP, np.nan))
    
    wDS -= eta[0,2] * np.outer(diff_rD,rS)
    wDS *= TDS
    wDS[...] = np.where(wDS<=0,wDS,dtype(0)) 
    mean_weights[1] = np.nanmean(np.where(TDS, wDS, np.nan))
    
    if flagRule==0:
        wPE -= eta[1,0] * np.outer(diff_rP,rE)
        wPE *= TPE
        wPE[...] = np.where(wPE>=0,wPE,dtype(0)) 
        mean_weights[2] = np.nanmean(np.where(TPE, wPE, np.nan))
    
    if flagRule==1:
        wPS += eta[1,2] * np.outer(mean_EP,rS)
    elif flagRule==0:
        wPS -= eta[1,2] * np.outer(diff_rP,rS)
    elif flagRule==2:
        wPS += eta[1,2] * np.outer(mean_PE,rS)
    wPS *= TPS
    wPS[...] = np.where(wPS<=0,wPS,dtype(0))  
    mean_weights[3] = np.nanmean(np.where(TPS, wPS, np.nan))
    
    if flagRule==1:
        wPV += eta[1,3] * np.outer(mean_EP,rV)
    elif flagRule==0:
        wPV -= eta[1,3] * np.outer(diff_rP,rV) 
    elif flagRule==2:
        wPV += eta[1,3] * np.outer(mean_PE,rV) 
    wPV *= TPV
    wPV[...] = np.where(wPV<=0,wPV,dtype(0)) 
    mean_weights[4] = np.nanmean(np.where(TPV, wPV, np.nan))
    
    return mean_weights


def RunStaticNetwork(NeuPar: Neurons, NetPar: Network, StimPar: Stimulation, SimPar: Simulation, 
                     folder: str, fln: str = '') -> None:
    
    CaSpike = NeuPar.CaSpike
    CaThreshold = NeuPar.CaThreshold
    FlagDendRec = NeuPar.FlagDendRec
    NCells = NeuPar.NCells
    gain = NeuPar.gain
    lambda_D = NeuPar.lambda_D
    lambda_E = NeuPar.lambda_E
    tau_inv_E = NeuPar.tau_inv_E
    tau_inv_I = NeuPar.tau_inv_I
    threshold = NeuPar.threshold
    N = np.int32(sum(NCells))
    ind_break = np.cumsum(NCells[1:],dtype=np.int32)[:-1]
    nE = NCells[0]

    wDE = NetPar.wDE
    wEP = NetPar.wEP
    wDS = NetPar.wDS 
    wPE = NetPar.wPE
    wPP = NetPar.wPP
    wPS = NetPar.wPS
    wPV = NetPar.wPV
    wSE = NetPar.wSE
    wSP = NetPar.wSP
    wSS = NetPar.wSS
    wSV = NetPar.wSV
    wVE = NetPar.wVE
    wVP = NetPar.wVP
    wVS = NetPar.wVS
    wVV = NetPar.wVV
    
    SD = StimPar.SD
    NStim = np.int32(len(StimPar.stim_visual))
    stim_visual = iter(StimPar.stim_visual)
    stim_motor = iter(StimPar.stim_motor)
    neurons_visual = StimPar.neurons_visual
    neurons_motor = StimPar.neurons_motor
    inp_ext_soma = StimPar.inp_ext_soma
    inp_ext_dend = StimPar.inp_ext_dend
    
    dt = SimPar.dt
    stim_length = SimPar.stim_length
    num_time_steps = np.int32(stim_length/dt)    
    
    StimSoma_E = np.zeros(nE, dtype=dtype)
    StimSoma_P = np.zeros(NCells[1], dtype=dtype)
    StimSoma_S = np.zeros(NCells[2], dtype=dtype)
    StimSoma_V = np.zeros(NCells[3], dtype=dtype)
    StimDend = np.zeros(NCells[0], dtype=dtype)
    stim_IN = np.zeros(N-nE, dtype=dtype)
    rE = np.zeros(nE, dtype=dtype)
    rP = np.zeros(NCells[1], dtype=dtype)
    rS = np.zeros(NCells[2], dtype=dtype)
    rV = np.zeros(NCells[3], dtype=dtype)
    
    noise_soma = np.zeros((N,num_time_steps),dtype=dtype)
    noise_dend = np.zeros((NCells[0],num_time_steps),dtype=dtype)
    
    path = 'Results/Data/' + folder
    if not os.path.exists(path):
        os.mkdir(path)

    fp = open(path +'/Data_StaticNetwork_' + fln + '.dat','w')   
    
    # main loop
    for s in range(NStim):
        
        print('Stimuli', str(s+1), '/', str(NStim))
        
        V = next(stim_visual)
        M = next(stim_motor)
        noise_soma[:] = np.random.normal(0,SD,size=(N, num_time_steps))
        noise_dend[:] = np.random.normal(0,SD,size=(NCells[0], num_time_steps))
        
        for tstep in range(num_time_steps):
            
            stim_IN[:] = gain*(V*neurons_visual[nE:] + M*neurons_motor[nE:] + noise_soma[nE:,tstep]) + inp_ext_soma[nE:]
            StimSoma_E[:] = V*neurons_visual[:nE] + noise_soma[:nE,tstep] + inp_ext_soma[:nE]
            StimSoma_P[:], StimSoma_S[:], StimSoma_V[:] = np.split(stim_IN, ind_break)
            StimDend[:] = M*neurons_motor[:nE] + noise_dend[:,tstep] + inp_ext_dend
            
            _ = RateDynamics(tau_inv_E, tau_inv_I, lambda_E, lambda_D, 
                             gain, threshold, CaSpike, CaThreshold, FlagDendRec, 
                             wEP, wDS, wDE, 
                             wPE, wPP, wPS, wPV, 
                             wSE, wSP, wSS, wSV, 
                             wVE, wVP, wVS, wVV,
                             rE, rP, rS, rV,
                             StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend,
                             dt)
            
            fp.write("%f" % (s*stim_length + (tstep+1)*dt))
            for i in range(NCells[0]):
                fp.write(" %f" % rE[i])
            for i in range(NCells[1]):
                fp.write(" %f" % rP[i])
            for i in range(NCells[2]):
                fp.write(" %f" % rS[i])
            for i in range(NCells[3]):
                fp.write(" %f" % rV[i])
            fp.write("\n") 
        
    fp.closed
    return


def RunPlasticNetwork(NeuPar: Neurons, NetPar: Network, StimPar: Stimulation, 
                      SimPar: Simulation, LearnPar: Learning, SavePar: SaveData,
                      folder: str, fln: str = '') -> None:
    
    CaSpike = NeuPar.CaSpike
    CaThreshold = NeuPar.CaThreshold
    FlagDendRec = NeuPar.FlagDendRec
    NCells = NeuPar.NCells
    gain = NeuPar.gain
    lambda_D = NeuPar.lambda_D
    lambda_E = NeuPar.lambda_E
    tau_inv_E = NeuPar.tau_inv_E
    tau_inv_I = NeuPar.tau_inv_I
    threshold = NeuPar.threshold
    N = np.int32(sum(NCells))
    ind_break = np.cumsum(NCells[1:],dtype=np.int32)[:-1]
    nE = NCells[0]

    wDE = NetPar.wDE
    wEP = NetPar.wEP
    wDS = NetPar.wDS 
    wPE = NetPar.wPE
    wPP = NetPar.wPP
    wPS = NetPar.wPS
    wPV = NetPar.wPV
    wSE = NetPar.wSE
    wSP = NetPar.wSP
    wSS = NetPar.wSS
    wSV = NetPar.wSV
    wVE = NetPar.wVE
    wVP = NetPar.wVP
    wVS = NetPar.wVS
    wVV = NetPar.wVV
  
    TEP = NetPar.TEP
    TDS = NetPar.TDS
    TPE = NetPar.TPE 
    TPS = NetPar.TPS
    TPV = NetPar.TPV 
    
    SD = StimPar.SD
    NStim = np.int32(len(StimPar.stim_visual))
    stim_visual = iter(StimPar.stim_visual)
    stim_motor = iter(StimPar.stim_motor)
    neurons_visual = StimPar.neurons_visual
    neurons_motor = StimPar.neurons_motor
    inp_ext_soma = StimPar.inp_ext_soma
    inp_ext_dend = StimPar.inp_ext_dend
    CT = StimPar.CT
    
    dt = SimPar.dt
    stim_length = SimPar.stim_length
    Tstop = (stim_length*NStim).astype(dtype)
    num_time_steps = np.int32(stim_length/dt)
    
    eta = LearnPar.eta
    rho_E = LearnPar.rho_E
    rho_P = LearnPar.rho_P
    decay = LearnPar.decay
    flagRule = LearnPar.flagRule
    
    StimSoma_E = np.zeros(nE, dtype=dtype)
    StimSoma_P = np.zeros(NCells[1], dtype=dtype)
    StimSoma_S = np.zeros(NCells[2], dtype=dtype)
    StimSoma_V = np.zeros(NCells[3], dtype=dtype)
    StimDend = np.zeros(NCells[0], dtype=dtype)
    stim_IN = np.zeros(N-nE, dtype=dtype)
    rE = np.zeros(nE, dtype=dtype)
    rP = np.zeros(NCells[1], dtype=dtype)
    rS = np.zeros(NCells[2], dtype=dtype)
    rV = np.zeros(NCells[3], dtype=dtype)
    activity_dend = np.zeros(nE, dtype=dtype)
    MeanWeightVec = np.zeros(5, dtype=dtype)
    idx = np.int32(0)
    
    noise_soma = np.zeros((N,num_time_steps),dtype=dtype)
    noise_dend = np.zeros((NCells[0],num_time_steps),dtype=dtype)
    
    nstep = SavePar.nstep
    ijk = SavePar.ijk
    if nstep>stim_length/dt:
        print('Warning: Data will not be saved. Decrease nstep!')   
                
    path = 'Results/Data/' + folder
    if not os.path.exists(path):
        os.mkdir(path)

    hdf = tables.open_file(path + '/Data_PlasticNetwork_' + fln + '.hdf', 'w')
    atom = tables.Float32Atom()
    hdf_rates = hdf.create_carray(hdf.root, 'rates', atom, (np.int32(Tstop/(nstep*dt))+1,len(ijk)+1))
    hdf_weights = hdf.create_carray(hdf.root, 'weights', atom, (np.int32(Tstop/(nstep*dt))+1,6)) 
        
    # main loop
    for s in range(NStim):

        V = next(stim_visual)
        M = next(stim_motor)
        noise_soma[:] = np.random.normal(0,SD,size=(N, num_time_steps))
        noise_dend[:] = np.random.normal(0,SD,size=(NCells[0], num_time_steps))
        
        for tstep in range(num_time_steps):
            
            stim_IN[:] = gain*(V*neurons_visual[nE:] + M*neurons_motor[nE:] + noise_soma[nE:,tstep]) + inp_ext_soma[nE:]
            StimSoma_E[:] = V*neurons_visual[:nE] + noise_soma[:nE,tstep] + inp_ext_soma[:nE]
            StimSoma_P[:], StimSoma_S[:], StimSoma_V[:] = np.split(stim_IN, ind_break)
            StimDend[:] = M*neurons_motor[:nE] + noise_dend[:,tstep] + inp_ext_dend
    
            activity_dend[:] = RateDynamics(tau_inv_E, tau_inv_I, lambda_E, lambda_D, 
                                            gain, threshold, CaSpike, CaThreshold, FlagDendRec, 
                                            wEP, wDS, wDE, 
                                            wPE, wPP, wPS, wPV, 
                                            wSE, wSP, wSS, wSV, 
                                            wVE, wVP, wVS, wVV,
                                            rE, rP, rS, rV,
                                            StimSoma_E, StimSoma_P, StimSoma_S, StimSoma_V, StimDend,
                                            dt)
            MeanWeightVec[:] = WeightDynamics(wEP, wDS, wPE, wPS, wPV,
                                              TEP, TDS, TPE, TPS, TPV,
                                              eta, rho_E, rho_P, decay, flagRule,
                                              rE, rP, rS, rV, activity_dend, FlagDendRec, CT)
            
            if ((tstep+1) % nstep==0):
                hdf_weights[idx,0] = s*stim_length + (tstep+1)*dt
                hdf_weights[idx,1:] = MeanWeightVec 
                hdf_rates[idx,0] = s*stim_length + (tstep+1)*dt
                hdf_rates[idx,1] = rE[ijk[0]] 
                hdf_rates[idx,2] = rP[ijk[1]] 
                hdf_rates[idx,3] = rS[ijk[2]] 
                hdf_rates[idx,4] = rV[ijk[3]] 
                idx += np.int32(1)
            
        if (s % 20 == 0):
            print('Stimuli', str(s+1), '/', str(NStim))
            hdf.flush()
        
    hdf.flush()
    hdf.close()
 
    return 


def BackgroundInput(NeuPar: Neurons, NetPar: Network, r0: list = None, Xternal: list = None) -> RateVec:
    
    NCells = NeuPar.NCells
    Nb = np.cumsum(NCells, dtype=np.int32)
    FlagDendRec = NeuPar.FlagDendRec
    gain = NeuPar.gain
    
    inp_ext_soma = np.zeros(np.sum(NCells), dtype=dtype)
    inp_ext_dend = np.zeros(NCells[0], dtype=dtype)
    
    wSE = NetPar.wSE
    wSV = NetPar.wSV
    wVE = NetPar.wVE
    
    if (Xternal is not None and r0 is None):
        
        inp_ext_soma[:Nb[0]] = Xternal[0]
        inp_ext_dend[:] = Xternal[-1]
        inp_ext_soma[Nb[0]:Nb[1]] = Xternal[1]
        inp_ext_soma[Nb[1]:Nb[2]] = Xternal[2]
        inp_ext_soma[Nb[2]:] = Xternal[3]
    
    elif(Xternal is None and r0 is not None):
        
        wEP = NetPar.wEP
        wPE = NetPar.wPE
        wPP = NetPar.wPP
        wPS = NetPar.wPS
        wPV = NetPar.wPV
        wSP = NetPar.wSP
        wSS = NetPar.wSS
        wVP = NetPar.wVP
        wVS = NetPar.wVS
        wVV = NetPar.wVV
        wDE = NetPar.wDE
        wDS = NetPar.wDS
        
        threshold = NeuPar.threshold
        lambda_E = NeuPar.lambda_E
        lambda_D = NeuPar.lambda_D
        
        rE_base = np.repeat(r0[0],NCells[0])
        rP_base = np.repeat(r0[1],NCells[1])
        rS_base = np.repeat(r0[2],NCells[2])
        rV_base = np.repeat(r0[3],NCells[3])
        
        if FlagDendRec==1:
            Dend = np.maximum(lambda_D*(wDE @ rE_base + wDS @ rS_base),0)
            inp_ext_soma[:Nb[0]] = (rE_base/gain + threshold - Dend)/(1-lambda_E) - wEP @ rP_base
        else:
            inp_ext_soma[:Nb[0]] = (rE_base/gain + threshold - lambda_D*(wDE @ rE_base + wDS @ rS_base))/(1-lambda_E) - wEP @ rP_base
        inp_ext_soma[Nb[0]:Nb[1]] = rP_base - wPE @ rE_base - wPP @ rP_base - wPS @ rS_base - wPV @ rV_base
        inp_ext_soma[Nb[1]:Nb[2]] = rS_base - wSE @ rE_base - wSP @ rP_base - wSS @ rS_base - wSV @ rV_base
        inp_ext_soma[Nb[2]:] = rV_base - wVE @ rE_base - wVP @ rP_base - wVS @ rS_base - wVV @ rV_base
        
    return inp_ext_soma, inp_ext_dend



def SaveNetworkPara(NeuPar: Neurons, NetPar: Network, 
                    StimPar: Stimulation, LearnPar: Learning,
                    folder: str, fln: str = ''):
    
    path = 'Results/Data/' + folder
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    filename = path + '/Data_NetworkParameters_' + fln + '.pickle'
    
    if LearnPar==None:
        with open(filename,'wb') as f:
            pickle.dump([NeuPar,NetPar, StimPar],f)
    else:
         with open(filename,'wb') as f:
            pickle.dump([NeuPar, NetPar, StimPar, LearnPar],f)   
            
