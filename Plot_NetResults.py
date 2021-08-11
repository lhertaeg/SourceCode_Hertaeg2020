#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting routines 
"""

# %% Import  & settings

import os
import tables
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap

from Analyse_Network import activity_realtive_to_base, Currents2PC_vs_time, weights_steady_state_theo
from Analyse_Network import pathway_strengths, MeanCurrent2PC_Phases

PC_cmap = LinearSegmentedColormap.from_list(name='PC_cmap', colors=['#212121','#DBB3B7','#9e3039']) 

# %% functions

def Plot_Comparison_QTvsCT(NeuPar, folder, file_QT, file_CT, minFS = 12, dpi = 500, G = None):
    
    ### Analysis 
    t_QT, _ , _, dR_over_R_pop_QT, _, _, _, _, _ = activity_realtive_to_base(NeuPar, folder, file_QT)
    t_CT, _ , _, dR_over_R_pop_CT, _, _, _, _, _ = activity_realtive_to_base(NeuPar, folder, file_CT)
    
    rate_QT = dR_over_R_pop_QT[:,0]
    rate_CT = dR_over_R_pop_CT[:,0]
    
    ### Plotting
    if G is None:
        GG = gridspec.GridSpec(2,1)
    else:
        GG = G
    
    ax1 = plt.subplot(GG[0,0])
    ax2 = plt.subplot(GG[1,0])
    
    ax1.axvspan(3000,4000,color='#F4F3EB')
    ax1.plot(t_QT[(t_QT>2800.0) & (t_QT<4500.0)],rate_QT[(t_QT>2800.0) & (t_QT<4500.0)],color='#9e3039',lw=1)
    ax1.plot(t_CT[(t_CT>2800.0) & (t_CT<4500.0)],rate_CT[(t_CT>2800.0) & (t_CT<4500.0)],color='#9e3039',lw=1,ls='--')
    ax1.set_xticks([3000,4000]), ax1.set_xticklabels([0,1])
    ax1.set_ylabel(r'$\Delta$R/R (%)',fontsize=minFS)
    ax1.tick_params(axis='both', which='both', size=2.0)
    ax1.legend(['QT','CT'],loc=0,fontsize=minFS-1,handlelength=1.5,frameon=False,handletextpad=0.3)
    ax1.set_title('Mismatch',fontsize=minFS, pad=2)
    sns.despine(ax=ax1)
    
    ax2.axvspan(5000,6000,color='#F4F3EB')
    ax2.plot(t_QT[(t_QT>4800.0) & (t_QT<6500.0)],rate_QT[(t_QT>4800.0) & (t_QT<6500.0)],color='#9e3039',lw=1)
    ax2.plot(t_CT[(t_CT>4800.0) & (t_CT<6500.0)],rate_CT[(t_CT>4800.0) & (t_CT<6500.0)],color='#9e3039',lw=1,ls='--')
    ax2.set_xticks([5000,6000]), ax2.set_xticklabels([0,1])
    ax2.set_xlabel('Time (s)',fontsize=minFS)
    ax2.set_ylabel(r'$\Delta$R/R (%)',fontsize=minFS)
    ax2.tick_params(axis='both', which='both', size=2.0)
    ax2.legend(['QT','CT'],loc=0,fontsize=minFS-1,handlelength=1.5,frameon=False,handletextpad=0.3)
    ax2.set_title('Playback',fontsize=minFS, pad=2)
    sns.despine(ax=ax2)
    
    ### Save
    PathFig = 'Results/Figures/' + folder
    if not os.path.exists(PathFig):
        os.mkdir(PathFig)
    
    plt.savefig(PathFig + '/Fig_Comp_QTvsCT.png', bbox_inches='tight',transparent=True,dpi=dpi)
    plt.close()
    

def Plot_Current2PC_OptoStim(NeuPar, NetPar, StimPar, SimPar, ListAllFiles, StimExtra_all,
                                folder: str, fln: str = '', minFS = 12, dpi = 500, LegendFlag = False, G = None):
    
    ### Analysis
    NetCurrPhases = np.zeros((len(ListAllFiles),4), dtype=np.float32())
    for i in range(len(ListAllFiles)):
        fln_single = ListAllFiles[i]
        NetCurrPhases[i,:] = MeanCurrent2PC_Phases(NeuPar, NetPar, StimPar, SimPar, folder, fln_single)
       
    ### Plotting
    if G is None:
        GG = gridspec.GridSpec(1,1)
    else:
        GG = G
    
    ax = plt.subplot(GG[0,0])
    ax.plot(StimExtra_all,NetCurrPhases[:,0],'.-',color='#416788',lw=1.5,ms=2)
    ax.plot(StimExtra_all,NetCurrPhases[:,1],'.-',color='#C33C54',lw=1.5,ms=2)
    ax.plot(StimExtra_all,NetCurrPhases[:,2],'.-',color='#FFA62B',lw=1.5,ms=2)
    ax.plot(StimExtra_all,NetCurrPhases[:,3],'.-',color='#81D2C7',lw=1.5,ms=2)
    if LegendFlag:
        ax.legend(['FB','MM','PB','BL'],loc=0,fontsize=minFS-1, handlelength=2)
    ax.fill_between(StimExtra_all,NeuPar.threshold, np.min(NetCurrPhases)*0.97 ,color='#E0E0E2', alpha=0.5)
    ax.set_xlim([StimExtra_all[0],StimExtra_all[-1]])
    ax.set_ylim(bottom=np.min(NetCurrPhases)*0.97)
    ax.set_ylabel('Net current in PCs (pA)',fontsize=minFS)
    ax.set_xlabel(r'stimulus (s$^{-1}$)',fontsize=minFS)
    ax.tick_params(axis='both', which='both', size=2.0, labelsize = minFS)
    sns.despine(ax=ax)
    
    ### Save
    PathFig = 'Results/Figures/' + folder
    if not os.path.exists(PathFig):
        os.mkdir(PathFig)
    
    plt.savefig(PathFig + '/NetCurrent2PC_OptoStim_' + fln + '.png', bbox_inches='tight',transparent=True,dpi=dpi)
    plt.close()
    
    return NetCurrPhases


def Plot_Response_StimuliDiff(NeuPar, factor_all, folder: str, minFS = 12, dpi = 500):
    
    ### Analysis
    Resp_MM = np.zeros(len(factor_all))
    Resp_PB = np.zeros(len(factor_all))
    
    for i in range(len(factor_all)):
        
        fln =  'After_Test_' + str(i)
        t, _, _, _, dR_over_R_pop_phases, _, _, _, _ = activity_realtive_to_base(NeuPar, folder, fln)
        Resp_MM[i] = dR_over_R_pop_phases[0,1]
        Resp_PB[i] = dR_over_R_pop_phases[0,2]
    
    ### Plotting & Save
    PathFig = 'Results/Figures/' + folder
    if not os.path.exists(PathFig):
        os.mkdir(PathFig)    
    
    plt.figure()
    plt.plot(1- factor_all, Resp_MM, '.-', color='#9e3039')
    plt.xlabel('(motor-visual)/motor',fontsize=minFS)
    plt.ylabel(r'$\Delta$R/R (%)',fontsize=minFS)
    plt.title('Mismatch (MM)',fontsize=minFS)
    plt.tick_params(axis='both', which='both', size=2.0)
    sns.despine()
    plt.savefig(PathFig + '/Plot_Response_StimuliDiff_MM.png', bbox_inches='tight',transparent=True,dpi=dpi)
    plt.close()
    
    plt.figure()
    plt.plot(1 - factor_all, Resp_PB, '.-', color='#9e3039')
    plt.ylim([-10,10])
    plt.xlabel('(visual-motor)/visual',fontsize=minFS)
    plt.ylabel(r'$\Delta$R/R (%)',fontsize=minFS)
    plt.title('Playback (PB)',fontsize=minFS)
    plt.tick_params(axis='both', which='both', size=2.0)
    sns.despine()
    plt.savefig(PathFig + '/Plot_Response_StimuliDiff_PB.png', bbox_inches='tight',transparent=True,dpi=dpi)
    plt.close()
    

def Plot_MM_Comparison_Training(NeuPar, fln_QT, fln_NT, folder: str, minFS = 12, dpi = 500):
    
    ### Analysis
    t_QT, _, _, dR_over_R_pop_QT, _, _, _, _, _ = activity_realtive_to_base(NeuPar, folder, fln_QT)
    t_NT, _, _, dR_over_R_pop_NT, _, _, _, _, _ = activity_realtive_to_base(NeuPar, folder, fln_NT)
    
    MM_rates_QT = dR_over_R_pop_QT[(t_QT>2800) & (t_QT<5000),:]
    MM_rates_NT = dR_over_R_pop_NT[(t_NT>2800) & (t_NT<5000),:]
    t_QT = t_QT[(t_QT>2800) & (t_QT<5000)]
    t_NT = t_NT[(t_NT>2800) & (t_NT<5000)]

    ### Plotting
    G = gridspec.GridSpec(1,4)
    plt.figure(num=None, facecolor='w', edgecolor='k', figsize=(15, 5), tight_layout='true')
    ax5 = plt.subplot(G[0,0])
    ax6 = plt.subplot(G[0,1])
    ax7 = plt.subplot(G[0,2])
    ax8 = plt.subplot(G[0,3])
    
    ax5.axvspan(3000,4000,color='#F4F3EB')
    ax5.plot(t_NT,MM_rates_NT[:,0],color='#DBB3B7')
    ax5.plot(t_QT,MM_rates_QT[:,0],color='#9e3039')
    ax5.set_xticks([3000,4000]), ax5.set_xticklabels([0,1])
    ax5.set_xlabel('Time (s)',fontsize=minFS), ax5.set_ylabel(r'$\Delta$R/R (%)',fontsize=minFS)
    ax5.legend(['NT','QT'],loc=4,fontsize=minFS,handlelength=1,frameon=False,handletextpad=0.2)
    ax5.set_title('PC population',fontsize=minFS,color='#9e3039',pad=2)#,loc='left')
    ax5.tick_params(axis='both', which='both', size=2.0)
    sns.despine(ax=ax5)
    
    ax6.axvspan(3000,4000,color='#F4F3EB')
    ax6.plot(t_NT,MM_rates_NT[:,1],color='#053c5e',alpha=0.5)
    ax6.plot(t_QT,MM_rates_QT[:,1],color='#053c5e')
    ax6.set_xticks([3000,4000]), ax6.set_xticklabels([0,1])
    ax6.set_xlabel('Time (s)',fontsize=minFS)
    ax6.legend(['NT','QT'],loc=1,fontsize=minFS,handlelength=1,frameon=False,handletextpad=0.2)
    ax6.set_title('PV population',fontsize=minFS,color='#053c5e',pad=2)
    ax6.tick_params(axis='both', which='both', size=2.0)
    sns.despine(ax=ax6)
    
    ax7.axvspan(3000,4000,color='#F4F3EB')
    ax7.plot(t_NT,MM_rates_NT[:,2],color='#87b2ce',alpha=0.5)
    ax7.plot(t_QT,MM_rates_QT[:,2],color='#87b2ce')
    ax7.set_xticks([3000,4000]), ax7.set_xticklabels([0,1])
    ax7.set_xlabel('Time (s)',fontsize=minFS)
    ax7.legend(['NT','QT'],loc=4,fontsize=minFS,handlelength=1,frameon=False,handletextpad=0.2)
    ax7.set_title('SOM population',fontsize=minFS,color='#87b2ce',pad=2)
    ax7.tick_params(axis='both', which='both', size=2.0)
    sns.despine(ax=ax7)
    
    ax8.axvspan(3000,4000,color='#F4F3EB')
    ax8.plot(t_NT,MM_rates_NT[:,3],color='#51a76d',alpha=0.5)
    ax8.plot(t_QT,MM_rates_QT[:,3],color='#51a76d')
    ax8.set_xticks([3000,4000]), ax8.set_xticklabels([0,1])
    ax8.set_xlabel('Time (s)',fontsize=minFS)
    ax8.legend(['NT','QT'],loc=4,fontsize=minFS,handlelength=1,frameon=False,handletextpad=0.2)
    ax8.set_title('VIP population',fontsize=minFS,color='#51a76d',pad=2)
    ax8.tick_params(axis='both', which='both', size=2.0)
    sns.despine(ax=ax8)
    
    ### Save
    PathFig = 'Results/Figures/' + folder
    if not os.path.exists(PathFig):
        os.mkdir(PathFig)
    
    plt.savefig(PathFig + '/Plot_MM_Comparison_Training.png', bbox_inches='tight',transparent=True,dpi=dpi)
    plt.close()
    

def BarPlot_Comparison_Training(NeuPar, fln_QT, fln_NT, folder: str, minFS = 12, dpi = 500):
    
    NCells = NeuPar.NCells
    NE = NCells[0]
    
    ### Analysis
    MM_Perc, MM_Resp = np.zeros(2), np.zeros(2)
    
    _, _, _, _, _, _, dR_over_R_MM_phases_QT, _, MM_Sign_QT = activity_realtive_to_base(NeuPar, folder, fln_QT)
    _, _, _, _, _, _, dR_over_R_MM_phases_NT, _, MM_Sign_NT = activity_realtive_to_base(NeuPar, folder, fln_NT)
     
    MM_Perc[0] = 100.0 * np.sum(MM_Sign_NT)/NE
    MM_Perc[1] = 100.0 * np.sum(MM_Sign_QT)/NE
    
    MM_Resp[0] = np.mean(dR_over_R_MM_phases_NT)
    MM_Resp[1] = np.mean(dR_over_R_MM_phases_QT)
    
    ### Plotting
    G = gridspec.GridSpec(1,2)
    ax3, ax4 = plt.subplot(G[0,0]), plt.subplot(G[0,1])
    
    ax3.bar([0,1],MM_Perc,color=['#DBB3B7','#9e3039'],width=0.5)
    ax3.set_xlim([-0.5,1.5])
    ax3.set_xticks([0,1]), ax3.set_xticklabels(['NT','QT'],fontsize=minFS)
    ax3.tick_params(axis='both', which='both', size=2.0)
    ax3.set_ylabel('# nPE neurons (%)',fontsize=minFS)
    ax3.set_title('Number of nPE neurons\n after learning',fontsize=minFS)
    sns.despine(ax=ax3)
    
    ax4.bar([0,1],MM_Resp,color=['#DBB3B7','#9e3039'],width=0.5)
    ax4.set_xlim([-0.5,1.5])
    ax4.set_xticks([0,1]), ax4.set_xticklabels(['NT','QT'],fontsize=minFS)
    ax4.tick_params(axis='both', which='both', size=2.0)
    ax4.set_ylabel(r'$\Delta$R/R (%)',fontsize=minFS)
    ax4.set_title('nPE neuron response \nin mismatch phase',fontsize=minFS)
    sns.despine(ax=ax4)
    
    ### Save
    PathFig = 'Results/Figures/' + folder
    if not os.path.exists(PathFig):
        os.mkdir(PathFig)
    
    plt.savefig(PathFig + '/BarPlot_Comparison_Training.png', bbox_inches='tight',transparent=True,dpi=dpi)
    plt.close()


def Heatmap_IN_manipulation(NeuPar, Panel, ManipulationType, folder: str, minFS = 12, dpi = 500):
    
    PathData = 'Results/Data/' + folder
    
    NCells = NeuPar.NCells
    PanelABC = ['a', 'b', 'c', 'd']
    ManipulationABC = ['act', 'inact']
    IN_type = ['PV', 'SOM', 'VIP'] 
    
    ### Analysis
    for i in range(3):
        
        fln = PanelABC[Panel-1] + '_' + ManipulationABC[ManipulationType-1] + '_' + IN_type[i]
        arr = np.loadtxt(PathData + '/Data_StaticNetwork_' + fln + '.dat',delimiter=' ')
        t, R = arr[:,0], arr[:,1:NCells[0]+1]
        
        if i==0:
           rate = np.zeros((3,len(t[t>500.0]))) 
        
        rE = np.mean(R,1)
        r_BL = np.mean(rE[(t>500.0) & (t<1000.0)])
        r = (rE[t>500.0]-r_BL)
        Dev_max = max(abs(r))
        
        if Dev_max >= 0.2:
            rate[i,:] = r/Dev_max
            
    ### Plotting
    plt.figure()
    
    data = pd.DataFrame(rate, columns=np.round(t[t>500.0]), index=range(3))
    ax = sns.heatmap(data,cmap=PC_cmap,xticklabels=False,yticklabels=False,cbar=False,vmin=-1,vmax=1)
    
    ax.axhline(1,c='w')
    ax.axhline(2,c='w')
    ax.axhline(3,c='w')
    
    ### Save
    PathFig = 'Results/Figures/' + folder
    if not os.path.exists(PathFig):
        os.mkdir(PathFig)
    
    fln = PanelABC[Panel-1] + '_' + ManipulationABC[ManipulationType-1]
    plt.savefig(PathFig + '/Heatmap_IN_manipulation_' + fln + '.png', bbox_inches='tight',transparent=True,dpi=dpi)
    plt.close()
        
    
def Plot_PopulationRate(NeuPar, folder: str, fln: str = '', ini: int = 0, minFS = 12, dpi = 500, G = None):
    
    ### Analysis
    NCells = NeuPar.NCells
    NC = np.cumsum(NCells)
    N = np.int32(sum(NCells))
    
    PathData = 'Results/Data/' + folder
    arr = np.loadtxt(PathData + '/Data_StaticNetwork_' + fln + '.dat',delimiter=' ')
    t, R = arr[:,0], arr[:,1:N+1]
    
    rE = np.mean(R[:,:NC[0]],1)
    rP = np.mean(R[:,NC[0]:NC[1]],1)
    rS = np.mean(R[:,NC[1]:NC[2]],1)
    rV = np.mean(R[:,NC[2]:],1)
    
    ### Plotting
    if G is None:
        GG = gridspec.GridSpec(5-ini,1)
        GG.update(hspace=0.0)
    else:
        GG = G
    
    Label = ['PC', 'PV', 'SOM', 'VIP']
    
    for j in range(ini,4):
        ax = plt.subplot(GG[j+1-ini,0])
        
        if j==0:
            r_BL = np.mean(rE[(t>500.0) & (t<1000.0)])
            r = (rE-r_BL)/r_BL
        elif j==1:
            r_BL = np.mean(rP[(t>500.0) & (t<1000.0)])
            r = (rP-r_BL)/r_BL
        elif j==2:
            r_BL = np.mean(rS[(t>500.0) & (t<1000.0)])
            r = (rS-r_BL)/r_BL
        elif j==3:
            r_BL = np.mean(rV[(t>500.0) & (t<1000.0)])
            r = (rV-r_BL)/r_BL
            
        activity = r/max(abs(r))
        ax.plot(t,activity,'#5E5E5E',lw=0.5)
        ax.text(0,0.55,Label[j],color='#5E5E5E',fontsize=minFS-1)
        
        ax.axvspan(1000,2000,color='#F4F3EB') 
        ax.axvspan(3000,4000,color='#F4F3EB')
        ax.axvspan(5000,6000,color='#F4F3EB')

        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim([500.0,7000.0])
        ax.set_ylim([-1.0,1.05]) 
        ax.axis('off')
           
    ### Save
    PathFig = 'Results/Figures/' + folder
    if not os.path.exists(PathFig):
        os.mkdir(PathFig)
    
    plt.savefig(PathFig + '/PopulationRate_' + fln + '.png', bbox_inches='tight',transparent=True,dpi=dpi)
    plt.close()
  

def Bar_pathways(NeuPar, NetPar, VE, VP, MP, folder: str, fln: str = '', minFS = 12, dpi = 500, pPE_flag = False, GG = None):
    
    ### Analysis
    (Path_SPE, Path_VPE, Path_SVPE, 
     Path_VSPE, Path_INP, Path_PE) = pathway_strengths(NeuPar, NetPar, VE, VP, MP, pPE_flag=pPE_flag)
    
    ### Plotting
    if GG is None:
        plt.figure()
        ax1 = plt.gca()
    else:
        ax1 = plt.subplot(GG[:,:])
    bar_width = 0.45
    
    ax1.bar([0,1.5],Path_INP,width=bar_width,fc='#CB904D',ec='#CB904D',lw=0)
    ax1.bar([0,1.5],Path_SPE,bottom=Path_INP,width=bar_width,fc='#941100',ec='#941100',lw=0)
    ax1.bar([0,1.5],Path_VPE,bottom=Path_INP+Path_SPE,width=bar_width,fc='#FF7E79',ec='#FF7E79',lw=0)  
        
    ax1.bar([0.5,2],Path_PE,width=bar_width,fc='#011993',ec='#011993',lw=0)
    ax1.bar([0.5,2],Path_SVPE,bottom=Path_PE,width=bar_width,fc='#0096FF',ec='#0096FF',lw=0)
    ax1.bar([0.5,2],Path_VSPE,bottom=Path_PE+Path_SVPE,width=bar_width,fc='#76D6FF',ec='#76D6FF',lw=0)
    
    ax1.set_xlim([-0.5,2.5])
    ax1.spines['bottom'].set_position('zero')
    ax1.tick_params(axis='x', which='both', size=0.0)
    ax1.tick_params(axis='y', which='both', size=2.0)
    ax1.set_ylabel('Pathway stgth',fontsize=minFS)
    ax1.set_xticks([0.25,1.75])
    ax1.set_xticklabels(['V','M'],fontsize=minFS)
    sns.despine(ax=ax1)
    
    ### Save
    PathFig = 'Results/Figures/' + folder
    if not os.path.exists(PathFig):
        os.mkdir(PathFig)
    
    plt.savefig(PathFig + '/PathwayStrg_' + fln + '.png', bbox_inches='tight',transparent=True,dpi=dpi)
    plt.close()    


def Plot_Convergence(NeuPar, NetPar, StimPar, LearnPar, stim_max,
                     folder: str, fln: str = '', minFS = 12, dpi = 500):
    
    ### Analysis
    w_theo, NCon = weights_steady_state_theo(NeuPar, NetPar, StimPar, LearnPar, stim_max)
    
    ### Plotting
    plt.figure()
    ax1 = plt.gca()
    Col = ['#76D6FF', '#7A81FF', '#9E3039', '#0096FF','#011993']
    
    PathData = 'Results/Data/' + folder
    fln1 = PathData + '/Data_PlasticNetwork_' + fln + '.hdf'
    f = tables.open_file(fln1, 'r')
    t = f.root.weights[:-1,0]      
    w = f.root.weights[:-1,1:]
 
    # plot normalized weights    
    for k in range(5):
        weight = w[:,k]*NCon[k]
        ax1.plot(t/(1000*60*60),weight/w_theo[k],c=Col[k])

    ax1.set_xlabel('Time (h)',fontsize=minFS)
    ax1.set_ylabel(r'w$_\mathrm{sim}$/w$_\mathrm{theo}$',fontsize=minFS)
    ax1.tick_params(axis='both', which='both', size=2.0)
    ax1.set_ylim([0,3])
    sns.despine(ax=ax1)
    
    ### Save
    PathFig = 'Results/Figures/' + folder
    if not os.path.exists(PathFig):
        os.mkdir(PathFig)
    
    plt.savefig(PathFig + '/Convergence_' + fln + '.png', bbox_inches='tight',transparent=True,dpi=dpi)
    plt.close()
    

def Plot_Currents2PC(NeuPar, NetPar, StimPar_test_long, SimPar_test,
                     folder: str, fln: str = '', minFS = 12, dpi = 500, G = None):
    
    ### Analysis
    t, sExc, sInh, rE = Currents2PC_vs_time(NeuPar, NetPar, StimPar_test_long, SimPar_test, folder, fln)
    
    ### Plotting
    if G is None:
        GG = gridspec.GridSpec(3,1)
        GG.update(hspace=0.5,wspace=0.1)
    else:
        GG = G
    
    T0 = 100.0
    Delta = 20.0
    r_range = 0.75
    
    ax2 = plt.subplot(GG[0:2,0])
    ax2.plot(t[t>T0],sExc[t>T0]+Delta,'#D9AAA7')
    ax2.text(21000,0.9*np.max(sExc),'Exc',color='#D9AAA7',va='top',fontsize=minFS)
    ax2.plot(t[t>T0],sInh[t>T0]-Delta,'#7CA7BF')
    ax2.text(21000,1.1*np.min(sInh),'Inh',color='#7CA7BF',va='top',fontsize=minFS)
    ax2.plot(t[t>T0],(sExc+sInh)[t>T0],'#847B89')
    ax2.text(21000,np.mean(sExc+sInh),'Net',color='#847B89',va='top',fontsize=minFS)
    ax2.set_ylabel('Input (pA)',fontsize=minFS)
    YB = np.max(np.abs(ax2.get_ybound()))
    ax2.set_ylim([-YB,YB])
    ax2.set_xticks([])
    ax2.tick_params(axis='both', which='both', size=2.0)
    sns.despine(ax=ax2,top=True,bottom=True,right=True)
    
    ax3 = plt.subplot(GG[2,0])
    ax3.plot(t[t>T0]/1000.0,rE[t>T0],color='#9e3039')
    ax3.text(21,np.mean(rE),'PC',color='#9e3039',va='top',fontsize=minFS)
    
    if np.std(rE[t>T0])/np.mean(rE[t>T0]) < 0.1:
        y_lb = np.mean(rE[t>T0]) * (1 - r_range) 
        y_ub = np.mean(rE[t>T0]) * (1 + r_range) 
        ax3.set_ylim([y_lb,y_ub])
                 
    ax3.tick_params(axis='both', which='both', size=2.0)
    ax3.set_xticks([0,10,20])
    ax3.set_xlabel('Time (s)',fontsize=minFS)
    ax3.set_ylabel('Rate \n(1/s)',fontsize=minFS,labelpad=9)
    sns.despine(ax=ax3)
    
    ### Save
    PathFig = 'Results/Figures/' + folder
    if not os.path.exists(PathFig):
        os.mkdir(PathFig)
    
    plt.savefig(PathFig + '/Current2PC_' + fln + '.png', bbox_inches='tight',transparent=True,dpi=dpi)
    plt.close()
     

def HeatmapTest(NeuPar, NE, folder: str, fln: str = '', MM_factor = 0, minFS = 12, dpi = 500, G = None):
    
    ### Analysis
    (t, dR_over_R, _,
     _, _, _, _, idx_MM, MM_Sign) = activity_realtive_to_base(NeuPar, folder, fln)
    
    ### Plotting
    if G is None:
        GG = gridspec.GridSpec(9,22)
        GG.update(hspace=0.5, wspace=0.5)
    else:
        GG = G
    
    dR_over_R = dR_over_R[:,:NE]
    isort = np.argsort(np.max(dR_over_R[(t>3200.0) & (t<4000.0),:],0))
    dR_over_R = dR_over_R[:,isort]    
    dR_over_R = dR_over_R[(t>500.0),:]
    
    max_abs_R = np.max(abs(dR_over_R),0)
    Mean_max_abs_R = np.mean(max_abs_R)
    SD_max_abs_R = np.std(max_abs_R)
    vc = Mean_max_abs_R + SD_max_abs_R
    cbar_tick = np.round(0.9*vc,0)

    t0 = np.arange(500.0,7000.0,0.1)
    V, M = np.zeros(len(t0)), np.zeros(len(t0))
    M[(t0>=1000.0) & (t0<2000.0)] = 1.0
    V[(t0>=1000.0) & (t0<2000.0)] = 1.0
    M[(t0>=3000.0) & (t0<4000.0)] = 1.0
    V[(t0>=3000.0) & (t0<4000.0)] = 1.0 * MM_factor
    M[(t0>=5000.0) & (t0<6000.0)] = 1.0 * MM_factor
    V[(t0>=5000.0) & (t0<6000.0)] = 1.0
    
    ax2 = plt.subplot(GG[:-2,:-1])
    cbar_ax = plt.subplot(GG[-1,:-1])
    
    data = pd.DataFrame(dR_over_R.T, columns=np.round(t[(t>500.0)]), index=range(NE))
    ax2 = sns.heatmap(data,ax=ax2, cmap=PC_cmap,xticklabels=False,yticklabels=False,cbar_ax=cbar_ax, 
                      cbar_kws={"orientation": "horizontal",'label':r'$\Delta$R/R (%)', 'ticks': [-cbar_tick,0,cbar_tick]},
                      vmin=-vc,vmax=vc)
    
    ax2.tick_params(axis='both', which='both', size=1.0, pad=2.0)
    ax2.set_yticks([]), ax2.set_xticks([])
    ax2.invert_yaxis()
    ax2.set_ylabel('Neurons',fontsize=minFS)
    
    axS = plt.subplot(GG[:-2,-1])
    
    NeuSig = np.where(MM_Sign[:NE]==1)[0]
    axS.plot([1,2],[range(70),range(70)],color='#B1B1B1',alpha=0.5,lw=3)
    if len(NeuSig)>0:
        IdxMM = np.argsort(isort)[NeuSig]
        axS.plot([1,2],[IdxMM,IdxMM],'-',color='k',lw=2.5)
    axS.set_ylim([0,69]), axS.set_xlim([1,2])
    axS.set_xticks([]), axS.set_yticks([])
    axS.axis('off')
    
    cbar_ax.tick_params(labelsize=minFS)
    cbar_ax.tick_params(size=1.0, pad=2.0)
    cbar_ax.xaxis.label.set_size(minFS)
    
    axI = plt.subplot(GG[-2,:-1])
    axI.plot(t0,V,color='#CB904D')
    axI.plot(t0,M,color='#218380')
    axI.set_xlim([500.0, 7000.0])
    axI.set_xticks([]), axI.set_yticks([])
    sns.despine(ax=axI), plt.axis('off')
    
    ### Save
    PathFig = 'Results/Figures/' + folder
    if not os.path.exists(PathFig):
        os.mkdir(PathFig)
    
    plt.savefig(PathFig + '/Heatmap_' + fln + '.png', bbox_inches='tight',transparent=True,dpi=dpi)
    plt.close()