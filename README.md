# Code for reproduction of Hertäg & Sprekeler (2020)
„Learning prediction error neurons in a canonical interneuron circuit“

This repository provides the code to **qualitatively** reproduce the publication figures, that is, creating, running, analysing and plotting the network model with excitatory cells and three types of inhibitory interneurons. 

To reproduce the panels of the publication figures, please run the corresponding python code (Run_\*.py) with the parameters specified within the script. After the script has been run successfully, it will save the data (Results/Data/\*) and the single panels of the figure (Results/Figures/\*).

Many of the figures involve running a network with synaptic plasticity which may take a while if you use a laptop without sufficient computing power. Also, note that the results may vary slightly because of different random numbers used in simulations. 

Remark 1: Note that we incorporated the gain factor present in Murayama et al. (2009) into the parameters in the paper but not in the simulations. Thus, the weights onto the excitatory neurons have different units than the weights onto interneurons (in case you wonder ;)).

Remark 2: A script for the reproduction of Figure 1–Figure supplement 1 is not explicitly given but the panels of that figure are automatically reproduced when Figure 1, Figure 6 and Figure 6–Figure supplement 1 are run.

In case of questions, problems or requests, feel free to contact the authors. 
