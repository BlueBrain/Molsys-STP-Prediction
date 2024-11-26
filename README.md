
# STP prediction from gene expression

Short-term synaptic plasticity (STP) in cortical synapses depends on the type of pre- and postsynaptic neurons.
Direct STP measurments are available for a few hundreds of synaptic pair types (spt) defined in electrophysiological studies
based on neuron location, molecular markers and other features.
On the other hand with the scRNA sequencing it become possible to characterise the complete transcriptomic space of the brain
neurons [].

Statistical modeling of STP dependence on scRNA expression potentially provides a way for prediction of STP parameters for
synaptic pair types when direct electrophysiological recordings are not available. 

The repositary contains a database of Short term plasticity phenomenological models for 93 synaptic pair types in neocortex and
hippocampus defined by major molecular markers and regional positions of pre- and postsynaptic cells. STP modeling was based on 
published electrophysiological studies data [] as well as Allen institute for brain studies (AIBS) synaptic physiology dataset []
The STP modeling data were combined with scRNA expression data for pre- and postsynaptic cell types in neocortex and hippocampus
from Allen institute []. 

The combined database was used for training statistical models of gene expression to STP dependence. The repositary contains 
tools for building and testing these models using approaches implemented in sklearn statistical learning package [] as well as a
pytorch implementation of the hierarchical linear modeling (HLM) approach based on approximate minimum description length principle [].

# Installation

The repositary depends on numpy, pandas, sklearn, pyplot and ...
The HLM approach requires pytorch with cuda installed and a GPU available. 

# Usage

**Classification_genes_to_STP_v1.ipynb** jupyter notebook contains an example pipeline for building and testing sklearn and HLM
regression and classification models for gene expression to STP dependence. 

The HLM algorithm can be runned on the BBP5 server. 
This is an approximate instruction to do this:
1. In the 1st Terminal window login to the bbp5 server:

    **ssh [username]@bbpv1.epfl.ch**
2. Reserve a GPU node:

    **salloc --account=proj64 --partition=interactive -t 4:00:00 --**
    **constraint=volta --gres=gpu:1**
3. In the 2nd Terminal window arange a tunnel for data excange with the GPU node substituting the hostname of the reserved GPU node
in place of [bbp5 node name]: 

    **ssh -N -f -L localhost:1777:localhost:1777 [username]@[bbp5 node name].bbp.epfl.ch**

4. In the 1st Terminal window start a jupyter sever session:

    **jupyter lab --no-browser --port=1777**
5. Open jupyter lab dashboard in a browser window. Then open and run the **Classification_genes_to_STP_v1.ipynb** notebook.

Note: It is assumed that the repository is cloned or downoaded to your bbp5 working directory and that all required packages are installed
in your bbp5 working space. Most of the required packages can be installed with conda package manager provided with Miniconda []. See
Miniconda installation instructions for details [].

Additional data can be downloaded at: https://drive.google.com/file/d/1aK_xVtPyv3Ns3XQgHQkPd5Bq6IlkIiO9/view?usp=sharing

#Funding
The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

#Copyright
Copyright (c) 2024 Blue Brain Project/EPFL



 
