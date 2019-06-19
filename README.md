# EdgeNets
Experimentation on EdgeNets. This is the code used for obtaining the results in the <a href="https://arxiv.org/abs/1903.01298">paper submitted</a> to EUSIPCO 2019. If any part of this code is used, the following paper must be cited: 

E. Isufi, F. Gama, and A. Ribeiro, "Generalizing Graph Convolutional Neural Networks with Edge-Variant Recursions on Graphs," in 27th Eur. Signal Process. Conf. A Coruña, Spain: EURASIP, 2-6 Sep. 2019.

Any questions, comments or suggestions, please e-mail Fernando Gama at fgama@seas.upenn.edu. The specific random seeds and resulting trained models used to get the results that appear in the paper can be obtained by request.

## Datasets
Two experiments are run.

<p>1. The first one, under the name <code>sourceLocalizationEdgeNets.py</code> is a synthetic experiment on a stochastic block model (SBM) graph. The objective of this problem is to localize the community that was the source of a diffused process.</p>

A. Decelle, F. Krzakala, C. Moore, and L. Zdeborová, "<a href="https://journals.aps.org/pre/abstract/10.1103/PhysRevE.84.066106">Asymptotic analysis of the stochastic block model for modular networks and its algorithmic applications</a>," Physical Review E, vol. 84, no. 6, p. 066106, Dec. 2011.

<p>2. The second one, on the file <code>authorshipAttributionEdgeNets.py</code> considers the problem of authorship attribution. The dataset is available under <code>authorData/</code> and the following paper must be cited whenever such dataset is used</p>

S. Segarra, M. Eisen, and A. Ribeiro, "<a href="https://ieeexplore.ieee.org/document/6638728">Authorship attribution through function word adjacency networks</a>," IEEE Trans. Signal Process., vol. 63, no. 20, pp. 5464–5478, Oct. 2015.

## Code
The code is written in Python3 and the machine learning toolbox is PyTorch. Details as follows.

### Dependencies
The following Python libraries are required: <code>os</code>, <code>numpy</code>, <code>matplotlib</code>, <code>pickle</code>, <code>datetime</code>, <code>scipy</code>, <code>torch</code>, <code>hdf5storage</code>, <code>torchvision</code>, <code>operator</code>, <code>tensorboardX</code> and <code>glob</code>, as well as a LaTeX installation.

### Concept
The two main files <code>sourceLocalizationEdgeNets.py</code> and <code>authorshipAttributionEdgeNets.py</code> consists of the two main experiments. The first lines of code on each of those files (after importing the corresponding libraries) have the main (hyper) parameters defined, which could be edited for running experiment under a different scenario.

The directory <code>Modules/</code> contains the implemented architectures <code>architectures.py</code>, the function to simultaneously train multiple models <code>train.py</code>, as well as an auxiliary container to bind together the architecture, the optimizer and the loss function <code>model.py</code> (as well as other utilities).

Under <code>Utils/</code> there are five modules containing utilitary classes and functions, such as those handling the graph <code>graphTools.py</code>, handling the data <code>dataTools.py</code>, the graph machine learning liberary <code>graphML.py</code> which implements the basic layers and functions for handling graph signals, some miscellaneous tools <code>miscTools.py</code> and the visualization tools on tensorboard <code>visualTools.py</code>.

### Run
The code runs by simply executing <code>python sourceLocalizationEdgeNets.py</code> or any of the two other experiments (assuming <code>python</code> is the command for Python3). For running the authorship attribution experiment, the .rar files on <code>authorData/</code> have to be uncompressed.
