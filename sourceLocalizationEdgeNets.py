# 2019/02/26~2019/03/04.
# Fernando Gama, fgama@seas.upenn.edu

# Source localization problem, testing the following models
#   Spectral GNN
#   Polynomial GNN
#   Node Variant GNN (Deg, EDS, SP)
#   Edge Variant GNN
#   Hybrid Edge Variant GNN (Deg, EDS, SP)

# We will not consider any kind of pooling, and just one layer architectures.
# The number of parameters of every architecture will be tried to be kept
# the same (or, at least, the same order).

# The problem is that of source localization. This simulates several graphs and
# runs several data realizations per graph.

# When it runs, it produces the following output:
#   - It trains the specified models and saves the best and the last model
#       of each realization on a directory named 'savedModels'
#   - It saves a pickle file with the torch random state and the numpy random
#       state for reproducibility.
#   - It saves a text file 'hyperparameters.txt' containing the specific
#       (hyper)parameters that control the run, together with the main (scalar)
#       results obtained.
#   - If desired, logs in tensorboardX the training loss and evaluation measure
#       both of the training set and the validation set. These tensorboardX logs
#       are saved in a logsTB directory.
#   - If desired, saves the vector variables of each realization (training and
#       validation loss and evaluation measure, respectively); this is saved
#       both in pickle and in Matlab(R) format. These variables are saved in a
#       trainVars directory.
#   - If desired, plots the training and validation loss and evaluation
#       performance for each of the models, together with the training loss and
#       validation evaluation performance for all models. The summarizing
#       variables used to construct the plots are also saved in both pickle and
#       Matlab(R) format. These plots (and variables) are in a figs directory.

#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

#\\\ Standard libraries:
import os
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import pickle
import datetime
from scipy.io import savemat

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#\\\ Own libraries:
import Utils.graphTools as graphTools
import Utils.dataTools
import Utils.graphML as gml
import Modules.architectures as archit
import Modules.model as model
import Modules.train as train

#\\\ Separate functions:
from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

thisFilename = 'sourceEdgeNets' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + today
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
# Create the file where all the (hyper)parameters are results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    
#\\\ Save seeds for reproducibility
#   PyTorch seeds
torchState = torch.get_rng_state()
torchSeed = torch.initial_seed()
#   Numpy seeds
numpyState = np.random.RandomState().get_state()
#   Collect all random states
randomStates = []
randomStates.append({})
randomStates[0]['module'] = 'numpy'
randomStates[0]['state'] = numpyState
randomStates.append({})
randomStates[1]['module'] = 'torch'
randomStates[1]['state'] = torchState
randomStates[1]['seed'] = torchSeed
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
saveSeed(randomStates, saveDir)

########
# DATA #
########

nNodes = 50 # Number of nodes
graphType = 'SBM' # Type of graph
nCommunities = 5 # Number of communities
probIntra = 0.8 # Intracommunity probability
probInter = 0.2 # Intercommunity probability

nTrain = 10000 # Number of training samples
nValid = int(0.24 * nTrain) # Number of validation samples
nTest = 200 # Number of testing samples
tMax = None # Maximum number of diffusion times (A^t for t < tMax)

nDataRealizations = 10 # Number of data realizations
nGraphRealizations = 10 # Number of graph realizations

#\\\ Save values:
writeVarValues(varsFile,
               {'nNodes': nNodes,
                'graphType': graphType,
                'nCommunities': nCommunities,
                'probIntra': probIntra,
                'probInter': probInter,
                'nTrain': nTrain,
                'nValid': nValid,
                'nTest': nTest,
                'tMax': tMax,
                'nDataRealizations': nDataRealizations,
                'nGraphRealizations': nGraphRealizations})

############
# TRAINING #
############

#\\\ Individual model training options
trainer = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.001 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.CrossEntropyLoss() # This applies a softmax before feeding
    # it into the NLL, so we don't have to apply the softmax ourselves.

#\\\ Overall training options
nEpochs = 20 # Number of epochs
batchSize = 100 # Batch size
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 20 # How many training steps to do the validation

#\\\ Save values
writeVarValues(varsFile,
               {'trainer': trainer,
                'learningRate': learningRate,
                'beta1': beta1,
                'lossFunction': lossFunction,
                'nEpochs': nEpochs,
                'batchSize': batchSize,
                'doLearningRateDecay': doLearningRateDecay,
                'learningRateDecayRate': learningRateDecayRate,
                'learningRateDecayPeriod': learningRateDecayPeriod,
                'validationInterval': validationInterval})

#################
# ARCHITECTURES #
#################

# Select which architectures to train and run

# Select desired node-orderings (for hybrid EV and node variant EV) so that
# the selected privileged nodes follows this criteria
doDegree = True
doSpectralProxies = True
doEDS = True

# Select desired architectures
doSpectralGNN = True
doPolynomialGNN = True
doNodeVariantGNN = True
doEdgeVariantGNN = True
doHybridEdgeVariantGNN = True

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. That is, any new architecture in this part, needs also
# to be coded later on. This is just to be easy to change the parameters once
# the architecture is created. Do not forget to add the name of the architecture
# to modelList.

modelList = []

# Parameters for all models, so we don't need to be changing each one in each
# of the models (this guarantees comparable computational complexity)

nFeatures = 5 # F: number of output features of the only layer
nIndepNodes = 5 # M: number of independent coefficients or priviliged nodes
nShifts = 5 # K: number of shift taps

#\\\\\\\\\\\\
#\\\ MODEL 1: Spectral GNN
#\\\\\\\\\\\\

if doSpectralGNN:
    
    ##############
    # PARAMETERS #
    ##############
    
    hParamsSpectral = {} # Hyperparameters (hParams)
    
    hParamsSpectral['name']= 'SpectralGNN'
    
    #\\\ Architecture parameters
    hParamsSpectral['F'] = [1, nFeatures] # Features per layer
    hParamsSpectral['M'] = [nIndepNodes] # Number of coefficients per layer
    hParamsSpectral['bias'] = True # Decide whether to include a bias term
    hParamsSpectral['sigma'] = nn.ReLU # Selected nonlinearity
    hParamsSpectral['N'] = [nNodes] # When no pooling is used, this have
        # to be the same as the number of nodes
    hParamsSpectral['rho'] = gml.NoPool # Summarizing function
    hParamsSpectral['alpha'] = [1] # These are ignored when there is no pooling,
        # better set it to 1 to make everything slightly faster
    hParamsSpectral['dimLayersMLP'] = [nCommunities] # Dimension of the fully
        # connected layers after the GCN layers
    
    #\\\ Save Values:
    writeVarValues(varsFile, hParamsSpectral)
    modelList += [hParamsSpectral['name']]

#\\\\\\\\\\\\
#\\\ MODEL 2: Polynomial GNN
#\\\\\\\\\\\\

if doPolynomialGNN:
    
    hParamsPolynomial = {} # Hyperparameters (hParams)
    
    hParamsPolynomial['name'] = 'PolynomiGNN' # Name of the architecture
    
    #\\\ Architecture parameters
    hParamsPolynomial['F'] = [1, nFeatures] # Features per layer
    hParamsPolynomial['K'] = [nShifts] # Number of filter taps per layer
    hParamsPolynomial['bias'] = True # Decide whether to include a bias term
    hParamsPolynomial['sigma'] = nn.ReLU # Selected nonlinearity
    hParamsPolynomial['N'] = [nNodes] # Number of nodes to keep at the end of
        # each layer
    hParamsPolynomial['rho'] = gml.NoPool # Summarizing function
    hParamsPolynomial['alpha'] = [1] # alpha-hop neighborhood that is
        #affected by the summary
    hParamsPolynomial['dimLayersMLP'] = [nCommunities] # Dimension of the fully
        # connected layers after the GCN layers
    
    #\\\ Save Values:
    writeVarValues(varsFile, hParamsPolynomial)
    modelList += [hParamsPolynomial['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 3: Node-Variant GNN ordered by Degree
#\\\\\\\\\\\\
    
if doDegree and doNodeVariantGNN:
    
    hParamsNVDeg = {} # Hyperparameters (hParams)
    
    hParamsNVDeg['name'] = 'NdVarGNNDeg' # Name of the architecture
    
    #\\\ Architecture parameters
    hParamsNVDeg['F'] = [1, nFeatures] # Features per layer
    hParamsNVDeg['K'] = [nShifts] # Number of shift taps per layer
    hParamsNVDeg['M'] = [nIndepNodes] # Number of node taps per layer
    hParamsNVDeg['bias'] = True # Decide whether to include a bias term
    hParamsNVDeg['sigma'] = nn.ReLU # Selected nonlinearity
    hParamsNVDeg['N'] = [nNodes] # Number of nodes to keep at the end of
        # each layer
    hParamsNVDeg['rho'] = gml.NoPool # Summarizing function
    hParamsNVDeg['alpha'] = [1] # alpha-hop neighborhood that is
        #affected by the summary
    hParamsNVDeg['dimLayersMLP'] = [nCommunities] # Dimension of the fully
        # connected layers after the GCN layers
    
    #\\\ Save Values:
    writeVarValues(varsFile, hParamsNVDeg)
    modelList += [hParamsNVDeg['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 4: Node-Variant GNN ordered by Spectral Proxies
#\\\\\\\\\\\\
    
if doSpectralProxies and doNodeVariantGNN:
    
    hParamsNVSpr = hParamsNVDeg.copy() # Hyperparameters (hParams)
    
    hParamsNVSpr['name'] = 'NdVarGNNSPr' # Name of the architecture
    
    #\\\ Save Values:
    writeVarValues(varsFile, hParamsNVSpr)
    modelList += [hParamsNVSpr['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 5: Node-Variant GNN ordered by EDS
#\\\\\\\\\\\\
    
if doEDS and doNodeVariantGNN:
    
    hParamsNVEDS = hParamsNVDeg.copy() # Hyperparameters (hParams)
    
    hParamsNVEDS['name'] = 'NdVarGNNEDS' # Name of the architecture
    
    #\\\ Save Values:
    writeVarValues(varsFile, hParamsNVEDS)
    modelList += [hParamsNVEDS['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 6: Edge-Variant GNN
#\\\\\\\\\\\\
    
if doEdgeVariantGNN:
    
    ##############
    # PARAMETERS #
    ##############
    
    hParamsEdgeVariant = {}
    
    hParamsEdgeVariant['name']= 'EdgeVariGNN'
    
    #\\\ Architecture parameters
    hParamsEdgeVariant['F'] = [1, nFeatures] # Features per layer
    hParamsEdgeVariant['K'] = [nShifts] # Number of shift taps per layer
    hParamsEdgeVariant['bias'] = True # Decide whether to include a bias term
    hParamsEdgeVariant['sigma'] = nn.ReLU # Selected nonlinearity
    hParamsEdgeVariant['N'] = [nNodes] # When no pooling is used, this have
        # to be the same
    hParamsEdgeVariant['rho'] = gml.NoPool # Summarizing function
    hParamsEdgeVariant['alpha'] = [1] # These are ignored when there is no pooling,
        # better set it to 1 to make everything slightly faster
    hParamsEdgeVariant['dimLayersMLP'] = [nCommunities] # Dimension of the fully
        # connected layers after the GCN layers
    
    #\\\ Save Values:
    writeVarValues(varsFile, hParamsEdgeVariant)
    modelList += [hParamsEdgeVariant['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 7: Hybrid Edge-Variant GNN ordered by Degree
#\\\\\\\\\\\\
    
if doDegree and doHybridEdgeVariantGNN:
    
    ##############
    # PARAMETERS #
    ##############
    
    hParamsHEVDeg = {}
    
    hParamsHEVDeg['name']= 'HybEVGNNDeg'
    
    #\\\ Architecture parameters
    hParamsHEVDeg['F'] = [1, nFeatures] # Features per layer
    hParamsHEVDeg['K'] = [nShifts] # Number of shift taps per layer
    hParamsHEVDeg['M'] = [nIndepNodes] # Number of selected EV nodes per layer
    hParamsHEVDeg['bias'] = True # Decide whether to include a bias term
    hParamsHEVDeg['sigma'] = nn.ReLU # Selected nonlinearity
    hParamsHEVDeg['N'] = [nNodes] # When no pooling is used, this have
        # to be the same
    hParamsHEVDeg['rho'] = gml.NoPool # Summarizing function
    hParamsHEVDeg['alpha'] = [1] # These are ignored when there is no pooling,
        # better set it to 1 to make everything slightly faster
    hParamsHEVDeg['dimLayersMLP'] = [nCommunities] # Dimension of the fully
        # connected layers after the GCN layers
    
    #\\\ Save Values:
    writeVarValues(varsFile, hParamsHEVDeg)
    modelList += [hParamsHEVDeg['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 8: Hybrid Edge-Variant GNN ordered by Spectral Proxies
#\\\\\\\\\\\\
    
if doSpectralProxies and doHybridEdgeVariantGNN:
    
    ##############
    # PARAMETERS #
    ##############
    
    hParamsHEVSpr = hParamsHEVDeg.copy()
    
    hParamsHEVSpr['name']= 'HybEVGNNSPr'
    
    #\\\ Save Values:
    writeVarValues(varsFile, hParamsHEVSpr)
    modelList += [hParamsHEVSpr['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 9: Hybrid Edge-Variant GNN ordered by EDS
#\\\\\\\\\\\\
    
if doEDS and doHybridEdgeVariantGNN:
    
    ##############
    # PARAMETERS #
    ##############
    
    hParamsHEVEDS = hParamsHEVDeg.copy()
    
    hParamsHEVEDS['name']= 'HybEVGNNEDS'
    
    #\\\ Save Values:
    writeVarValues(varsFile, hParamsHEVEDS)
    modelList += [hParamsHEVEDS['name']]

###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doLogging = False # Log into tensorboard
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)
# Parameters:
printInterval = 0 # After how many training steps, print the partial results
xAxisMultiplierTrain = 100 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 10 # How many validation steps in between those shown,
    # same as above.

#\\\ Save values:
writeVarValues(varsFile,
               {'doPrint': doPrint,
                'doLogging': doLogging,
                'doSaveVars': doSaveVars,
                'doFigs': doFigs,
                'saveDir': saveDir,
                'printInterval': printInterval})

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ Determine processing unit:
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
# Notify:
if doPrint:
    print("Device selected: %s" % device)

#\\\ Logging options
if doLogging:
    from Utils.visualTools import Visualizer
    logsTB = os.path.join(saveDir, 'logsTB')
    logger = Visualizer(logsTB, name='visualResults')

#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each for the trained models.
# It basically is a dictionary, containing a list of lists. The key of the
# dictionary determines de the model, then the first list index determines
# which graph, and the second list index, determines which realization within
# that graph. Then, this will be converted to numpy to compute mean and standard
# deviation (across the graph dimension).
accBest = {} # Accuracy for the best model
accLast = {} # Accuracy for the last model
for thisModel in modelList: # Create an element for each graph realization,
    # each of these elements will later be another list for each realization.
    # That second list is created empty and just appends the results.
    accBest[thisModel] = [None] * nGraphRealizations
    accLast[thisModel] = [None] * nGraphRealizations

####################
# TRAINING OPTIONS #
####################

# Training phase. It has a lot of options that are input through a
# dictionary of arguments.
# The value of this options was decided above with the rest of the parameters.
# This just creates a dictionary necessary to pass to the train function.

trainingOptions = {}

if doLogging:
    trainingOptions['logger'] = logger
if doSaveVars:
    trainingOptions['saveDir'] = saveDir
if doPrint:
    trainingOptions['printInterval'] = printInterval
if doLearningRateDecay:
    trainingOptions['learningRateDecayRate'] = learningRateDecayRate
    trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
trainingOptions['validationInterval'] = validationInterval

#%%##################################################################
#                                                                   #
#                    GRAPH REALIZATION                              #
#                                                                   #
#####################################################################

# Start generating a new graph for each of the number of graph realizations that
# we previously specified.

for graph in range(nGraphRealizations):

    # The accBest and accLast variables, for each model, have a list with a
    # total number of elements equal to the number of graphs we will generate
    # Now, for each graph, we have multiple data realization, so we want, for
    # each graph, to create a list to hold each of those values
    for thisModel in modelList:
        accBest[thisModel][graph] = []
        accLast[thisModel][graph] = []

    #%%##################################################################
    #                                                                   #
    #                    DATA HANDLING                                  #
    #                                                                   #
    #####################################################################

    #########
    # GRAPH #
    #########

    # Create graph
    G = graphTools.Graph(graphType, nNodes, nCommunities, probIntra, probInter)
    G.computeGFT() # Compute the eigendecomposition of the stored GSO

    ################
    # SOURCE NODES #
    ################

    # For the source localization problem, we have to select which ones, of all
    # the nodes, will act as source nodes. This is determined by a list of
    # indices indicating which nodes to choose as sources.
    sourceNodes = []
    # In particular, we select the node with highest degree on each community.
    # (Recall that, in an SBM, all nodes have the same average degree)
    if graphType == 'SBM':
        # We are assuming here that the nodes are ordered in such a way that
        # all nodes belonging to a given community are consecutive.

        # Number of nodes per community: floor division (We're assuming that all
        # communities have the same number of nodes, and if the number of nodes
        # is not divisible, then the first communities take those spare nodes;
        # see Utils.GraphTools for details)
        nNodesC = [nNodes//nCommunities] * nCommunities
        c = 0 # counter for community
        while sum(nNodesC) < nNodes: # If there are still nodes to put in
        # communities do it one for each (balanced communities)
            nNodesC[c] = nNodesC[c] + 1
            c += 1
        # The above 5 lines of code assume a very specific way of creating the
        # SBMs. The objective is to identify which of the nodes belong to which
        # community.

        # Start-end indices for the nodes in each community (all nodes belonging
        # to a given community, are consecutive in the ordering)
        nNodesCIndex = [0] + np.cumsum(nNodesC).tolist()
        degree = np.diag(G.D) # degree vector for all nodes
        for c in range(nCommunities):
            # Choose only the nodes in this community
            thisCommunityDegree = degree[nNodesCIndex[c]:nNodesCIndex[c+1]]
            # Sort them by degree
            degreeSorted = np.argsort(thisCommunityDegree)
            # Keep the one with largest degree (and add the indexing offset)
            sourceNodes += [degreeSorted[-1] + nNodesCIndex[c]]

    # We have now created the graph and selected the source nodes on that graph.
    # So now we proceed to generate random data realizations, different
    # realizations of diffusion processes.

    for realization in range(nDataRealizations):

        ############
        # DATASETS #
        ############

        #   Now that we have the list of nodes we are using as sources, then we
        #   can go ahead and generate the datasets.
        data = Utils.dataTools.SourceLocalization(G, nTrain, nValid, nTest,
                                                  sourceNodes, tMax = tMax)
        data.astype(torch.float64)
        data.to(device)

        #%%##################################################################
        #                                                                   #
        #                    MODELS INITIALIZATION                          #
        #                                                                   #
        #####################################################################

        # This is the dictionary where we store the models (in a model.Model
        # class, that is then passed to training).
        modelsGNN = {}

        # If a new model is to be created, it should be called for here.
        
        #%%\\\\\\\\\\
        #\\\ MODEL 1: Spectral GNN
        #\\\\\\\\\\\\
        
        if doSpectralGNN:

            thisName = hParamsSpectral['name']
    
            # Name the model for this specific realization
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
    
            ##############
            # PARAMETERS #
            ##############
    
            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2
    
            #\\\ Ordering
            S, order = graphTools.permIdentity(G.S/np.max(np.diag(G.E)))
            # order is an np.array with the ordering of the nodes with respect
            # to the original GSO (the original GSO is kept in G.S).
    
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = archit.SpectralGNN(# Graph filtering
                                            hParamsSpectral['F'],
                                            hParamsSpectral['M'],
                                            hParamsSpectral['bias'],
                                            # Nonlinearity
                                            hParamsSpectral['sigma'],
                                            # Pooling
                                            hParamsSpectral['N'],
                                            hParamsSpectral['rho'],
                                            hParamsSpectral['alpha'],
                                            # MLP
                                            hParamsSpectral['dimLayersMLP'],
                                            # Structure
                                            S)
            thisArchit.to(device)
    
            #############
            # OPTIMIZER #
            #############
    
            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            thisLossFunction = lossFunction # (if different from default,
                                            # change it here)
    
            #########
            # MODEL #
            #########
    
            Spectral = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)
    
            modelsGNN[thisName] = Spectral
    
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
        
        #%%\\\\\\\\\\
        #\\\ MODEL 2: Polynomial GNN
        #\\\\\\\\\\\\
        
        if doPolynomialGNN:
    
            thisName = hParamsPolynomial['name']
    
            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
    
            ##############
            # PARAMETERS #
            ##############
    
            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2
    
            #\\\ Ordering
            S, order = graphTools.permIdentity(G.S/np.max(np.diag(G.E)))
            # order is an np.array with the ordering of the nodes with respect
            # to the original GSO (the original GSO is kept in G.S).
    
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = archit.SelectionGNN(# Graph filtering
                                             hParamsPolynomial['F'],
                                             hParamsPolynomial['K'],
                                             hParamsPolynomial['bias'],
                                             # Nonlinearity
                                             hParamsPolynomial['sigma'],
                                             # Pooling
                                             hParamsPolynomial['N'],
                                             hParamsPolynomial['rho'],
                                             hParamsPolynomial['alpha'],
                                             # MLP
                                             hParamsPolynomial['dimLayersMLP'],
                                             # Structure
                                             S)
            # This is necessary to move all the learnable parameters to be
            # stored in the device (mostly, if it's a GPU)
            thisArchit.to(device)
    
            #############
            # OPTIMIZER #
            #############
    
            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            thisLossFunction = lossFunction 
    
            #########
            # MODEL #
            #########
    
            Polynomial = model.Model(thisArchit, thisLossFunction, thisOptim,
                                 thisName, saveDir, order)
    
            modelsGNN[thisName] = Polynomial
    
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
    
        #%%\\\\\\\\\\
        #\\\ MODEL 3: Node-Variant GNN ordered by Degree
        #\\\\\\\\\\\\
        
        if doDegree and doNodeVariantGNN:
    
            thisName = hParamsNVDeg['name']
    
            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
    
            ##############
            # PARAMETERS #
            ##############
    
            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2
    
            #\\\ Ordering
            S, order = graphTools.permDegree(G.S/np.max(np.diag(G.E)))
    
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = archit.NodeVariantGNN(# Graph filtering
                                               hParamsNVDeg['F'],
                                               hParamsNVDeg['K'],
                                               hParamsNVDeg['M'],
                                               hParamsNVDeg['bias'],
                                               # Nonlinearity
                                               hParamsNVDeg['sigma'],
                                               # Pooling
                                               hParamsNVDeg['N'],
                                               hParamsNVDeg['rho'],
                                               hParamsNVDeg['alpha'],
                                               # MLP
                                               hParamsNVDeg['dimLayersMLP'],
                                               # Structure
                                               S)

            thisArchit.to(device)
    
            #############
            # OPTIMIZER #
            #############
    
            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            thisLossFunction = lossFunction
    
            #########
            # MODEL #
            #########
    
            NVDeg = model.Model(thisArchit, thisLossFunction, thisOptim,
                                 thisName, saveDir, order)
    
            modelsGNN[thisName] = NVDeg
    
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
    
        #%%\\\\\\\\\\
        #\\\ MODEL 4: Node-Variant GNN ordered by Spectral Proxies
        #\\\\\\\\\\\\
        
        if doSpectralProxies and doNodeVariantGNN:
    
            thisName = hParamsNVSpr['name']
    
            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
    
            ##############
            # PARAMETERS #
            ##############
    
            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2
    
            #\\\ Ordering
            S, order = graphTools.permSpectralProxies(G.S/np.max(np.diag(G.E)))
    
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = archit.NodeVariantGNN(# Graph filtering
                                               hParamsNVSpr['F'],
                                               hParamsNVSpr['K'],
                                               hParamsNVSpr['M'],
                                               hParamsNVSpr['bias'],
                                               # Nonlinearity
                                               hParamsNVSpr['sigma'],
                                               # Pooling
                                               hParamsNVSpr['N'],
                                               hParamsNVSpr['rho'],
                                               hParamsNVSpr['alpha'],
                                               # MLP
                                               hParamsNVSpr['dimLayersMLP'],
                                               # Structure
                                               S)

            thisArchit.to(device)
    
            #############
            # OPTIMIZER #
            #############
    
            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            thisLossFunction = lossFunction
    
            #########
            # MODEL #
            #########
    
            NVSpr = model.Model(thisArchit, thisLossFunction, thisOptim,
                                 thisName, saveDir, order)
    
            modelsGNN[thisName] = NVSpr
    
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
    
        #%%\\\\\\\\\\
        #\\\ MODEL 5: Node-Variant GNN ordered by EDS
        #\\\\\\\\\\\\
        
        if doEDS and doNodeVariantGNN:
    
            thisName = hParamsNVEDS['name']
    
            # If more than one graph or data realization is going to be carried
            # out, we are going to store all of thos models separately, so that
            # any of them can be brought back and studied in detail.
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
    
            ##############
            # PARAMETERS #
            ##############
    
            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2
    
            #\\\ Ordering
            S, order = graphTools.permEDS(G.S/np.max(np.diag(G.E)))
    
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = archit.NodeVariantGNN(# Graph filtering
                                               hParamsNVEDS['F'],
                                               hParamsNVEDS['K'],
                                               hParamsNVEDS['M'],
                                               hParamsNVEDS['bias'],
                                               # Nonlinearity
                                               hParamsNVEDS['sigma'],
                                               # Pooling
                                               hParamsNVEDS['N'],
                                               hParamsNVEDS['rho'],
                                               hParamsNVEDS['alpha'],
                                               # MLP
                                               hParamsNVEDS['dimLayersMLP'],
                                               # Structure
                                               S)

            thisArchit.to(device)
    
            #############
            # OPTIMIZER #
            #############
    
            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            thisLossFunction = lossFunction
    
            #########
            # MODEL #
            #########
    
            NVEDS = model.Model(thisArchit, thisLossFunction, thisOptim,
                                 thisName, saveDir, order)
    
            modelsGNN[thisName] = NVEDS
    
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
    
        #%%\\\\\\\\\\
        #\\\ MODEL 6: Edge-Variant GNN
        #\\\\\\\\\\\\
        
        if doEdgeVariantGNN:

            thisName = hParamsEdgeVariant['name']
    
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
    
            ##############
            # PARAMETERS #
            ##############
    
            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2
    
            #\\\ Ordering
            S, order = graphTools.permIdentity(G.S/np.max(np.diag(G.E)))
    
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = archit.EdgeVariantGNN(# Graph filtering
                                             hParamsEdgeVariant['F'],
                                             hParamsEdgeVariant['K'],
                                             hParamsEdgeVariant['bias'],
                                             # Nonlinearity
                                             hParamsEdgeVariant['sigma'],
                                             # Pooling
                                             hParamsEdgeVariant['N'],
                                             hParamsEdgeVariant['rho'],
                                             hParamsEdgeVariant['alpha'],
                                             # MLP
                                             hParamsEdgeVariant['dimLayersMLP'],
                                             # Structure
                                             S)
            thisArchit.to(device)
    
            #############
            # OPTIMIZER #
            #############
    
            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            thisLossFunction = lossFunction
    
            #########
            # MODEL #
            #########
    
            EdgeVariant = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)
    
            modelsGNN[thisName] = EdgeVariant
    
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
    
        #%%\\\\\\\\\\
        #\\\ MODEL 7: Hybrid Edge-Variant GNN ordered by Degree
        #\\\\\\\\\\\\
        
        if doDegree and doHybridEdgeVariantGNN:

            thisName = hParamsHEVDeg['name']
    
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
    
            ##############
            # PARAMETERS #
            ##############
    
            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2
    
            #\\\ Ordering
            S, order = graphTools.permDegree(G.S/np.max(np.diag(G.E)))
    
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = archit.HybridEdgeVariantGNN(# Graph filtering
                                                  hParamsHEVDeg['F'],
                                                  hParamsHEVDeg['K'],
                                                  hParamsHEVDeg['M'],
                                                  hParamsHEVDeg['bias'],
                                                  # Nonlinearity
                                                  hParamsHEVDeg['sigma'],
                                                  # Pooling
                                                  hParamsHEVDeg['N'],
                                                  hParamsHEVDeg['rho'],
                                                  hParamsHEVDeg['alpha'],
                                                  # MLP
                                                  hParamsHEVDeg['dimLayersMLP'],
                                                  # Structure
                                                  S)
            thisArchit.to(device)
    
            #############
            # OPTIMIZER #
            #############
    
            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            thisLossFunction = lossFunction 
    
            #########
            # MODEL #
            #########
    
            HEVDeg = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)
    
            modelsGNN[thisName] = HEVDeg
    
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
    
        #%%\\\\\\\\\\
        #\\\ MODEL 8: Hybrid Edge-Variant GNN ordered by Spectral Proxies
        #\\\\\\\\\\\\
        
        if doSpectralProxies and doHybridEdgeVariantGNN:

            thisName = hParamsHEVSpr['name']
    
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
    
            ##############
            # PARAMETERS #
            ##############
    
            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2
    
            #\\\ Ordering
            S, order = graphTools.permSpectralProxies(G.S/np.max(np.diag(G.E)))
    
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = archit.HybridEdgeVariantGNN(# Graph filtering
                                                  hParamsHEVSpr['F'],
                                                  hParamsHEVSpr['K'],
                                                  hParamsHEVSpr['M'],
                                                  hParamsHEVSpr['bias'],
                                                  # Nonlinearity
                                                  hParamsHEVSpr['sigma'],
                                                  # Pooling
                                                  hParamsHEVSpr['N'],
                                                  hParamsHEVSpr['rho'],
                                                  hParamsHEVSpr['alpha'],
                                                  # MLP
                                                  hParamsHEVSpr['dimLayersMLP'],
                                                  # Structure
                                                  S)
            thisArchit.to(device)
    
            #############
            # OPTIMIZER #
            #############
    
            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            thisLossFunction = lossFunction 
    
            #########
            # MODEL #
            #########
    
            HEVSpr = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)
    
            modelsGNN[thisName] = HEVSpr
    
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})
    
        #%%\\\\\\\\\\
        #\\\ MODEL 9: Hybrid Edge-Variant GNN ordered by EDS
        #\\\\\\\\\\\\
        
        if doEDS and doHybridEdgeVariantGNN:

            thisName = hParamsHEVEDS['name']
    
            if nGraphRealizations > 1:
                thisName += 'G%02d' % graph
            if nDataRealizations > 1:
                thisName += 'R%02d' % realization
    
            ##############
            # PARAMETERS #
            ##############
    
            #\\\ Optimizer options
            #   (If different from the default ones, change here.)
            thisTrainer = trainer
            thisLearningRate = learningRate
            thisBeta1 = beta1
            thisBeta2 = beta2
    
            #\\\ Ordering
            S, order = graphTools.permEDS(G.S/np.max(np.diag(G.E)))
    
            ################
            # ARCHITECTURE #
            ################
    
            thisArchit = archit.HybridEdgeVariantGNN(# Graph filtering
                                                  hParamsHEVEDS['F'],
                                                  hParamsHEVEDS['K'],
                                                  hParamsHEVEDS['M'],
                                                  hParamsHEVEDS['bias'],
                                                  # Nonlinearity
                                                  hParamsHEVEDS['sigma'],
                                                  # Pooling
                                                  hParamsHEVEDS['N'],
                                                  hParamsHEVEDS['rho'],
                                                  hParamsHEVEDS['alpha'],
                                                  # MLP
                                                  hParamsHEVEDS['dimLayersMLP'],
                                                  # Structure
                                                  S)
            thisArchit.to(device)
    
            #############
            # OPTIMIZER #
            #############
    
            if thisTrainer == 'ADAM':
                thisOptim = optim.Adam(thisArchit.parameters(),
                                       lr = learningRate, betas = (beta1,beta2))
            elif thisTrainer == 'SGD':
                thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
            elif thisTrainer == 'RMSprop':
                thisOptim = optim.RMSprop(thisArchit.parameters(),
                                          lr = learningRate, alpha = beta1)
    
            ########
            # LOSS #
            ########
    
            thisLossFunction = lossFunction 
    
            #########
            # MODEL #
            #########
    
            HEVEDS = model.Model(thisArchit, thisLossFunction, thisOptim,
                                    thisName, saveDir, order)
    
            modelsGNN[thisName] = HEVEDS
    
            writeVarValues(varsFile,
                           {'name': thisName,
                            'thisTrainer': thisTrainer,
                            'thisLearningRate': thisLearningRate,
                            'thisBeta1': thisBeta1,
                            'thisBeta2': thisBeta2})

        #%%##################################################################
        #                                                                   #
        #                    TRAINING                                       #
        #                                                                   #
        #####################################################################


        ############
        # TRAINING #
        ############

        # On top of the rest of the training options, we pass the identification
        # of this specific graph/data realization.

        if nGraphRealizations > 1:
            trainingOptions['graphNo'] = graph
        if nDataRealizations > 1:
            trainingOptions['realizationNo'] = realization

        # This is the function that trains the models detailed in the dictionary
        # modelsGNN using the data data, with the specified training options.
        train.MultipleModels(modelsGNN, data,
                             nEpochs = nEpochs, batchSize = batchSize,
                             **trainingOptions)

        #%%##################################################################
        #                                                                   #
        #                    EVALUATION                                     #
        #                                                                   #
        #####################################################################

        # Now that the model has been trained, we evaluate them on the test
        # samples.

        # We have two versions of each model to evaluate: the one obtained
        # at the best result of the validation step, and the last trained model.

        ########
        # DATA #
        ########

        xTest, yTest = data.getSamples('test')

        ##############
        # BEST MODEL #
        ##############

        if doPrint:
            print("Total testing accuracy (Best):", flush = True)

        for key in modelsGNN.keys():
            # Update order and adapt dimensions (this data has one input feature,
            # so we need to add that dimension; make it from B x N to B x F x N)
            xTestOrdered = xTest[:,modelsGNN[key].order].unsqueeze(1)

            with torch.no_grad():
                # Process the samples
                yHatTest = modelsGNN[key].archit(xTestOrdered)
                # yHatTest is of shape
                #   testSize x numberOfClasses
                # We compute the accuracy
                thisAccBest = data.evaluate(yHatTest, yTest)

            if doPrint:
                print("%s: %4.2f%%" % (key, thisAccBest * 100.), flush = True)

            # Save value
            writeVarValues(varsFile,
                       {'accBest%s' % key: thisAccBest})

            # Now check which is the model being trained
            for thisModel in modelList:
                # If the name in the modelList is contained in the name with
                # the key, then that's the model, and save it
                # For example, if 'SelGNNDeg' is in thisModelList, then the
                # correct key will read something like 'SelGNNDegG01R00' so
                # that's the one to save.
                if thisModel in key:
                    accBest[thisModel][graph] += [thisAccBest.item()]
                # This is so that we can later compute a total accuracy with
                # the corresponding error.

        ##############
        # LAST MODEL #
        ##############

        # And repeat for the last model

        if doPrint:
            print("Total testing accuracy (Last):", flush = True)

        # Update order and adapt dimensions
        for key in modelsGNN.keys():
            modelsGNN[key].load(label = 'Last')
            xTestOrdered = xTest[:,modelsGNN[key].order].unsqueeze(1)

            with torch.no_grad():
                # Process the samples
                yHatTest = modelsGNN[key].archit(xTestOrdered)
                # yHatTest is of shape
                #   testSize x numberOfClasses
                # We compute the accuracy
                thisAccLast = data.evaluate(yHatTest, yTest)

            if doPrint:
                print("%s: %4.2f%%" % (key, thisAccLast * 100), flush = True)

            # Save values:
            writeVarValues(varsFile,
                       {'accLast%s' % key: thisAccLast})
            # And repeat for the last model:
            for thisModel in modelList:
                if thisModel in key:
                    accLast[thisModel][graph] += [thisAccLast.item()]

############################
# FINAL EVALUATION RESULTS #
############################

# Now that we have computed the accuracy of all runs, we can obtain a final
# result (mean and standard deviation)

meanAccBestPerGraph = {} # Compute the mean accuracy (best) across all
    # realizations data realizations of a graph
meanAccLastPerGraph = {} # Compute the mean accuracy (last) across all
    # realizations data realizations of a graph
meanAccBest = {} # Mean across graphs (after having averaged across data
    # realizations)
meanAccLast = {} # Mean across graphs
stdDevAccBest = {} # Standard deviation across graphs
stdDevAccLast = {} # Standard deviation across graphs

if doPrint:
    print("\nFinal evaluations (%02d graphs, %02d realizations)" % (
            nGraphRealizations, nDataRealizations))

for thisModel in modelList:
    # Convert the lists into a nGraphRealizations x nDataRealizations matrix
    accBest[thisModel] = np.array(accBest[thisModel])
    accLast[thisModel] = np.array(accLast[thisModel])

    # Compute the mean (across realizations for a given graph)
    meanAccBestPerGraph[thisModel] = np.mean(accBest[thisModel], axis = 1)
    meanAccLastPerGraph[thisModel] = np.mean(accLast[thisModel], axis = 1)

    # And now compute the statistics (across graphs)
    meanAccBest[thisModel] = np.mean(meanAccBestPerGraph[thisModel])
    meanAccLast[thisModel] = np.mean(meanAccLastPerGraph[thisModel])
    stdDevAccBest[thisModel] = np.std(meanAccBestPerGraph[thisModel])
    stdDevAccLast[thisModel] = np.std(meanAccLastPerGraph[thisModel])

    # And print it:
    if doPrint:
        print("\t%s: %6.2f%% (+-%6.2f%%) [Best] %6.2f%% (+-%6.2f%%) [Last]" % (
                thisModel,
                meanAccBest[thisModel] * 100,
                stdDevAccBest[thisModel] * 100,
                meanAccLast[thisModel] * 100,
                stdDevAccLast[thisModel] * 100))

    # Save values
    writeVarValues(varsFile,
               {'meanAccBest%s' % thisModel: meanAccBest[thisModel],
                'stdDevAccBest%s' % thisModel: stdDevAccBest[thisModel],
                'meanAccLast%s' % thisModel: meanAccLast[thisModel],
                'stdDevAccLast%s' % thisModel : stdDevAccLast[thisModel]})

#%%##################################################################
#                                                                   #
#                    PLOT                                           #
#                                                                   #
#####################################################################

# Finally, we might want to plot several quantities of interest

if doFigs and doSaveVars:

    ###################
    # DATA PROCESSING #
    ###################

    # Again, we have training and validation metrics (loss and accuracy
    # -evaluation-) for many runs, so we need to carefully load them and compute
    # the relevant statistics from these realizations.

    #\\\ SAVE SPACE:
    # Create the variables to save all the realizations. This is, again, a
    # dictionary, where each key represents a model, and each model is a list
    # of lists, one list for each graph, and one list for each data realization.
    # Each data realization, in this case, is not a scalar, but a vector of
    # length the number of training steps (or of validation steps)
    lossTrain = {}
    evalTrain = {}
    lossValid = {}
    evalValid = {}
    # Initialize the graph dimension
    for thisModel in modelList:
        lossTrain[thisModel] = [None] * nGraphRealizations
        evalTrain[thisModel] = [None] * nGraphRealizations
        lossValid[thisModel] = [None] * nGraphRealizations
        evalValid[thisModel] = [None] * nGraphRealizations
        # Initialize the data realization dimension with empty lists to then
        # append each realization when we load it.
        for G in range(nGraphRealizations):
            lossTrain[thisModel][G] = []
            evalTrain[thisModel][G] = []
            lossValid[thisModel][G] = []
            evalValid[thisModel][G] = []

    #\\\ FIGURES DIRECTORY:
    saveDirFigs = os.path.join(saveDir,'figs')
    # If it doesn't exist, create it.
    if not os.path.exists(saveDirFigs):
        os.makedirs(saveDirFigs)

    #\\\ LOAD DATA:
    # Path where the saved training variables should be
    pathToTrainVars = os.path.join(saveDir,'trainVars')
    # Get all the training files:
    allTrainFiles = next(os.walk(pathToTrainVars))[2]
    # Go over each of them (this can't be empty since we are also checking for
    # doSaveVars to be true, what guarantees that the variables have been saved.
    for file in allTrainFiles:
        # Check that it is a pickle file
        if '.pkl' in file:
            # Open the file
            with open(os.path.join(pathToTrainVars,file),'rb') as fileTrainVars:
                # Load it
                thisVarsDict = pickle.load(fileTrainVars)
                # store them
                nBatches = thisVarsDict['nBatches']
                thisLossTrain = thisVarsDict['lossTrain']
                thisEvalTrain = thisVarsDict['evalTrain']
                thisLossValid = thisVarsDict['lossValid']
                thisEvalValid = thisVarsDict['evalValid']
                if 'graphNo' in thisVarsDict.keys():
                    thisG = thisVarsDict['graphNo']
                else:
                    thisG = 0
                if 'realizationNo' in thisVarsDict.keys():
                    thisR = thisVarsDict['realizationNo']
                else:
                    thisR = 0
                # And add them to the corresponding variables
                for key in thisLossTrain.keys():
                # This part matches each realization (saved with a different
                # name due to identification of graph and data realization) with
                # the specific model.
                    for thisModel in modelList:
                        if thisModel in key:
                            lossTrain[thisModel][thisG] += [thisLossTrain[key]]
                            evalTrain[thisModel][thisG] += [thisEvalTrain[key]]
                            lossValid[thisModel][thisG] += [thisLossValid[key]]
                            evalValid[thisModel][thisG] += [thisEvalValid[key]]
    # Now that we have collected all the results, we have that each of the four
    # variables (lossTrain, evalTrain, lossValid, evalValid) has a list of lists
    # for each key in the dictionary. The first list goes through the graph, and
    # for each graph, it goes through data realizations. Each data realization
    # is actually an np.array.

    #\\\ COMPUTE STATISTICS:
    # The first thing to do is to transform those into a matrix with all the
    # realizations, so create the variables to save that.
    meanLossTrainPerGraph = {}
    meanEvalTrainPerGraph = {}
    meanLossValidPerGraph = {}
    meanEvalValidPerGraph = {}
    meanLossTrain = {}
    meanEvalTrain = {}
    meanLossValid = {}
    meanEvalValid = {}
    stdDevLossTrain = {}
    stdDevEvalTrain = {}
    stdDevLossValid = {}
    stdDevEvalValid = {}
    # Initialize the variables
    for thisModel in modelList:
        meanLossTrainPerGraph[thisModel] = [None] * nGraphRealizations
        meanEvalTrainPerGraph[thisModel] = [None] * nGraphRealizations
        meanLossValidPerGraph[thisModel] = [None] * nGraphRealizations
        meanEvalValidPerGraph[thisModel] = [None] * nGraphRealizations
        for G in range(nGraphRealizations):
            # Transform into np.array
            lossTrain[thisModel][G] = np.array(lossTrain[thisModel][G])
            evalTrain[thisModel][G] = np.array(evalTrain[thisModel][G])
            lossValid[thisModel][G] = np.array(lossValid[thisModel][G])
            evalValid[thisModel][G] = np.array(evalValid[thisModel][G])
            # So, finally, for each model and each graph, we have a np.array of
            # shape:  nDataRealizations x number_of_training_steps
            # And we have to average these to get the mean across all data
            # realizations for each graph
            meanLossTrainPerGraph[thisModel][G] = \
                                    np.mean(lossTrain[thisModel][G], axis = 0)
            meanEvalTrainPerGraph[thisModel][G] = \
                                    np.mean(evalTrain[thisModel][G], axis = 0)
            meanLossValidPerGraph[thisModel][G] = \
                                    np.mean(lossValid[thisModel][G], axis = 0)
            meanEvalValidPerGraph[thisModel][G] = \
                                    np.mean(evalValid[thisModel][G], axis = 0)
        # And then convert this into np.array for all graphs
        meanLossTrainPerGraph[thisModel] = \
                                    np.array(meanLossTrainPerGraph[thisModel])
        meanEvalTrainPerGraph[thisModel] = \
                                    np.array(meanEvalTrainPerGraph[thisModel])
        meanLossValidPerGraph[thisModel] = \
                                    np.array(meanLossValidPerGraph[thisModel])
        meanEvalValidPerGraph[thisModel] = \
                                    np.array(meanEvalValidPerGraph[thisModel])
        # And compute the statistics
        meanLossTrain[thisModel] = \
                            np.mean(meanLossTrainPerGraph[thisModel], axis = 0)
        meanEvalTrain[thisModel] = \
                            np.mean(meanEvalTrainPerGraph[thisModel], axis = 0)
        meanLossValid[thisModel] = \
                            np.mean(meanLossValidPerGraph[thisModel], axis = 0)
        meanEvalValid[thisModel] = \
                            np.mean(meanEvalValidPerGraph[thisModel], axis = 0)
        stdDevLossTrain[thisModel] = \
                            np.std(meanLossTrainPerGraph[thisModel], axis = 0)
        stdDevEvalTrain[thisModel] = \
                            np.std(meanEvalTrainPerGraph[thisModel], axis = 0)
        stdDevLossValid[thisModel] = \
                            np.std(meanLossValidPerGraph[thisModel], axis = 0)
        stdDevEvalValid[thisModel] = \
                            np.std(meanEvalValidPerGraph[thisModel], axis = 0)

    ####################
    # SAVE FIGURE DATA #
    ####################

    # And finally, we can plot. But before, let's save the variables mean and
    # stdDev so, if we don't like the plot, we can re-open them, and re-plot
    # them, a piacere.
    #   Pickle, first:
    varsPickle = {}
    varsPickle['nEpochs'] = nEpochs
    varsPickle['nBatches'] = nBatches
    varsPickle['meanLossTrain'] = meanLossTrain
    varsPickle['stdDevLossTrain'] = stdDevLossTrain
    varsPickle['meanEvalTrain'] = meanEvalTrain
    varsPickle['stdDevEvalTrain'] = stdDevEvalTrain
    varsPickle['meanLossValid'] = meanLossValid
    varsPickle['stdDevLossValid'] = stdDevLossValid
    varsPickle['meanEvalValid'] = meanEvalValid
    varsPickle['stdDevEvalValid'] = stdDevEvalValid
    with open(os.path.join(saveDirFigs,'figVars.pkl'), 'wb') as figVarsFile:
        pickle.dump(varsPickle, figVarsFile)
    #   Matlab, second:
    varsMatlab = {}
    varsMatlab['nEpochs'] = nEpochs
    varsMatlab['nBatches'] = nBatches
    for thisModel in modelList:
        varsMatlab['meanLossTrain' + thisModel] = meanLossTrain[thisModel]
        varsMatlab['stdDevLossTrain' + thisModel] = stdDevLossTrain[thisModel]
        varsMatlab['meanEvalTrain' + thisModel] = meanEvalTrain[thisModel]
        varsMatlab['stdDevEvalTrain' + thisModel] = stdDevEvalTrain[thisModel]
        varsMatlab['meanLossValid' + thisModel] = meanLossValid[thisModel]
        varsMatlab['stdDevLossValid' + thisModel] = stdDevLossValid[thisModel]
        varsMatlab['meanEvalValid' + thisModel] = meanEvalValid[thisModel]
        varsMatlab['stdDevEvalValid' + thisModel] = stdDevEvalValid[thisModel]
    savemat(os.path.join(saveDirFigs, 'figVars.mat'), varsMatlab)

    ########
    # PLOT #
    ########

    # Compute the x-axis
    xTrain = np.arange(0, nEpochs * nBatches, xAxisMultiplierTrain)
    xValid = np.arange(0, nEpochs * nBatches, \
                          validationInterval*xAxisMultiplierValid)

    # If we do not want to plot all the elements (to avoid overcrowded plots)
    # we need to recompute the x axis and take those elements corresponding
    # to the training steps we want to plot
    if xAxisMultiplierTrain > 1:
        # Actual selected samples
        selectSamplesTrain = xTrain
        # Go and fetch tem
        for thisModel in modelList:
            meanLossTrain[thisModel] = meanLossTrain[thisModel]\
                                                    [selectSamplesTrain]
            stdDevLossTrain[thisModel] = stdDevLossTrain[thisModel]\
                                                        [selectSamplesTrain]
            meanEvalTrain[thisModel] = meanEvalTrain[thisModel]\
                                                    [selectSamplesTrain]
            stdDevEvalTrain[thisModel] = stdDevEvalTrain[thisModel]\
                                                        [selectSamplesTrain]
    # And same for the validation, if necessary.
    if xAxisMultiplierValid > 1:
        selectSamplesValid = np.arange(0, len(meanLossValid[thisModel]), \
                                       xAxisMultiplierValid)
        for thisModel in modelList:
            meanLossValid[thisModel] = meanLossValid[thisModel]\
                                                    [selectSamplesValid]
            stdDevLossValid[thisModel] = stdDevLossValid[thisModel]\
                                                        [selectSamplesValid]
            meanEvalValid[thisModel] = meanEvalValid[thisModel]\
                                                    [selectSamplesValid]
            stdDevEvalValid[thisModel] = stdDevEvalValid[thisModel]\
                                                        [selectSamplesValid]

    #\\\ LOSS (Training and validation) for EACH MODEL
    for key in meanLossTrain.keys():
        lossFig = plt.figure(figsize=(1.61*5, 1*5))
        plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
                     color = '#01256E', linewidth = 2,
                     marker = 'o', markersize = 3)
        plt.errorbar(xValid, meanLossValid[key], yerr = stdDevLossValid[key],
                     color = '#95001A', linewidth = 2,
                     marker = 'o', markersize = 3)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        lossFig.savefig(os.path.join(saveDirFigs,'loss%s.pdf' % key),
                        bbox_inches = 'tight')

    #\\\ ACCURACY (Training and validation) for EACH MODEL
    for key in meanEvalTrain.keys():
        accFig = plt.figure(figsize=(1.61*5, 1*5))
        plt.errorbar(xTrain, meanEvalTrain[key], yerr = stdDevEvalTrain[key],
                     color = '#01256E', linewidth = 2,
                     marker = 'o', markersize = 3)
        plt.errorbar(xValid, meanEvalValid[key], yerr = stdDevEvalValid[key],
                     color = '#95001A', linewidth = 2,
                     marker = 'o', markersize = 3)
        plt.ylabel(r'Accuracy')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        accFig.savefig(os.path.join(saveDirFigs,'eval%s.pdf' % key),
                        bbox_inches = 'tight')

    # LOSS (training) for ALL MODELS
    allLossTrain = plt.figure(figsize=(1.61*5, 1*5))
    for key in meanLossTrain.keys():
        plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
                     linewidth = 2, marker = 'o', markersize = 3)
    plt.ylabel(r'Loss')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanLossTrain.keys()))
    allLossTrain.savefig(os.path.join(saveDirFigs,'allLossTrain.pdf'),
                    bbox_inches = 'tight')

    # ACCURACY (validation) for ALL MODELS
    allEvalValid = plt.figure(figsize=(1.61*5, 1*5))
    for key in meanEvalValid.keys():
        plt.errorbar(xValid, meanEvalValid[key], yerr = stdDevEvalValid[key],
                     linewidth = 2, marker = 'o', markersize = 3)
    plt.ylabel(r'Accuracy')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanEvalValid.keys()))
    allEvalValid.savefig(os.path.join(saveDirFigs,'allEvalValid.pdf'),
                    bbox_inches = 'tight')