# 2018/12/4~2019/03/04.
# Fernando Gama, fgama@seas.upenn.edu
"""
dataTools.py Data management module

Several tools to manage data

SourceLocalization (class): creates the datasets for a source localization problem
"""

import numpy as np
import torch
import hdf5storage # This is required to import old Matlab(R) files.

import Utils.graphTools as graph

class _dataForClassification:
    # Internal supraclass from which data classes inherit.
    # There are certian methods that all Data classes must have:
    #   getSamples(), evluate(), to() and astype().
    # To avoid coding this methods over and over again, we create a class from
    # which the data can inherit this basic methods.
    # Note that this is called "ForClassification" since the evaluate method
    # is only for classification evaluations.
    # However, it is true that getSamples() might be useful beyond the 
    # classification problem, so we might, eventually, consider a different
    # internal class.
    def __init__(self):
        # Minimal set of attributes that all data classes should have
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = None
        self.samples['train']['labels'] = None
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = None
        self.samples['valid']['labels'] = None
        self.samples['test'] = {}
        self.samples['test']['signals'] = None
        self.samples['test']['labels'] = None
        
    def getSamples(self, samplesType, *args):
        # type: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['labels']
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # The reshape is to avoid squeezing if only one sample is
                # requested
                x = x[selectedIndices,:].reshape([args[0], x.shape[1]])
                y = y[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                x = x[args[0], :]
                # If only one element is selected, avoid squeezing. Given that
                # the element can be a list (which has property len) or an
                # np.array (which doesn't have len, but shape), then we can
                # only avoid squeezing if we check that it has been sequeezed
                # (or not)
                if len(x.shape) == 1:
                    x = x.reshape([1, x.shape[0]])
                # And assign the labels
                y = y[args[0]]

        return x, y

    def astype(self, dataType):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        if repr(dataType).find('torch') == -1:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                       = dataType(self.samples[key][secondKey])
        else:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                    = torch.tensor(self.samples[key][secondKey]).type(dataType)

        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        # This can only be done if they are torch tensors
        if repr(self.dataType).find('torch') >= 0:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                      = self.samples[key][secondKey].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device

    def evaluate(self, yHat, y, tol = 1e-9):
        """
        Return the accuracy (ratio of yHat = y)
        """
        N = len(y)
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
            yHat = torch.argmax(yHat, dim = 1).type(self.dataType)
            #   And compute the error
            totalErrors = torch.sum(torch.abs(yHat - y) > tol)
            accuracy = 1 - totalErrors.type(self.dataType)/N
        else:
            yHat = np.array(yHat)
            y = np.array(y)
            #   We compute the target label (hardmax)
            yHat = np.argmax(yHat, axis = 1).astype(y.dtype)
            #   And compute the error
            totalErrors = np.sum(np.abs(yHat - y) > tol)
            accuracy = 1 - totalErrors.astype(self.dataType)/N
        #   And from that, compute the accuracy
        return accuracy
        

class SourceLocalization(_dataForClassification):
    """
    SourceLocalization: Creates the dataset for a source localization problem

    Initialization:

    Input:
        G (class): Graph on which to diffuse the process, needs an attribute
            .N with the number of nodes (int) and attribute .W with the
            adjacency matrix (np.array)
        nTrain (int): number of training samples
        nValid (int): number of validation samples
        nTest (int): number of testing samples
        sourceNodes (list of int): list of indices of nodes to be used as
            sources of the diffusion process
        tMax (int): maximum diffusion time, if None, the maximum diffusion time
            is the size of the graph (default: None)
        dataType (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved.

    Methods:

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """

    def __init__(self, G, nTrain, nValid, nTest, sourceNodes, tMax = None,
                 dataType = np.float64, device = 'cpu'):
        # Initialize parent
        super().__init__()
        # store attributes
        self.dataType = dataType
        self.device = device
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        # If no tMax is specified, set it the maximum possible.
        if tMax == None:
            tMax = G.N
        #\\\ Generate the samples
        # Get the largest eigenvalue of the weighted adjacency matrix
        EW, VW = graph.computeGFT(G.W, order = 'totalVariation')
        eMax = np.max(EW)
        # Normalize the matrix so that it doesn't explode
        Wnorm = G.W / eMax
        # total number of samples
        nTotal = nTrain + nValid + nTest
        # sample source nodes
        sampledSources = np.random.choice(sourceNodes, size = nTotal)
        # sample diffusion times
        sampledTimes = np.random.choice(tMax, size = nTotal)
        # Since the signals are generated as W^t * delta, this reduces to the
        # selection of a column of W^t (the column corresponding to the source
        # node). Therefore, we generate an array of size tMax x N x N with all
        # the powers of the matrix, and then we just simply select the
        # corresponding column for the corresponding time
        lastWt = np.eye(G.N, G.N)
        Wt = lastWt.reshape([1, G.N, G.N])
        for t in range(1,tMax):
            lastWt = lastWt @ Wnorm
            Wt = np.concatenate((Wt, lastWt.reshape([1, G.N, G.N])), axis = 0)
        x = Wt[sampledTimes, :, sampledSources]
        # Now, we have the signals and the labels
        signals = x # nTotal x N (CS notation)
        # Finally, we have to match the source nodes to the corresponding labels
        # which start at 0 and increase in integers.
        nodesToLabels = {}
        for it in range(len(sourceNodes)):
            nodesToLabels[sourceNodes[it]] = it
        labels = [nodesToLabels[x] for x in sampledSources] # nTotal
        # Split and save them
        self.samples['train']['signals'] = signals[0:nTrain, :]
        self.samples['train']['labels'] = labels[0:nTrain]
        self.samples['valid']['signals'] = signals[nTrain:nTrain+nValid, :]
        self.samples['valid']['labels'] = labels[nTrain:nTrain+nValid]
        self.samples['test']['signals'] = signals[nTrain+nValid:nTotal, :]
        self.samples['test']['labels'] = labels[nTrain+nValid:nTotal]
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
    
class Authorship(_dataForClassification):
    """
    Authorship: Loads the dataset of 19th century writers for the authorship
        attribution problem

    Initialization:

    Input:
        authorName (string): which is the selected author to attribute plays to
        ratioTrain (float): ratio of the total texts to be part of the training
            set
        ratioValid (float): ratio of the train texts to be part of the
            validation set
        dataPath (string): path to where the authorship data is located
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)

    Methods:
        
    .loadData(dataPath): load the data found in dataPath and store it in 
        attributes .authorData and .functionWords

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """
    
    def __init__(self, authorName, ratioTrain, ratioValid, dataPath,
                 dataType = np.float64, device = 'cpu'):
        # Initialize parent
        super().__init__()
        # Store
        self.dataType = dataType
        self.device = device
        # Load data
        self.loadData(dataPath)
        # Check that the authorName is a valid name
        assert authorName in self.authorData.keys()
        # Get the selected author's data
        thisAuthorData = self.authorData[authorName].copy()
        nExcerpts = thisAuthorData['wordFreq'].shape[0] # Number of excerpts
            # by the selected author
        nTrainAuthor = round(ratioTrain * nExcerpts)
        nValidAuthor = round(ratioValid * nTrainAuthor)
        nTestAuthor = nExcerpts - nTrainAuthor
        nTrainAuthor = nTrainAuthor - nValidAuthor
        # Now, we know how many training, validation and testing samples from
        # the required author. But we will also include an equal amount of
        # other authors, therefore
        self.nTrain = round(2 * nTrainAuthor)
        self.nValid = round(2 * nValidAuthor)
        self.nTest = round(2 * nTestAuthor)
        
        # Now, let's get the corresponding signals for the author
        xAuthor = thisAuthorData['wordFreq']
        # Get a random permutation of these works, and split them accordingly
        randPerm = np.random.permutation(nExcerpts)
        # Save the indices corresponding to each split
        randPermTrain = randPerm[0:nTrainAuthor]
        randPermValid = randPerm[nTrainAuthor:nTrainAuthor+nValidAuthor]
        randPermTest = randPerm[nTrainAuthor+nValidAuthor:nExcerpts]
        xAuthorTrain = xAuthor[randPermTrain, :]
        xAuthorValid = xAuthor[randPermValid, :]
        xAuthorTest = xAuthor[randPermTest, :]
        # And we will store this split
        self.selectedAuthor = {}
        # Copy all data
        self.selectedAuthor['all'] = thisAuthorData.copy()
        # Copy word frequencies
        self.selectedAuthor['train'] = {}
        self.selectedAuthor['train']['wordFreq'] = xAuthorTrain.copy()
        self.selectedAuthor['valid'] = {}
        self.selectedAuthor['valid']['wordFreq'] = xAuthorValid.copy()
        self.selectedAuthor['test'] = {}
        self.selectedAuthor['test']['wordFreq'] = xAuthorTest.copy()
        # Copy WANs
        self.selectedAuthor['train']['WAN'] = \
                              thisAuthorData['WAN'][randPermTrain, :, :].copy()
        self.selectedAuthor['valid']['WAN'] = \
                              thisAuthorData['WAN'][randPermValid, :, :].copy()
        self.selectedAuthor['test']['WAN'] = \
                               thisAuthorData['WAN'][randPermTest, :, :].copy()
        # Now we need to get an equal amount of works from the rest of the
        # authors.
        xRest = np.empty([0, xAuthorTrain.shape[1]]) # Create an empty matrix
        # to store all the works by the rest of the authors.
        # Now go author by author gathering all works
        for key in self.authorData.keys():
            # Only for authors that are not the selected author
            if key is not authorName:
                thisAuthorTexts = self.authorData[key]['wordFreq']
                xRest = np.concatenate((xRest, thisAuthorTexts), axis = 0)
        # After obtaining all works, xRest is of shape nRestOfData x nWords
        # We now need to select at random from this other data, but only up
        # to nExcerpts. Therefore, we will randperm all the indices, but keep
        # only the first nExcerpts indices.
        randPerm = np.random.permutation(xRest.shape[0])
        randPerm = randPerm[0:nExcerpts] # nExcerpts x nWords
        # And now we should just get the appropriate number of texts from these
        # other authors.
        # Compute how many samples for each case
        nTrainRest = self.nTrain - nTrainAuthor
        nValidRest = self.nValid - nValidAuthor
        nTestRest = self.nTest - nTestAuthor
        # And obtain those
        xRestTrain = xRest[randPerm[0:nTrainRest], :]
        xRestValid = xRest[randPerm[nTrainRest:nTrainRest + nValidRest], :]
        xRestTest = xRest[randPerm[nTrainRest+nValidRest:nExcerpts], :]
        # Now construct the signals and labels. Signals is just the 
        # concatenation of each of these excerpts. Labels is just a bunch of
        # 1s followed by a bunch of 0s
        # Obs.: The fact that the dataset is ordered now, it doesn't matter,
        # since it will be shuffled at each epoch.
        xTrain = np.concatenate((xAuthorTrain, xRestTrain), axis = 0)
        labelsTrain = np.concatenate((np.ones(nTrainAuthor),
                                      np.zeros(nTrainRest)), axis = 0)
        xValid = np.concatenate((xAuthorValid, xRestValid), axis = 0)
        labelsValid = np.concatenate((np.ones(nValidAuthor),
                                      np.zeros(nValidRest)), axis = 0)
        xTest = np.concatenate((xAuthorTest, xRestTest), axis = 0)
        labelsTest = np.concatenate((np.ones(nTestAuthor),
                                     np.zeros(nTestRest)), axis = 0)
        # And assign them to the required attribute samples
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = xTrain
        self.samples['train']['labels'] = labelsTrain
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = xValid
        self.samples['valid']['labels'] = labelsValid
        self.samples['test'] = {}
        self.samples['test']['signals'] = xTest
        self.samples['test']['labels'] = labelsTest
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
        
    
    def loadData(self, dataPath):
        # TODO: Analyze if it's worth it to create a .pkl and load that 
        # directly once the data has been appropriately parsed. It's just
        # that loading with hdf5storage takes a couple of second that
        # could be saved if the .pkl file is faster.
        rawData = hdf5storage.loadmat(dataPath)
        # rawData is a dictionary with four keys:
        #   'all_authors': contains the author list
        #   'all_freqs': contains the word frequency count for each excerpt
        #   'all_wans': contains the WANS for each excerpt
        #   'function_words': a list of the functional words
        # The issue is that hdf5storage, while necessary to load old 
        # Matlab(R) files, gives the data in a weird format, that we need
        # to adapt and convert.
        # The data will be structured as follows. We will have an
        # authorData dictionary of dictionaries: the first key will be the
        # author name, the second key will be either freqs or wans to
        # access either one or another.
        # We will also clean up and save the functional word list, although
        # we do not need to use it.
        authorData = {} # Create dictionary
        for it in range(len(rawData['all_authors'])):
            thisAuthor = str(rawData['all_authors'][it][0][0][0])
            # Each element in rawData['all_authors'] is nested in a couple
            # of lists, so that's why we need the three indices [0][0][0] 
            # to reach the string with the actual author name.
            # Get the word frequency
            thisWordFreq = rawData['all_freqs'][0][it] # 1 x nWords x nData
            # Again, the [0] is due to the structure of the data
            # Let us get rid of that extra 1, and then transpose this to be
            # stored as nData x nWords (since nWords is the dimension of 
            # the number of nodes the network will have; CS notation)
            thisWordFreq = thisWordFreq.squeeze(0).T # nData x nWords
            # Finally, get the WANs
            thisWAN = rawData['all_wans'][0][it] # nWords x nWords x nData
            thisWAN = thisWAN.transpose(2, 0, 1) # nData x nWords x nWords
            # Obs.: thisWAN is likely not symmetric, so the way this is 
            # transposed matters. In this case, since thisWAN was intended
            # to be a tensor in matlab (where the last index is the 
            # collection of matrices), we just throw that last dimension to
            # the front (since numpy consider the first index as the 
            # collection index).
            # Now we can create the dictionary and save the corresopnding
            # data.
            authorData[thisAuthor] = {}
            authorData[thisAuthor]['wordFreq'] = thisWordFreq
            authorData[thisAuthor]['WAN'] = thisWAN
        # And at last, gather the list of functional words
        functionWords = [] # empty list to store the functional words
        for word in rawData['function_words']:
            functionWords.append(str(word[0][0][0]))
        # Store all the data recently collected
        self.authorData = authorData
        self.functionWords = functionWords