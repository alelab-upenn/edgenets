# 2018/12/05~2019/03/04.
# Fernando Gama, fgama@seas.upenn.edu
"""
architectures.py Architectures module

Definition of GNN architectures.

SelectionGNN: implements the selection GNN architecture
SpectralGNN: implements the selection GNN architecture using spectral filters
NodeVariantGNN: implements the selection GNN architecture with node-variant 
    graph filters
EdgeVariantGNN: implements the selection GNN architecture with edge-variant
    graph filters
HybridEdgeVariantGNN: implements the selection GNN architecture with hybrid
    edge-variant graph filters
"""

import numpy as np
import torch
import torch.nn as nn

import Utils.graphML as gml
import Utils.graphTools as graphTools

zeroTolerance = 1e-9 # Values below this number are considered zero.

class SelectionGNN(nn.Module):
    """
    SelectionGNN: implement the selection GNN architecture

    Initialization:

        SelectionGNN(dimNodeSignals, nFilterTaps, bias, # Graph Filtering
                     nonlinearity, # Nonlinearity
                     nSelectedNodes, poolingFunction, poolingSize, # Pooling
                     dimLayersMLP, # MLP in the end
                     GSO) # Structure

        Input:
            dimNodeSignals (list of int): dimension of the signals at each layer
            nFilterTaps (list of int): number of filter taps on each layer
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nFilterTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nFilterTaps) = L.
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
            GSO (np.array): graph shift operator of choice.

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        SelectionGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nFilterTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nFilterTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Filter taps
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        # Store the rest of the variables
        self.bias = bias # Boolean
        self.S = torch.tensor(GSO)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        # Now compute the neighborhoods which we need for the pooling operation
        self.neighborhood = [] # This one will have length L. For each layer
        # we need a matrix of size N[l+1] (the neighbors we actually need) times
        # the maximum size of the neighborhood:
        #   nOutputNodes x maxNeighborhoodSize
        # The padding has to be done with the neighbor itself.
        # Remember that the picking is always of the top nodes.
        for l in range(self.L):
            # And, in this case, I should not use the powers, so the function
            # has to be something like
            thisNeighborhood = graphTools.computeNeighborhood(
                            np.array(GSO), self.alpha[l],
                            self.N[l+1], self.N[l], 'matrix')
            self.neighborhood.append(torch.tensor(thisNeighborhood))
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            gfl.append(gml.GraphFilter(self.F[l], self.F[l+1], self.K[l],
                                              self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            gfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            gfl.append(self.sigma())
            #\\ Pooling
            gfl.append(self.rho(self.N[l], self.N[l+1]))
            # Same as before, this is 3*l+2
            gfl[3*l+2].addNeighborhood(self.neighborhood[l])
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y = self.GFL(x)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.GFL[3*l].addGSO(self.S)
            self.neighborhood[l] = self.neighborhood[l].to(device)
            self.GFL[3*l+2].addNeighborhood(self.neighborhood[l])
            
class SpectralGNN(nn.Module):
    """
    SpectralGNN: implement the selection GNN architecture using spectral filters

    Initialization:

        SpectralGNN(dimNodeSignals, nCoeff, bias, # Graph Filtering
                    nonlinearity, # Nonlinearity
                    nSelectedNodes, poolingFunction, poolingSize, # Pooling
                    dimLayersMLP, # MLP in the end
                    GSO) # Structure

        Input:
            dimNodeSignals (list of int): dimension of the signals at each layer
            nCoeff (list of int): number of coefficients on each layer
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nCoeff[l] is the number of coefficients for the
                filters implemented at layer l+1, thus len(nCoeff) = L.
            >> Obs.: If nCoeff[l] is less than the size of the graph, the
                remaining coefficients are interpolated by means of a cubic
                spline.
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
            GSO (np.array): graph shift operator of choice.

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        SpectralGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nCoeff, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nCoeff) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nCoeff)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nCoeff)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nCoeff) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.M = nCoeff # Filter taps
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        self.S = torch.tensor(GSO)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        # Now compute the neighborhoods which we need for the pooling operation
        self.neighborhood = [] # This one will have length L. For each layer
        # we need a matrix of size N[l+1] (the neighbors we actually need) times
        # the maximum size of the neighborhood:
        #   nOutputNodes x maxNeighborhoodSize
        # The padding has to be done with the neighbor itself.
        # Remember that the picking is always of the top nodes.
        for l in range(self.L):
            # And, in this case, I should not use the powers, so the function
            # has to be something like
            thisNeighborhood = graphTools.computeNeighborhood(
                            np.array(GSO), self.alpha[l],
                            self.N[l+1], self.N[l], 'matrix')
            self.neighborhood.append(torch.tensor(thisNeighborhood))
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        sgfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            sgfl.append(gml.SpectralGF(self.F[l], self.F[l+1], self.M[l],
                                              self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            sgfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            sgfl.append(self.sigma())
            #\\ Pooling
            sgfl.append(self.rho(self.N[l], self.N[l+1]))
            # Same as before, this is 3*l+2
            sgfl[3*l+2].addNeighborhood(self.neighborhood[l])
        # And now feed them into the sequential
        self.SGFL = nn.Sequential(*sgfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y = self.SGFL(x)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.SGFL[3*l].addGSO(self.S)
            self.neighborhood[l] = self.neighborhood[l].to(device)
            self.SGFL[3*l+2].addNeighborhood(self.neighborhood[l])
            
class NodeVariantGNN(nn.Module):
    """
    NodeVariantGNN: implement the selection GNN architecture using node variant
        graph filters

    Initialization:

        NodeVariantGNN(dimNodeSignals, nShiftTaps, nNodeTaps, bias, # Filtering
                       nonlinearity, # Nonlinearity
                       nSelectedNodes, poolingFunction, poolingSize, # Pooling
                       dimLayersMLP, # MLP in the end
                       GSO) # Structure

        Input:
            dimNodeSignals (list of int): dimension of the signals at each layer
            nShiftTaps (list of int): number of shift taps on each layer
            nNodeTaps (list of int): number of node taps on each layer
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nShiftTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nShiftTaps) = L.
            >> Obs.: The length of the nShiftTaps and nNodeTaps has to be the
                same, and every element of one list is associated with the
                corresponding one on the other list to create the appropriate
                NVGF filter at each layer.
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
            GSO (np.array): graph shift operator of choice.

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        NodeVariantGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nShiftTaps, nNodeTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than the number of
        # filter taps (because of the input number of features)
        assert len(dimNodeSignals) == len(nShiftTaps) + 1
        # The length of the shift taps list should be equal to the length of the
        # node taps list
        assert len(nShiftTaps) == len(nNodeTaps)
        # nSelectedNodes should be a list of size nShiftTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nShiftTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nShiftTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nShiftTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nShiftTaps # Filter Shift taps
        self.M = nNodeTaps # Filter node taps
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        self.S = torch.tensor(GSO)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        # Now compute the neighborhoods which we need for the pooling operation
        self.neighborhood = [] # This one will have length L. For each layer
        # we need a matrix of size N[l+1] (the neighbors we actually need) times
        # the maximum size of the neighborhood:
        #   nOutputNodes x maxNeighborhoodSize
        # The padding has to be done with the neighbor itself.
        # Remember that the picking is always of the top nodes.
        for l in range(self.L):
            # And, in this case, I should not use the powers, so the function
            # has to be something like
            thisNeighborhood = graphTools.computeNeighborhood(
                            np.array(GSO), self.alpha[l],
                            self.N[l+1], self.N[l], 'matrix')
            self.neighborhood.append(torch.tensor(thisNeighborhood))
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        nvgfl = [] # Node Variant GF Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            nvgfl.append(gml.NodeVariantGF(self.F[l], self.F[l+1],
                                           self.K[l], self.M[l],
                                           self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            nvgfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            nvgfl.append(self.sigma())
            #\\ Pooling
            nvgfl.append(self.rho(self.N[l], self.N[l+1]))
            # Same as before, this is 3*l+2
            nvgfl[3*l+2].addNeighborhood(self.neighborhood[l])
        # And now feed them into the sequential
        self.NVGFL = nn.Sequential(*nvgfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y = self.NVGFL(x)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.NVGFL[3*l].addGSO(self.S)
            self.neighborhood[l] = self.neighborhood[l].to(device)
            self.NVGFL[3*l+2].addNeighborhood(self.neighborhood[l])
            
class EdgeVariantGNN(nn.Module):
    """
    EdgeVariantGNN: implement the selection GNN architecture using edge variant
        graph filters (through masking, not placement)

    Initialization:

        EdgeVariantGNN(dimNodeSignals, nShiftTaps, bias, # Graph Filtering
                       nonlinearity, # Nonlinearity
                       nSelectedNodes, poolingFunction, poolingSize, # Pooling
                       dimLayersMLP, # MLP in the end
                       GSO) # Structure

        Input:
            dimNodeSignals (list of int): dimension of the signals at each layer
            nShiftTaps (list of int): number of shift taps on each layer
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nShiftTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nShiftTaps) = L.
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
            GSO (np.array): graph shift operator of choice.

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        EdgeVariantGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nShiftTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than the number of
        # filter taps (because of the input number of features)
        assert len(dimNodeSignals) == len(nShiftTaps) + 1
        # nSelectedNodes should be a list of size nShiftTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nShiftTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nShiftTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nShiftTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nShiftTaps # Filter Shift taps
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        self.S = torch.tensor(GSO)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        # Now compute the neighborhoods which we need for the pooling operation
        self.neighborhood = [] # This one will have length L. For each layer
        # we need a matrix of size N[l+1] (the neighbors we actually need) times
        # the maximum size of the neighborhood:
        #   nOutputNodes x maxNeighborhoodSize
        # The padding has to be done with the neighbor itself.
        # Remember that the picking is always of the top nodes.
        for l in range(self.L):
            # And, in this case, I should not use the powers, so the function
            # has to be something like
            thisNeighborhood = graphTools.computeNeighborhood(
                            np.array(GSO), self.alpha[l],
                            self.N[l+1], self.N[l], 'matrix')
            self.neighborhood.append(torch.tensor(thisNeighborhood))
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        evgfl = [] # Node Variant GF Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            evgfl.append(gml.EdgeVariantGF(self.F[l], self.F[l+1],
                                           self.K[l], self.N[0],
                                           self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            evgfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            evgfl.append(self.sigma())
            #\\ Pooling
            evgfl.append(self.rho(self.N[l], self.N[l+1]))
            # Same as before, this is 3*l+2
            evgfl[3*l+2].addNeighborhood(self.neighborhood[l])
        # And now feed them into the sequential
        self.EVGFL = nn.Sequential(*evgfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y = self.EVGFL(x)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.EVGFL[3*l].addGSO(self.S)
            self.neighborhood[l] = self.neighborhood[l].to(device)
            self.EVGFL[3*l+2].addNeighborhood(self.neighborhood[l])
            
class HybridEdgeVariantGNN(nn.Module):
    """
    HybridEdgeVariantGNN: implement the selection GNN architecture using hybrid
        edge variant graph filters (through masking, not placement)

    Initialization:

        HybridEdgeVariantGNN(dimNodeSignals, nShiftTaps, nFilterNodes, bias, 
                             nonlinearity, # Nonlinearity
                             nSelectedNodes, poolingFunction, poolingSize,
                             dimLayersMLP, # MLP in the end
                             GSO) # Structure

        Input:
            dimNodeSignals (list of int): dimension of the signals at each layer
            nShiftTaps (list of int): number of shift taps on each layer
            nFilterNodes (list of int): number of nodes selected for the EV part
                of the hybrid EV filtering (recall that the first ones in the 
                given permutation of S are the nodes selected)
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nShiftTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nShiftTaps) = L.
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
            GSO (np.array): graph shift operator of choice.

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        HybridEdgeVariantGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nShiftTaps, nFilterNodes, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than the number of
        # filter taps (because of the input number of features)
        assert len(dimNodeSignals) == len(nShiftTaps) + 1
        # Filter nodes is a list of int with the number of nodes to select for
        # the EV part at each layer; it should have the same length as the 
        # number of filter taps
        assert len(nFilterNodes) == len(nShiftTaps)
        # nSelectedNodes should be a list of size nShiftTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nShiftTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nShiftTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nShiftTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nShiftTaps # Filter Shift taps
        self.M = nFilterNodes
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        self.S = torch.tensor(GSO)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        # Now compute the neighborhoods which we need for the pooling operation
        self.neighborhood = [] # This one will have length L. For each layer
        # we need a matrix of size N[l+1] (the neighbors we actually need) times
        # the maximum size of the neighborhood:
        #   nOutputNodes x maxNeighborhoodSize
        # The padding has to be done with the neighbor itself.
        # Remember that the picking is always of the top nodes.
        for l in range(self.L):
            # And, in this case, I should not use the powers, so the function
            # has to be something like
            thisNeighborhood = graphTools.computeNeighborhood(
                            np.array(GSO), self.alpha[l],
                            self.N[l+1], self.N[l], 'matrix')
            self.neighborhood.append(torch.tensor(thisNeighborhood))
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        hevgfl = [] # Node Variant GF Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            hevgfl.append(gml.HybridEdgeVariantGF(self.F[l], self.F[l+1],
                                           self.K[l], self.M[l], self.N[0],
                                           self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            hevgfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            hevgfl.append(self.sigma())
            #\\ Pooling
            hevgfl.append(self.rho(self.N[l], self.N[l+1]))
            # Same as before, this is 3*l+2
            hevgfl[3*l+2].addNeighborhood(self.neighborhood[l])
        # And now feed them into the sequential
        self.HEVGFL = nn.Sequential(*hevgfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y = self.HEVGFL(x)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.HEVGFL[3*l].addGSO(self.S)
            self.neighborhood[l] = self.neighborhood[l].to(device)
            self.HEVGFL[3*l+2].addNeighborhood(self.neighborhood[l])