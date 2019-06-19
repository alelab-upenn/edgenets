# 2018/12/03~2019/03/04.
# Fernando Gama, fgama@seas.upenn.edu
"""
graphTools.py Tools for handling graphs

Functions:

adjacencyToLaplacian: transform an adjacency matrix into a Laplacian matrix
normalizeAdjacency: compute the normalized adjacency
normalizeLaplacian: compute the normalized Laplacian
computeGFT: Computes the eigenbasis of a GSO
matrixPowers: computes the matrix powers
computeNonzeroRows: compute nonzero elements across rows
computeNeighborhood: compute the neighborhood of a graph
isConnected: determines if a graph is connected
createGraph: creates an adjacency marix
permIdentity: identity permutation
permDegree: order nodes by degree
permSpectralProxies: order nodes by spectral proxies score
permEDS: order nodes by EDS score
splineBasis: Returns the B-spline basis (taken from github.com/mdeff)

Classes:

Graph: class containing a graph
"""

import numpy as np
import scipy.sparse

zeroTolerance = 1e-9 # Values below this number are considered zero.

# If adjacency matrices are not symmetric these functions might not work as
# desired: the degree will be the in-degree to each node, and the Laplacian
# is not defined for directed graphs. Same caution is advised when using
# graphs with self-loops.

def adjacencyToLaplacian(W):
    """
    adjacencyToLaplacian: Computes the Laplacian from an Adjacency matrix

    Input:

        W (np.array): adjacency matrix

    Output:

        L (np.array): Laplacian matrix
    """
    # Check that the matrix is square
    assert W.shape[0] == W.shape[1]
    # Compute the degree vector
    d = np.sum(W, axis = 1)
    # And build the degree matrix
    D = np.diag(d)
    # Return the Laplacian
    return D - W

def normalizeAdjacency(W):
    """
    NormalizeAdjacency: Computes the degree-normalized adjacency matrix

    Input:

        W (np.array): adjacency matrix

    Output:

        A (np.array): degree-normalized adjacency matrix
    """
    # Check that the matrix is square
    assert W.shape[0] == W.shape[1]
    # Compute the degree vector
    d = np.sum(W, axis = 1)
    # Invert the square root of the degree
    d = 1/np.sqrt(d)
    # And build the square root inverse degree matrix
    D = np.diag(d)
    # Return the Normalized Adjacency
    return D @ W @ D

def normalizeLaplacian(L):
    """
    NormalizeLaplacian: Computes the degree-normalized Laplacian matrix

    Input:

        L (np.array): Laplacian matrix

    Output:

        normL (np.array): degree-normalized Laplacian matrix
    """
    # Check that the matrix is square
    assert L.shape[0] == L.shape[2]
    # Compute the degree vector (diagonal elements of L)
    d = np.diag(L)
    # Invert the square root of the degree
    d = 1/np.sqrt(d)
    # And build the square root inverse degree matrix
    D = np.diag(d)
    # Return the Normalized Laplacian
    return D @ L @ D

def computeGFT(S, order = 'no'):
    """
    computeGFT: Computes the frequency basis (eigenvectors) and frequency
        coefficients (eigenvalues) of a given GSO

    Input:

        S (np.array): graph shift operator matrix
        order (string): 'no', 'increasing', 'totalVariation' chosen order of
            frequency coefficients (default: 'no')

    Output:

        E (np.array): diagonal matrix with the frequency coefficients
            (eigenvalues) in the diagonal
        V (np.array): matrix with frequency basis (eigenvectors)
    """
    # Check the correct order input
    assert order == 'totalVariation' or order == 'no' or order == 'increasing'
    # Check the matrix is square
    assert S.shape[0] == S.shape[1]
    # Check if it is symmetric
    symmetric = np.allclose(S, S.T, atol = zeroTolerance)
    # Then, compute eigenvalues and eigenvectors
    if symmetric:
        e, V = np.linalg.eigh(S)
    else:
        e, V = np.linalg.eig(S)
    # Sort the eigenvalues by the desired error:
    if order == 'totalVariation':
        eMax = np.max(e)
        sortIndex = np.argsort(np.abs(e - eMax))
    if order == 'increasing':
        sortIndex = np.argsort(np.abs(e))
    else:
        sortIndex = np.arange(0, S.shape[0])
    e = e[sortIndex]
    V = V[:, sortIndex]
    E = np.diag(e)
    return E, V

def matrixPowers(S,K):
    """
    matrixPowers(A, K) Computes the matrix powers A^k for k = 0, ..., K-1

    Inputs:
        A: either a single N x N matrix or a collection E x N x N of E matrices.
        K: integer, maximum power to be computed (up to K-1)

    Outputs:
        AK: either a collection of K matrices K x N x N (if the input was a
            single matrix) or a collection E x K x N x N (if the input was a
            collection of E matrices).

    Observation: Operates on torch.tensor variable A. Returns torch.tesnor
    variables AK.
    """
    # S can be either a single GSO (N x N) or a collection of GSOs (E x N x N)
    if len(S.shape) == 2:
        N = S.shape[0]
        assert S.shape[1] == N
        E = 1
        S = S.reshape(1, N, N)
        scalarWeights = True
    elif len(S.shape) == 3:
        E = S.shape[0]
        N = S.shape[1]
        assert S.shape[2] == N
        scalarWeights = False

    # Now, let's build the powers of S:
    thisSK = np.tile(np.eye(N, N).reshape(1,N,N), [E, 1, 1])
    SK = thisSK.reshape(E, 1, N, N)
    for k in range(1,K):
        thisSK = thisSK @ S
        SK = np.concatenate((SK, thisSK.reshape(E, 1, N, N)), axis = 1)
    # Take out the first dimension if it was a single GSO
    if scalarWeights:
        SK = SK.reshape(K, N, N)

    return SK

def computeNonzeroRows(S, Nl = 'all'):
    """
    computeNonzeroRows: Find the position of the nonzero elements of each
        row of a matrix

    Input:

        S (np.array): matrix
        Nl (int or 'all'): number of rows to compute the nonzero elements; if
            'all', then Nl = S.shape[0]. Rows are counted from the top.

    Output:

        nonzeroElements (list): list of size Nl where each element is an array
            of the indices of the nonzero elements of the corresponding row.
    """
    # Find the position of the nonzero elements of each row of the matrix S.
    # Nl = 'all' means for all rows, otherwise, it will be an int.
    if Nl == 'all':
        Nl = S.shape[0]
    assert Nl <= S.shape[0]
    # Save neighborhood variable
    neighborhood = []
    # For each of the selected nodes
    for n in range(Nl):
        neighborhood += [np.flatnonzero(S[n,:])]

    return neighborhood

def computeNeighborhood(S, K, N = 'all', nb = 'all', outputType = 'list'):
    """
    computeNeighborhood: compute the K-hop neighborhood of a graph

        computeNeighborhood(W, K, N = 'all', nb = 'all', outputType = 'list')

    Input:
        W (np.array): adjacency matrix
        K (int): K-hop neighborhood to compute the neighbors
        N (int or 'all'): how many nodes (from top) to compute the neighbors
            from (default: 'all').
        nb (int or 'all'): how many nodes to consider valid when computing the
            neighborhood (i.e. nodes beyhond nb are not trimmed out of the
            neighborhood; note that nodes smaller than nb that can be reached
            by nodes greater than nb, are included. default: 'all')
        outputType ('list' or 'matrix'): choose if the output is given in the
            form of a list of arrays, or a matrix with zero-padding of neighbors
            with neighborhoods smaller than the maximum neighborhood
            (default: 'list')

    Output:
        neighborhood (np.array or list): contains the indices of the neighboring
            nodes following the order established by the adjacency matrix.
    """
    # outputType is either a list (a list of np.arrays) or a matrix.
    assert outputType == 'list' or outputType == 'matrix'
    # Here, we can assume S is already sparse, in which case is a list of
    # sparse matrices, or that S is full, in which case it is a 3-D array.
    if isinstance(S, list):
        # If it is a list, it has to be a list of matrices, where the length
        # of the list has to be the number of edge weights. But we actually need
        # to sum over all edges to be sure we consider all reachable nodes on
        # at least one of the edge dimensions
        newS = 0.
        for e in len(S):
            # First check it's a matrix, and a square one
            assert len(S[e]) == 2
            assert S[e].shape[0] == S[e].shape[1]
            # For each edge, convert to sparse (in COO because we care about
            # coordinates to find the neighborhoods)
            newS += scipy.sparse.coo_matrix(
                              (np.abs(S[e]) > zeroTolerance).astype(S[e].dtype))
        S = (newS > zeroTolerance).astype(newS.dtype)
    else:
        # if S is not a list, check that it is either a E x N x N or a N x N
        # array.
        assert len(S.shape) == 2 or len(S.shape) == 3
        if len(S.shape) == 3:
            assert S.shape[1] == S.shape[2]
            # If it has an edge feature dimension, just add over that dimension.
            # We only need one non-zero value along the vector to have an edge
            # there. (Obs.: While normally assume that all weights are positive,
            # let's just add on abs() value to avoid any cancellations).
            S = np.sum(np.abs(S), axis = 0)
            S = scipy.sparse.coo_matrix((S > zeroTolerance).astype(S.dtype))
        else:
            # In this case, if it is a 2-D array, we do not need to add over the
            # edge dimension, so we just sparsify it
            assert S.shape[0] == S.shape[1]
            S = scipy.sparse.coo_matrix((S > zeroTolerance).astype(S.dtype))
    # Now, we finally have a sparse, binary matrix, with the connections.
    # Now check that K and N are correct inputs.
    # K is an int (target K-hop neighborhood)
    # N is either 'all' or an int determining how many rows
    assert K >= 0 # K = 0 is just the identity
    # Check how many nodes we want to obtain
    if N == 'all':
        N = S.shape[0]
    if nb == 'all':
        nb = S.shape[0]
    assert N >= 0 and N <= S.shape[0] # Cannot return more nodes than there are
    assert nb >= 0 and nb <= S.shape[0]

    # All nodes are in their own neighborhood, so
    allNeighbors = [ [n] for n in range(S.shape[0])]
    # Now, if K = 0, then these are all the neighborhoods we need.
    # And also keep track only about the nodes we care about
    neighbors = [ [n] for n in range(N)]
    # But if K > 0
    if K > 0:
        # Let's start with the one-hop neighborhood of all nodes (we need this)
        nonzeroS = list(S.nonzero())
        # This is a tuple with two arrays, the first one containing the row
        # index of the nonzero elements, and the second one containing the
        # column index of the nonzero elements.
        # Now, we want the one-hop neighborhood of all nodes (and all nodes have
        # a one-hop neighborhood, since the graphs are connected)
        for n in range(len(nonzeroS[0])):
            # The list in index 0 is the nodes, the list in index 1 is the
            # corresponding neighbor
            allNeighbors[nonzeroS[0][n]].append(nonzeroS[1][n])
        # Now that we have the one-hop neighbors, we just need to do a depth
        # first search looking for the one-hop neighborhood of each neighbor
        # and so on.
        oneHopNeighbors = allNeighbors.copy()
        # We have already visited the nodes themselves, since we already
        # gathered the one-hop neighbors.
        visitedNodes = [ [n] for n in range(N)]
        # Keep only the one-hop neighborhood of the ones we're interested in
        neighbors = [list(set(allNeighbors[n])) for n in range(N)]
        # For each hop
        for k in range(1,K):
            # For each of the nodes we care about
            for i in range(N):
                # Take each of the neighbors we already have
                for j in neighbors[i]:
                    # and if we haven't visited those neighbors yet
                    if j not in visitedNodes[i]:
                        # Just look for our neighbor's one-hop neighbors and
                        # add them to the neighborhood list
                        neighbors[i].extend(oneHopNeighbors[j])
                        # And don't forget to add the node to the visited ones
                        # (we already have its one-hope neighborhood)
                        visitedNodes[i].append(j)
                # And now that we have added all the new neighbors, we just
                # get rid of those that appear more than once
                neighbors[i] = list(set(neighbors[i]))

    # Now that all nodes have been collected, get rid of those beyond nb
    for i in range(N):
        # Get the neighborhood
        thisNeighborhood = neighbors[i].copy()
        # And get rid of the excess nodes
        neighbors[i] = [j for j in thisNeighborhood if j < nb]


    if outputType == 'matrix':
        # List containing all the neighborhood sizes
        neighborhoodSizes = [len(x) for x in neighbors]
        # Obtain max number of neighbors
        maxNeighborhoodSize = max(neighborhoodSizes)
        # then we have to check each neighborhood and find if we need to add
        # more nodes (itself) to pad it so we can build a matrix
        paddedNeighbors = []
        for n in range(N):
            paddedNeighbors += [np.concatenate(
                       (neighbors[n],
                        n * np.ones(maxNeighborhoodSize - neighborhoodSizes[n]))
                                )]
        # And now that every element in the list paddedNeighbors has the same
        # length, we can make it a matrix
        neighbors = np.array(paddedNeighbors, dtype = np.int)

    return neighbors

def isConnected(W):
    """
    isConnected: determine if a graph is connected

    Input:
        W (np.array): adjacency matrix

    Output:
        connected (bool): True if the graph is connected, False otherwise
    """
    L = adjacencyToLaplacian(W)
    E, V = computeGFT(L)
    e = np.diag(E) # only eigenvavlues
    # Check how many values are greater than zero:
    nComponents = np.sum(e < zeroTolerance) # Number of connected components
    if nComponents == 1:
        connected = True
    else:
        connected = False
    return connected

def createGraph(graphType, N, *args):
    """
    createGraph: creates a graph of a specified type

    Function under construction. Right now, it only works to create an 'SBM' and
    to fuse several adjacency matrix by using graphType 'fuseEdges'
    """
    # Check
    assert N >= 0

    if graphType == 'SBM':
        assert(len(args)) == 3
        C = args[0] # Number of communities
        assert int(C) == C # Check that the number of communities is an integer
        pii = args[1] # Intracommunity probability
        pij = args[2] # Intercommunity probability
        assert 0 <= pii <= 1 # Check that they are valid probabilities
        assert 0 <= pij <= 1
        # We create the SBM as follows: we generate random numbers between
        # 0 and 1 and then we compare them elementwise to a matrix of the
        # same size of pii and pij to set some of them to one and other to
        # zero.
        # Let's start by creating the matrix of pii and pij.
        # First, we need to know how many numbers on each community.
        nNodesC = [N//C] * C # Number of nodes per community: floor division
        c = 0 # counter for community
        while sum(nNodesC) < N: # If there are still nodes to put in communities
        # do it one for each (balanced communities)
            nNodesC[c] = nNodesC[c] + 1
            c += 1
        # So now, the list nNodesC has how many nodes are on each community.
        # We proceed to build the probability matrix.
        # We create a zero matrix
        probMatrix = np.zeros([N,N])
        # And fill ones on the block diagonals following the number of nodes.
        # For this, we need the cumulative sum of the number of nodes
        nNodesCIndex = [0] + np.cumsum(nNodesC).tolist()
        # The zero is added because it is the first index
        for c in range(C):
            probMatrix[ nNodesCIndex[c] : nNodesCIndex[c+1] , \
                        nNodesCIndex[c] : nNodesCIndex[c+1] ] = \
                np.ones([nNodesC[c], nNodesC[c]])
        # The matrix probMatrix has one in the block diagonal, which should
        # have probabilities p_ii and 0 in the offdiagonal that should have
        # probabilities p_ij. So that
        probMatrix = pii * probMatrix + pij * (1 - probMatrix)
        # has pii in the intracommunity blocks and pij in the intercommunity
        # blocks.
        # Now we're finally ready to generate a connected graph
        connectedGraph = False
        while not connectedGraph:
            # Generate random matrix
            W = np.random.rand(N,N)
            W = (W < probMatrix).astype(np.float64)
            # This matrix will have a 1 if the element ij is less or equal than
            # p_ij, so that if p_ij = 0.8, then it will be 1 80% of the times
            # (on average).
            # We need to make it undirected and without self-loops, so keep the
            # upper triangular part after the main diagonal
            W = np.triu(W, 1)
            # And add it to the lower triangular part
            W = W + W.T
            # Now let's check that it is connected
            connectedGraph = isConnected(W)
    elif graphType == 'fuseEdges':
        # This alternative assumes that there are multiple graphs that have to
        # be fused into one.
        # This will be done in two ways: average or sum.
        # On top, options will include: to symmetrize it or not, to make it
        # connected or not.
        # The input data is a tensor E x N x N where E are the multiple edge
        # features that we want to fuse.
        # Argument N is ignored
        # Data
        assert len(args) == 7
        W = args[0] # Data in format E x N x N
        assert len(W.shape) == 3
        N = W.shape[1] # Number of nodes
        assert W.shape[1] == W.shape[2]
        # Name the list with all nodes to keep
        nodeList = args[6] # This should be an empty list
        allNodes = np.arange(N)
        # What type of node aggregation
        aggregationType = args[1]
        assert aggregationType == 'sum' or aggregationType == 'avg'
        if aggregationType == 'sum':
            W = np.sum(W, axis = 0)
        elif aggregationType == 'avg':
            W = np.mean(W, axis = 0)
        # Normalization (sum of rows or columns is equal to 1)
        normalizationType = args[2]
        if normalizationType == 'rows':
            rowSum = np.sum(W, axis = 1).reshape([N, 1])
            rowSum[np.abs(rowSum) < zeroTolerance] = 1.
            W = W/np.tile(rowSum, [1, N])
        elif normalizationType == 'cols':
            colSum = np.sum(W, axis = 0).reshape([1, N])
            colSum[np.abs(colSum) < zeroTolerance] = 1.
            W = W/np.tile(colSum, [N, 1])
        # Discarding isolated nodes
        isolatedNodes = args[3] # if True, isolated nodes are allowed, if not,
        # discard them
        if isolatedNodes == False:
            # A Node is isolated when it's degree is zero
            degVector = np.sum(np.abs(W), axis = 0)
            # Keep nodes whose degree is not zero
            keepNodes = np.nonzero(degVector > zeroTolerance)
            # Get the first element of the output tuple, for some reason if
            # we take keepNodes, _ as the output it says it cannot unpack it.
            keepNodes = keepNodes[0]
            W = W[keepNodes][:, keepNodes]
            # Update the nodes kept
            allNodes = allNodes[keepNodes]
        # Check if we need to make it undirected or not
        forceUndirected = args[4] # if True, make it undirected by using the
            # average between nodes (careful, some edges might cancel)
        if forceUndirected == True:
            W = 0.5 * (W + W.T)
        # Finally, making it a connected graph
        forceConnected = args[5] # if True, make the graph connected
        if forceConnected == True:
            connectedFlag = isConnected(0.5 * (W + W.T))
            while connectedFlag == False:
                # We will remove nodes according to their degree, starting with
                # those of less degree. This has no connection whatsoever with
                # connectivity of the graph, and there should be a better 
                # solution. But this still gives a reasonable graph for the
                # cases considered.
                # Compute the degree
                degVector = np.sum(np.abs(W), axis = 0)
                # Sort them by degree (by default, argsort goes from smaller
                # to largest)
                sortedNodes = np.argsort(degVector)
                # Discard the ndoe with the smallest degree
                keepNodes = sortedNodes[1:]
                # Update W
                W = W[keepNodes][:,keepNodes]
                # Compute the connectedness
                connectedFlag = isConnected(0.5 * (W+W.T))
                # Update the list of nodes kept
                allNodes = allNodes[keepNodes]
                # This chooses nodes in the positions indicated by keep nodes
                # which has the same length as the actual allNodes, but 
                # allNodes has been keeping track of all the nodes so it is
                # actually taking out the proper nodes
        # To end, update the node list, so that it is returned through argument
        nodeList.extend(allNodes.tolist())
            
    return W

# Permutation functions

def permIdentity(S):
    """
    permIdentity: determines the identity permnutation

    Input:
        S (np.array): matrix

    Output:
        permS (np.array): matrix permuted (since, there's no permutation, it's
              the same input matrix)
        order (list): list of indices to make S become permS.
    """
    assert len(S.shape) == 2 or len(S.shape) == 3
    if len(S.shape) == 2:
        assert S.shape[0] == S.shape[1]
        S = S.reshape([1, S.shape[0], S.shape[1]])
        scalarWeights = True
    else:
        assert S.shape[1] == S.shape[2]
        scalarWeights = False
    # Number of nodes
    N = S.shape[1]
    # Identity order
    order = np.arange(N)
    # If the original GSO assumed scalar weights, get rid of the extra dimension
    if scalarWeights:
        S = S.reshape([N, N])

    return S, order.tolist()

def permDegree(S):
    """
    permDegree: determines the permutation by degree (nodes ordered from highest
        degree to lowest)

    Input:
        S (np.array): matrix

    Output:
        permS (np.array): matrix permuted
        order (list): list of indices to permute S to turn into permS.
    """
    assert len(S.shape) == 2 or len(S.shape) == 3
    if len(S.shape) == 2:
        assert S.shape[0] == S.shape[1]
        S = S.reshape([1, S.shape[0], S.shape[1]])
        scalarWeights = True
    else:
        assert S.shape[1] == S.shape[2]
        scalarWeights = False
    # Compute the degree
    d = np.sum(np.sum(S, axis = 1), axis = 0)
    # Sort ascending order (from min degree to max degree)
    order = np.argsort(d)
    # Reverse sorting
    order = np.flip(order,0)
    # And update S
    S = S[:,order,:][:,:,order]
    # If the original GSO assumed scalar weights, get rid of the extra dimension
    if scalarWeights:
        S = S.reshape([S.shape[1], S.shape[2]])

    return S, order.tolist()

def permSpectralProxies(S):
    """
    permSpectralProxies: determines the permutation by the spectral proxies
        score (from highest to lowest)

    Input:
        S (np.array): matrix

    Output:
        permS (np.array): matrix permuted
        order (list): list of indices to permute S to turn into permS.
    """
    # Design decisions:
    k = 8 # Parameter of the spectral proxies method. This is fixed for
    # consistency with the calls of the other permutation functions.
    # Design decisions: If we are given a multi-edge GSO, we're just going to
    # average all the edge dimensions and use that to compute the spectral
    # proxies.
    # Check S is of correct shape
    assert len(S.shape) == 2 or len(S.shape) == 3
    # If it is a matrix, just use it
    if len(S.shape) == 2:
        assert S.shape[0] == S.shape[1]
        scalarWeights = True
        simpleS = S.copy()
    # If it is a tensor of shape E x N x N, average over dimension E.
    else:
        assert S.shape[1] == S.shape[2]
        scalarWeights = False
        # Average over dimension E
        simpleS = np.mean(S, axis = 0)

    N = simpleS.shape[0] # Number of nodes
    ST = simpleS.conj().T # Transpose of S, needed for the method
    Sk = np.linalg.matrix_power(simpleS,k) # S^k
    STk = np.linalg.matrix_power(ST,k) # (S^T)^k
    STkSk = STk @ Sk # (S^T)^k * S^k, needed for the method

    nodes = [] # Where to save the nodes, order according the criteria
    it = 1
    M = N # This opens up the door if we want to use this code for the actual
    # selection of nodes, instead of just ordering

    while len(nodes) < M:
        remainingNodes = [n for n in range(N) if n not in nodes]
        # Computes the eigenvalue decomposition
        phi_eig, phi_ast_k = np.linalg.eig(
                STkSk[remainingNodes][:,remainingNodes])
        phi_ast_k = phi_ast_k[:][:,np.argmin(phi_eig.real)]
        abs_phi_ast_k_2 = np.square(np.absolute(phi_ast_k))
        newNodePos = np.argmax(abs_phi_ast_k_2)
        nodes.append(remainingNodes[newNodePos])
        it += 1

    if scalarWeights:
        S = S[nodes,:][:,nodes]
    else:
        S = S[:,nodes,:][:,:,nodes]
    return S, nodes

def permEDS(S):
    """
    permEDS: determines the permutation by the experimentally designed sampling
        score (from highest to lowest)

    Input:
        S (np.array): matrix

    Output:
        permS (np.array): matrix permuted
        order (list): list of indices to permute S to turn into permS.
    """
    # Design decisions: If we are given a multi-edge GSO, we're just going to
    # average all the edge dimensions and use that to compute the spectral
    # proxies.
    # Check S is of correct shape
    assert len(S.shape) == 2 or len(S.shape) == 3
    # If it is a matrix, just use it
    if len(S.shape) == 2:
        assert S.shape[0] == S.shape[1]
        scalarWeights = True
        simpleS = S.copy()
    # If it is a tensor of shape E x N x N, average over dimension E.
    else:
        assert S.shape[1] == S.shape[2]
        scalarWeights = False
        # Average over dimension E
        simpleS = np.mean(S, axis = 0)

    E, V = np.linalg.eig(simpleS) # Eigendecomposition of S
    kappa = np.max(np.absolute(V), axis=1)

    kappa2 = np.square(kappa) # The probabilities assigned to each node are
    # proportional to kappa2, so in the mean, the ones with largest kappa^2
    # would be "sampled" more often, and as suche are more important (i.e.
    # they have a higher score)

    # Sort ascending order (from min degree to max degree)
    order = np.argsort(kappa2)
    # Reverse sorting
    order = np.flip(order,0)

    if scalarWeights:
        S = S[order,:][:,order]
    else:
        S = S[:,order,:][:,:,order]

    return S, order.tolist()


class Graph():
    """
    Graph: class to handle a graph with several of its properties

    Initialization:

        graphType ('SBM'): graph type, others coming soon.
        N (int): number of nodes
        [optionalArguments]: related to the specific type of graph. Details
            coming soon.

    Attributes:

        .N (int): number of nodes
        .M (int): number of edges
        .W (np.array): weighted adjacency matrix
        .D (np.array): degree matrix
        .A (np.array): unweighted adjacency matrix
        .L (np.array): Laplacian matrix (if graph is undirected and has no
           self-loops)
        .S (np.array): graph shift operator (weighted adjacency matrix by
           default)
        .E (np.array): eigenvalue (diag) matrix (graph frequency coefficients)
        .V (np.array): eigenvector matrix (graph frequency basis)
        .undirected (bool): True if the graph is undirected
        .selfLoops (bool): True if the graph has self-loops

    Methods:

        .setGSO(S, GFT = 'no'): sets a new GSO
        Inputs:
            S (np.array): new GSO matrix (has to have the same number of nodes),
                updates attribute .S
            GFT ('no', 'increasing' or 'totalVariation'): order of
                eigendecomposition; if 'no', no eigendecomposition is made, and
                the attributes .V and .E are set to None
    """
    # in this class we provide, easily as attributes, the basic notions of
    # a graph. This serve as a building block for more complex notions as well.
    def __init__(self, graphType, N, *args):
        assert N > 0
        #\\\ Create the graph (Outputs adjacency matrix):
        self.W = createGraph(graphType, N, *args)
        # TODO: Let's start easy: make it just an N x N matrix. We'll see later
        # the rest of the things just as handling multiple features and stuff.
        #\\\ Number of nodes:
        self.N = (self.W).shape[0]
        #\\\ Bool for graph being undirected:
        self.undirected = np.allclose(self.W, (self.W).T, atol = zeroTolerance)
        #   np.allclose() gives true if matrices W and W.T are the same up to
        #   atol.
        #\\\ Bool for graph having self-loops:
        self.selfLoops = True \
                        if np.sum(np.abs(np.diag(self.W)) > zeroTolerance) > 0 \
                        else False
        #\\\ Degree matrix:
        self.D = np.diag(np.sum(self.W, axis = 1))
        #\\\ Number of edges:
        self.M = int(np.sum(np.triu(self.W)) if self.undirected \
                                                    else np.sum(self.W))
        #\\\ Unweighted adjacency:
        self.A = (np.abs(self.W) > 0).astype(self.W.dtype)
        #\\\ Laplacian matrix:
        #   Only if the graph is undirected and has no self-loops
        if self.undirected and not self.selfLoops:
            self.L = adjacencyToLaplacian(self.W)
        else:
            self.L = None
        #\\\ GSO (Graph Shift Operator):
        #   The weighted adjacency matrix by default
        self.S = self.W
        #\\\ GFT: Declare variables but do not compute it unless specifically
        # requested
        self.E = None # Eigenvalues
        self.V = None # Eigenvectors
    
    def computeGFT(self):
        # Compute the GFT of the stored GSO
        if self.S is not None:
            #\\ GFT:
            #   Compute the eigenvalues (E) and eigenvectors (V)
            self.E, self.V = computeGFT(self.S, order = 'totalVariation')

    def setGSO(self, S, GFT = 'no'):
        # This simply sets a matrix as a new GSO. It has to have the same number
        # of nodes (otherwise, it's a different graph!) and it can or cannot
        # compute the GFT, depending on the options for GFT
        assert S.shape[0] == S.shape[1] == self.N
        assert GFT == 'no' or GFT == 'increasing' or GFT == 'totalVariation'
        # Set the new GSO
        self.S = S
        if GFT == 'no':
            self.E = None
            self.V = None
        else:
            self.E, self.V = computeGFT(self.S, order = GFT)

def splineBasis(K, x, degree=3):
    # Function taken verbatim (except for function name), from 
    # https://github.com/mdeff/cnn_graph/blob/master/lib/models.py#L662
    """
    Return the B-spline basis.
    K: number of control points.
    x: evaluation points
       or number of evenly distributed evaluation points.
    degree: degree of the spline. Cubic spline by default.
    """
    if np.isscalar(x):
        x = np.linspace(0, 1, x)

    # Evenly distributed knot vectors.
    kv1 = x.min() * np.ones(degree)
    kv2 = np.linspace(x.min(), x.max(), K-degree+1)
    kv3 = x.max() * np.ones(degree)
    kv = np.concatenate((kv1, kv2, kv3))

    # Cox - DeBoor recursive function to compute one spline over x.
    def cox_deboor(k, d):
        # Test for end conditions, the rectangular degree zero spline.
        if (d == 0):
            return ((x - kv[k] >= 0) & (x - kv[k + 1] < 0)).astype(int)

        denom1 = kv[k + d] - kv[k]
        term1 = 0
        if denom1 > 0:
            term1 = ((x - kv[k]) / denom1) * cox_deboor(k, d - 1)

        denom2 = kv[k + d + 1] - kv[k + 1]
        term2 = 0
        if denom2 > 0:
            term2 = ((-(x - kv[k + d + 1]) / denom2) * cox_deboor(k + 1, d - 1))

        return term1 + term2

    # Compute basis for each point
    basis = np.column_stack([cox_deboor(k, degree) for k in range(K)])
    basis[-1,-1] = 1
    return basis