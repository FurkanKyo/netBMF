
#######################
#### Network-constrained Binary Matrix Factorization (netBMF)
# How to use:
#       python netBMF.py <inputfile> <outputfile> <isMH> <isNaive> <numPatterns_k> <maxIter> <minSize> <minFreq> <numSeeds> <seedStragey>
# Example call:
#       python -u netBMF.py data/smallbehance.mat smallbehance_output.mat 0 0 5 40 2 2 5 0
# Input parameter:
#       inputfile: a *.mat file containing S & G matrices
#       outputfile: to store the resulting M & B
#       isMH (0/1): 1 if MH sampling, 0 if MCMC sampling
#       isNaive (0/1): 1 if netBMF-naive, 0 if netBMF-incremental
#       numPatterns_k: number of patterns to find
#       maxIter: maximum number of steps in sampling (for each seed node)
#       minSize: size threshold (minimum number of nodes in a pattern)
#       minFreq: frequency threshold (minimum frequency of a valid pattern)
#       seedStrategy (0/1): how to pick seed nodes for sampling; 0 if based on node frequency, 1 if based on node pair frequency
#
# Inputfile format: 2 binary matrices in a single *.mat file (MATLAB file format)
#       S (network_states x nodes) : network state matrix
#       G (nodes x nodes) : symmetric adjacency matrix of an undirected graph among nodes
#   The input matrix can be normal matrices or sparse matrices in MATLAB for large data.
# Outputfile format: 2 binary matrices in a single *.mat file (MATLAB file format)
#       M (network_states x patterns) : mapping matrix (which network states contain which patterns)
#       B (patterns x nodes) : pattern (basis) matrix which patterns contain which nodes
#
# Required python library: numpy, scipy, networkx
#######################
__author__ = "Furkan Kocayusufoglu"

import time
import networkx as nx
from random import choice
import random
from random import randint
import numpy as np
import sys
import scipy.io as sio
import heapq
from operator import itemgetter
import scipy.sparse as ss

def main():
    start = time.time()
    GSfile = sys.argv[1]
    outfile = sys.argv[2]
    isMH = int(sys.argv[3]) # 0: MCMC, 1: MH
    isNaive = int(sys.argv[4]) # 0: incre, 1: naive
    numberofpatterns = int(sys.argv[5])
    numberofiterations = int(sys.argv[6])
    minPatternSize = int(sys.argv[7])
    minPatternFreq = int(sys.argv[8])
    numSeeds = int(sys.argv[9])
    seedStrategy = int(sys.argv[10]) # 0: nodefreq, 1: nodepairfreq
    
    typestr = ('MH' if isMH else 'MCMC') + ('naive' if isNaive else 'incre')

    params = {'inputFile':GSfile,'numPatterns' : numberofpatterns, 'minSize' : minPatternSize, 'minFreq':minPatternFreq, 'numSeeds' :numSeeds, 'isMH':isMH,'numIters':numberofiterations,'isNaive':isNaive,'seedStrategy':seedStrategy}

    delAddProb = [0.3,0.7]
    mat = sio.loadmat(GSfile, squeeze_me=True)
    S = ss.csc_matrix(mat['S'])
    G = ss.csc_matrix(mat['G'])
    numNodes = S.shape[1] #G.shape[0]
    numStates = S.shape[0]

    print("G and S Read...", numNodes, "nodes.", numStates, "states. DONE!", time.time()-start, 'seconds')

    Strans = (S.transpose() > 0).tolil()
    NodeFreqDict = [set(l) for l in Strans.rows]
    InitialNodeFreqDict = [l.copy() for l in NodeFreqDict]
    print("NodeDicts... DONE!", time.time()-start, 'seconds')

    # Covered set
    C = np.empty(S.shape)

    firstThreshold = 1
    count = 0
    patternSet = [[] for _ in range(numberofpatterns)]
    patternSetDict = [[] for _ in range(numberofpatterns)]
    terminate = False
    innerGains = []
    singleNodeGains = [] 
    Sflag = S.todok()
    runningTime = [0] * numberofpatterns
    start = time.time()
    while count < numberofpatterns and terminate == False:
        innerIteration = 0
        alreadyPicked = []

        while innerIteration < numSeeds:
            # Get new prob distribution after update
            GNodes = list(set(range(numNodes)) - set(alreadyPicked))
            if minPatternFreq > 1:
                InitialNodeProb = ComputeInitialPairFreqProb(Sflag, GNodes, G) 
            else:
                InitialNodeProb = ComputeInitialNodeProb(NodeFreqDict, GNodes)

            if InitialNodeProb is None:
                terminate = innerIteration == 0
                break
            
            N = np.random.choice(GNodes, p=InitialNodeProb)
            alreadyPicked.append(N)
            if len(NodeFreqDict[N]) == 0:
                continue
                
            curGain = len(NodeFreqDict[N])
            patternNodes = [N]
            patternSubgraphs = list(InitialNodeFreqDict[N])
            maxPatternNodes = patternNodes
            maxPatternSubgraphs = patternSubgraphs
            maxPatternGain = curGain
            for iteration in range(numberofiterations):
                addition = np.random.rand() <= delAddProb[True]
                Neighbors = findEditNodes(patternNodes,G,addition)
                if len(Neighbors) == 0:
                    continue
                NeighborNodeProb = ComputeNeighborNodeProb(NodeFreqDict, patternSubgraphs, Neighbors, addition)
                Neighbor = np.random.choice(Neighbors, p=[NeighborNodeProb[x] for x in Neighbors])           

                # new candidate pattern
                if addition:
                    tempPatternNodes = patternNodes + [Neighbor]
                else:
                    tempPatternNodes = list(set(patternNodes) - set([Neighbor]))

                TotalGain, SubgraphsAfter = computeMarginalGain(S, C, tempPatternNodes, InitialNodeFreqDict, isNaive)
                    
                # compute the acceptance probability
                if isMH:
                    TempNeighbors = findEditNodes(tempPatternNodes,G,not addition) 
                    TempNeighborNodeProb = ComputeNeighborNodeProb(NodeFreqDict, SubgraphsAfter, TempNeighbors, (not addition))

                    Qij = NeighborNodeProb[Neighbor] * delAddProb[addition]
                    Qji = TempNeighborNodeProb[Neighbor] * delAddProb[not addition]
                    acceptProb = TotalGain/curGain * Qji / Qij
                else: #MCMC                        
                    acceptProb = TotalGain/curGain
                    
                if acceptProb >= np.random.rand():
                    patternNodes = tempPatternNodes
                    patternSubgraphs = SubgraphsAfter
                    curGain = TotalGain
                    if TotalGain > maxPatternGain and len(patternNodes) >= minPatternSize and len(patternSubgraphs) >= minPatternFreq:
                        maxPatternGain = TotalGain
                        maxPatternNodes = patternNodes
                        maxPatternSubgraphs = patternSubgraphs
            patternNodes = maxPatternNodes
            patternSubgraphs = maxPatternSubgraphs
            patternGain = maxPatternGain
        
            ## try improving the patterns by removing and adding nodes
            patternNodes, patternSubgraphs, patternGain = improve_pattern(patternNodes, patternSubgraphs, patternGain, G, InitialNodeFreqDict, S, C, minPatternFreq, minPatternSize, isNaive)
            patternNodes.sort()
            patternSubgraphs.sort()

            if ((patternGain, patternNodes, patternSubgraphs, len(patternNodes), len(patternSubgraphs)) not in innerGains) and (len(patternSubgraphs) >= minPatternFreq) and (len(patternNodes) >= minPatternSize):
                innerGains.append((patternGain, patternNodes, patternSubgraphs, len(patternNodes), len(patternSubgraphs)))
            innerIteration += 1        
        ## cannot find any good pattern --> stop sampling
        if len(innerGains) == 0:
            print('No more good patterns.')
            break

        ## sort the sampled subgraphs to find the best ones
        innerGains = sorted(innerGains, key=itemgetter(0,3))

        ## pick the best patterns
        maxPattern = innerGains.pop()
        patternGain, patternNodes, patternSubgraphs = maxPattern[:3]
        patternSet[count] = patternNodes[:]
        patternSetDict[count] = patternSubgraphs[:]

        # Update NodeFreqDict
        for n in patternNodes:
            C[patternSubgraphs, n] = 1
            for s in patternSubgraphs:
                if s in NodeFreqDict[n]:
                    NodeFreqDict[n].remove(s)

        Mcol = ss.csc_matrix(([1] * len(patternSubgraphs), (patternSubgraphs, [0] * len(patternSubgraphs))), shape=(numStates, 1))
        Brow = ss.csr_matrix(([1] * len(patternNodes), ([0] * len(patternNodes), patternNodes)), shape=(1, numNodes))
        Sflag[Mcol * Brow > 0] = 0

        ## update the gain and frequency of candidate patterns
        innerGains = updateGains(S, C, innerGains, InitialNodeFreqDict, minPatternFreq, minPatternSize, isNaive)

        ## record timing
        runningTime[count] = time.time()-start
        print("")
        print(typestr + ": P" + str(count) + ": " + str(patternSet[count]) + " with gain " + str(patternGain), '. Time:',runningTime[count],'seconds\n')
        count += 1

    # Calculate Accuracy and Error --TODO: Convert the calculations to sparce matrix format!!
    MatrixM = np.zeros(shape=(numStates, numberofpatterns))
    MatrixB = np.zeros(shape=(numberofpatterns, numNodes))
    DecomposedS = np.zeros(shape=(numStates,numNodes))
    for i in range(0, numberofpatterns):
        MatrixM[patternSetDict[i], i] = 1
        MatrixB[i, patternSet[i]] = 1
        for k in patternSetDict[i]:
            DecomposedS[k, patternSet[i]] = 1
    Sparse_decomposedS = ss.csc_matrix(DecomposedS)
    error = np.sum(np.abs(S - Sparse_decomposedS)) / np.sum(S)
    accuracy = 1-error

    print(typestr, sys.argv, ". ACCURACY:", accuracy, 'numPatterns:', count, '. Running Time:', runningTime[count-1], 'seconds')

    sio.savemat(outfile, {'B': MatrixB, 'M': MatrixM, 'runningTime':runningTime, 'params':params,'error':error,'numPatterns':count})
    print("Saved to ", outfile)

# HELPER FUNCTIONS
def updateGains(S, C, Gains, NodeFreqDict, minPatternFreq, minPatternSize, isNaive):
    GainsTemp = []
    for i in range(len(Gains)):
        patternNodes = Gains[i][1]
        patternNodes.sort()
        SubgraphsFound = computeMappedStates(patternNodes, NodeFreqDict, isNaive)
        if not SubgraphsFound:
            continue
        SubgraphsFound.sort()

        MatrixM = np.zeros(S.shape[1])
        MatrixM[patternNodes] = 1        
        totalGain = 0
        for s in SubgraphsFound:
            MatrixW = np.full(S.shape[1],-1,np.int)
            MatrixW[S[s,:].nonzero()[1]] = 1
            MatrixW[C[s,].nonzero()[0]] = 0
            totalGain += np.dot(MatrixW, MatrixM)

        patternSize = len(patternNodes)
        patternFreq = len(SubgraphsFound)

        if totalGain > 0 and patternFreq >= minPatternFreq and patternSize >= minPatternSize:
            GainsTemp.append((totalGain, patternNodes, SubgraphsFound, patternSize, patternFreq))

    return GainsTemp

def findAddableNodes(curNodes,G):
    '''get the nodes that are neighbors of curNodes in G
    G is given as a sparse adjacency matrix
    '''
    if not curNodes:
        return []
    return list(set(ss.find(G[:,curNodes])[0])-set(curNodes))

def findRemovableNodes(curNodes,G):
    '''get the nodes from curNodes that can be removed without disconnecting the subgraph
    '''
    if not curNodes or len(curNodes) < 2:
        return []
    g = nx.from_scipy_sparse_matrix(G.tocsr()[curNodes, :].tocsc()[:, curNodes])
    return list(set(curNodes) - set([curNodes[i] for i in nx.articulation_points(g)]))

def findEditNodes(curNodes,G,isAddition):
    '''find nodes that can be added or delete
    so that the new subgraph is still connected
    '''
    return findAddableNodes(curNodes,G) if isAddition else findRemovableNodes(curNodes,G)

def ComputeInitialNodeProb(NodeFreqDict, GNodes):
    freq = [len(NodeFreqDict[x])*len(NodeFreqDict[x]) for x in GNodes]
    totalFreq = float(sum(freq))
    if totalFreq > 0:
        return [f/totalFreq for f in freq]
    return None

def ComputeInitialPairFreqProb(Sflag, GNodes, G):
    A = (Sflag.transpose() * Sflag).multiply(G)
    ego = A.sum(0).tolist()[0]
    pairfreq = [ego[x]*ego[x] for x in GNodes]
    totalFreq = float(sum(pairfreq))
    if totalFreq > 0:
        return [f/totalFreq for f in pairfreq]
    return None
	
def ComputeNeighborNodeProb(NodeFreqDict, patternSubgraphs, Neighbors, Addition):
    patternSubgraphs = set(patternSubgraphs)
    if Addition:
        potential = np.array([len(patternSubgraphs & NodeFreqDict[n]) + 1 for n in Neighbors])
    else:
        potential = np.array([1.0/(len(patternSubgraphs & NodeFreqDict[n]) + 1) for n in Neighbors])
    sumCount = float(sum(potential))

    if sumCount > 0.0:
        NeighborNodeProb = dict(zip(Neighbors,potential/sumCount))
    else:
        NeighborNodeProb = dict(zip(Neighbors,[1.0/len(Neighbors)] * len(Neighbors)))
    return NeighborNodeProb

def ComputeNeighborNodeOverlapStates(NodeFreqDict, patternSubgraphs, Neighbors):
    patternSubgraphs = set(patternSubgraphs)
    overlap = np.array([len(patternSubgraphs & NodeFreqDict[n]) for n in Neighbors])
    return overlap

def computeMarginalGain(S, C, patternNodes, NodeFreqDict, isNaive):
	SubgraphsFound = computeMappedStates(patternNodes, NodeFreqDict, isNaive)
	if len(SubgraphsFound) == 0:
		return 0.001, []
	MatrixM = np.zeros(S.shape[1])
	MatrixM[patternNodes] = 1
	totalGain = 0
	for s in SubgraphsFound:
		MatrixW = np.full(S.shape[1],-1,np.int)
		MatrixW[S[s,:].nonzero()[1]] = 1
		MatrixW[C[s,].nonzero()[0]] = 0
		totalGain += np.dot(MatrixW, MatrixM)
	return totalGain if totalGain > 0 else 0.001, SubgraphsFound

def computeMappedStates(patternNodes, NodeFreqDict, isNaive):
    if len(patternNodes) == 0:
        return []

    if isNaive:
        SubgraphsFound = NodeFreqDict[patternNodes[0]].copy()
        for i in range(1,len(patternNodes)):
            SubgraphsFound &= NodeFreqDict[patternNodes[i]]
            if not SubgraphsFound:
                break
        SubgraphsFound = list(SubgraphsFound)
    else:
        halfPatternSize = len(patternNodes)/2.0
        SubgraphOverlapDict = {}
        for n in patternNodes:
            for s in NodeFreqDict[n]:
                SubgraphOverlapDict[s] = SubgraphOverlapDict.get(s,0) + 1
        SubgraphsFound = [s for s in SubgraphOverlapDict if SubgraphOverlapDict[s] > halfPatternSize]

    return SubgraphsFound

def improve_pattern(patternNodes, patternSubgraphs, patternGain, G, InitialNodeFreqDict, S, C, minPatternFreq, minPatternSize, isNaive):
    canImprove = True
    while canImprove:
        canImprove = False
        # try removing nodes
        isChanged = True
        while isChanged and len(patternNodes) > minPatternSize:
            isChanged = False
            FinalNonCutSet = findRemovableNodes(patternNodes,G)
            FinalNonCutSet.sort()
            for i,nc in enumerate(FinalNonCutSet):
                tempPatternNodes = list(set(patternNodes) - set([nc]))
                TotalGain, SubgraphsAfter = computeMarginalGain(S, C, tempPatternNodes, InitialNodeFreqDict, isNaive)
                bestPatternGain = patternGain
                if TotalGain > bestPatternGain:
                    bestPatternNodes = tempPatternNodes
                    bestPatternGain = TotalGain
                    bestPatternSubgraphs = SubgraphsAfter
                    isChanged = True
                    canImprove = True
            if isChanged:
                patternNodes = bestPatternNodes
                patternGain = bestPatternGain
                patternSubgraphs = bestPatternSubgraphs
        # try adding nodes
        isChanged = True
        while isChanged:
            isChanged = False
            Neighbors = findAddableNodes(patternNodes,G)
            Neighbors.sort()
            for i,n in enumerate(Neighbors):
                tempPatternNodes = patternNodes + [n]
                TotalGain, SubgraphsAfter = computeMarginalGain(S, C, tempPatternNodes, InitialNodeFreqDict, isNaive)
                if TotalGain > patternGain and len(SubgraphsAfter) >= minPatternFreq:
                    patternNodes = tempPatternNodes
                    patternSubgraphs = SubgraphsAfter
                    patternGain = TotalGain
                    isChanged = True
                    canImprove = True
            # only try adding once
            break
        break

    return patternNodes, patternSubgraphs, patternGain

if __name__ == '__main__':
    sys.exit(main())
