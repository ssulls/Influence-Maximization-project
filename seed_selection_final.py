#!/usr/bin/env python
# coding: utf-8

# In[1]:


#  ALL NECESSARY IMPORTS ::


# In[2]:


#product for generating all permutations for seed selection parameter fine-tuning
from itertools import product
#itemgetter for sorting lists of tuples / dicts by their second element / value
from operator import itemgetter
#math.log for calculating logs of probabilities in WC1 and WC2
from math import log
#copy.deepcopy for deepcopying graphs in seed selection models
from copy import deepcopy
#numpy to manipulate lists, calculate means, etc.
import numpy as np
#
import pandas as pd
#time.time to calculate and compare running time and efficiency
from time import time
#Counter to count frequencies in lists, for averaging edges in seed selection models
from collections import Counter
#networkx to generate and handle networks from data
import networkx as nx
#csv to extract network data from .csv files
import csv
#winsound to alert me when propagation is complete for long proccessing periods
import winsound
#matplotlib.pyplot for plotting graphs and charts
import matplotlib.pyplot as plt
#matplotlib.offsetbox.AnchoredText for anchored textboxes on plotted figures
from matplotlib.offsetbox import AnchoredText

#Dataset dictionary
#Title : weighted, directed, filepath to .csv file
datasets = {
    #BitcoinOTC dataset (5875 nodes, 35592 edges)
    #(directed, weighted, signed)
    "BitcoinOTC": (True, True, 
         r"D:\Sully\Documents\Computer Science BSc\Year 3\Term 2\Individual Project\datasets\soc-sign-bitcoinotc.csv"),
    #Facebook dataset (4039 nodes, 88234 edges)
    #(undirected, unweighted, unsigned)
    "Facebook": (False, False, 
         r"D:\Sully\Documents\Computer Science BSc\Year 3\Term 2\Individual Project\datasets\facebook.csv")
}


# In[3]:


#  CASCADE, ITERATION, PROPAGATION & SUCCESS FUNCTIONS ::
#
#  Functions to perform cascade & propagation on a given graph 
#  with a given seed set and a given number of iterations of a 
#  given cascade model.
#
#  Functions included:
#  1. Model-specific success functions
#  2. Propagation function
#  3. Iteration function
#  4. Cascade (ties everything together) function


# In[4]:


#Determine propagation success for the various models
#(includes quality factor to differentiate positive/negative influence)
#(includes a switch penalty for nodes switching sign)

#Apply quality factor and switch factor variables
def successVars(sign, switch, qf=1, sf=1):
    if not sign:
        qf = (1-qf)
    if not switch:
        sf = 0
    return qf*(1-sf)

#Calculate whether propagation is successful (model-specific)
def success(successModel, sign, switch, timeDelay, g, target, targeting, pp, qf, sf, a):
    if successModel == 'IC':
        succ = (pp*successVars(sign, switch, qf, sf)*g[targeting][target]['trust']*timeDelay)
    elif successModel == 'WC1':
        if a:
            recip = g.nodes[target]['degRecip']
        else:
            recip = (1 / g.in_degree(target))
        succ = (recip*successVars(sign, switch, qf, sf)*g[targeting][target]['trust']*timeDelay)
    elif successModel == 'WC2':
        if a:
            relDeg = g[targeting][target]['relDeg']
        else:
            snd = sum([(g.out_degree(neighbour)) for neighbour in g.predecessors(target)])
            relDeg = (g.out_degree(targeting) / snd)
        succ = (relDeg*successVars(sign, switch, qf, sf)*timeDelay*g[targeting][target]['trust'])
    return np.random.uniform(0,1) < succ

#Returns probability with only the variables
#(no trust values, degree reciprocals or relational degrees)
def basicProb(weighted=False, *nodes):
    return pp * successVars(True, False)


# In[5]:


#One complete turn of propagation from a given set of the newly
#  activated (positive & negative) nodes from the last turn.
#(1. new negative nodes attempt to negatively influence their neighbours)
#(2. new positive nodes attempt to positively influence their neighbours)
#(3. new positive nodes attempt to negatively influence their neighbours)
def propagateTurn(g, pn, pos, nn, neg, trv, td, successMod, pp, qf, sf, a):
    posCurrent, negCurrent = set(), set()
    for node in nn:
        for neighbour in g.neighbors(node):
            if (node, neighbour) not in trv:
                #Negative influence to neighbours of negative nodes
                if success(successMod, False, (neighbour in pos), td, g, neighbour, node, pp, qf, sf, a):
                    negCurrent.add(neighbour)
                trv.add((neighbour, node))
                trv.add((node, neighbour))
                
    for node in pn:
        for neighbour in g.neighbors(node):
            if (node, neighbour) not in trv:
                #Positive influence to neighbours of positive nodes
                if neighbour not in negCurrent and success(successMod, True, (neighbour in neg), td, g, neighbour, node, pp, qf, sf, a):
                    posCurrent.add(neighbour)
                #Negative influence to neighbours of positive nodes
                elif neighbour not in negCurrent and neighbour not in posCurrent and success(successMod, False, (neighbour in pos), td, g, neighbour, node, pp, qf, sf, a):
                    negCurrent.add(neighbour)
                trv.add((neighbour, node))
                trv.add((node, neighbour))
    return(posCurrent, negCurrent, trv)


# In[6]:


#Calculate average positive spread over a given number of iterations
def iterate(g, s, it, successFunc, pp, qf, sf, tf, ret, a):
    #If no number of iterations is given, one is calculated based on the
    #  ratio of nodes to edges within the graph, capped at 2000.
    if not it:
        neRatio = (len(g)/(g.size()))
        if neRatio > 0.555:
            it = 2000
        else:
            it = ((neRatio/0.165)**(1/1.75))*1000
    influence = []
    if ret:
        infCount = []
    for i in range(it):
        #Randomness seeded for repeatability & robustness
        np.random.seed(i)
        positive, posNew, negative, negNew, traversed, timeFactor = set(), set(s), set(), set(), set(), 1
        #while there are newly influenced nodes from last turn...
        while posNew or negNew:
            #new nodes assigned to placeholder variables
            posLastTurn, negLastTurn = posNew, negNew
            #propagation turn is performed, returning positive&negative nodes and traversed edges
            posNew, negNew, traversed = propagateTurn(g, posNew, positive, negNew, negative, traversed, timeFactor, successFunc, pp, qf, sf, a)
            #Positive and negative nodes are recalculated
            positive, negative = (positive.union(posNew, posLastTurn) - negNew), (negative.union(negNew, negLastTurn) - posNew)
            #Time delay is taken away from the time factor
            timeFactor -= tf
        if ret:
            #Positive nodes added to list
            for p in positive:
                influence.append(p)
            #Number of nodes added to list
            infCount.append(len(positive))
        else:
            #Number of positive nodes added to list
            influence.append(len(positive))
    #If nodes are being returned
    if ret:
        #Average list of positive nodes are returned
        counts = Counter(influence)
        result = (sorted(counts, key=counts.get, reverse=True))[:int(np.mean(infCount))]
    #If nodes aren't being returned
    else:
        #Mean is returned
        result = np.mean(influence)
    return result


# In[7]:


#Propagation probability declared outside the function, because
#  some seed selection models use it without the cascade function.
pp = 0.2
#Determine the cascade model and run the iteration function 
#  with the appropriate success function
def cascade(g, s, its=0, 
            model='IC', assign=1, ret=False, 
            p=pp, qf=0.7, sf=0.8, tf=0.04):
    #g = graph, s = set of seed nodes, its = num of iterations
    #model = cascade model, #assign model, #return nodes?
    #p = propagation probability, qf = quality factor
    #sf = switch factor, tf = time factor
    #Model is determined and appropriate success function is assigned
    #print(f'model = {model},  assign = {assign}  its = {its}\npp = {p}, qf = {qf}, sf = {sf}, tf = {tf} \n')
    if model != 'IC' and assign:
        assignSelect(g, model, assign)
    success = model
    return iterate(g, s, its, success, p, qf, sf, tf, ret, assign)

#Propagation models and their names are compiled into a list
propMods = [('IC', "Independent Cascade"), 
            ('WC1', "Weighted Cascade 1"), 
            ('WC2', "Weighted Cascade 2")]


# In[8]:


#Return a set of all reachable nodes from a given node or set of nodes,
#  by recursive depth-first traversal.
#Used in improved greedy & mixed greedy seed selection models
def reach(g, node, reached, traversed):
    for neighbour in g.neighbors(node):
        if (node,neighbour) not in traversed and neighbour not in reached:
            reached.add(neighbour)
            traversed.add((node, neighbour))
            reached, traversed = reach(g, neighbour, reached, traversed)
    return reached, traversed

def reachable(g, s):
    reached, traversed = set(), set()
    for node in s:
        if node not in g:
            continue
        else:
            reached.add(node)
        reached, traversed = reach(g, node, reached, traversed)
    return reached


# In[9]:


#  NETWORK GRAPH SETUP ::
#
#  Functions to generate network graphs from various csv files,
#  and assign meaningful attributes to the nodes/edges to save
#  processing time during propagation.
#
#  Datasets/graphs included:
#  1. soc-BitcoinOTC
#  2. ego-Facebook


# In[10]:


#Removes any unconnected components of a given graph
def removeUnconnected(g):
    components = sorted(list(nx.weakly_connected_components(g)), key=len)
    while len(components)>1:
        component = components[0]
        for node in component:
            g.remove_node(node)
        components = components[1:]


# In[11]:


#Generates NetworkX graph from given file path:
def generateNetwork(name, weighted, directed, path):
    #graph is initialized and named, dataframe is initialized
    newG = nx.DiGraph(Graph = name)
    data = pd.DataFrame()
    #pandas dataframe is read from .csv file,
    #  with weight if weighted, without if not
    if weighted:
        data = pd.read_csv(path, header=None, usecols=[0,1,2],
                           names=['Node 1', 'Node 2', 'Weight'])
        wMax, wMin = data[['Weight']].max().item(), data[['Weight']].min().item()
    else:
        data = pd.read_csv(path, header=None, usecols=[0,1],
                           names=['Node 1', 'Node 2'])
    #offset is calculated from minimum nodes
    nodeOffset = min(data[['Node 1', 'Node 2']].min())
    #each row of dataframe is added to graph as an edge
    for row in data.itertuples(False, None):
        #trust=weight, & distance=(1-trust)
        if weighted:
            trustval = ((row[2]-wMin)/(wMax-wMin))
        else:
            trustval = 1
        newG.add_edge(row[1]-nodeOffset, row[0]-nodeOffset, 
                      trust=trustval, distance=(1-trustval))
        #if graph is undirected, edges are added again in reverse
        if not directed:
            newG.add_edge(row[0]-nodeOffset, row[1]-nodeOffset, 
                          trust=trustval, distance=(1-trustval))
    #unconnected components are removed
    if directed:
        removeUnconnected(newG)
    return newG


# In[12]:


#Generate graphs and compile into dictionaries:

#Dictionaries for groups of graphs are intialized
graphs, mockGraphs, rndmGraphs, diGraphs, allGraphs = {}, {}, {}, {}, {}

#Generate graphs from real datasets using the datasets dictionary
for g in datasets:
    realGraph = generateNetwork((g + " Network"), 
                                datasets[g][0], datasets[g][1], 
                                datasets[g][2])
    graphs[g], diGraphs[g], allGraphs[g] = realGraph, realGraph, realGraph


#Generate various mock graphs for testing and debugging:

#Custom, small directed, unweighted graph
mockG, name = nx.DiGraph(), "mock-custom"
testedges = [(1,2), (2,4), (2,5), (2,6), (3,5), (4,5), (5,9), (5,10), (6,8),
            (7,8), (8,9)]
mockG.add_edges_from(testedges)
nx.set_edge_attributes(mockG, 1, 'trust')
mockGraphs[name], diGraphs[name] = mockG, mockG

#Medium-sized path graph
#(each node only has edges to the node before and/or after it)
mockG, name = nx.path_graph(100), "mock-path"
nx.set_edge_attributes(mockG, 1, 'trust')
mockGraphs[name] = mockG

#Medium-sized, randomly generated directed, unweighted graph
mockG, name = nx.DiGraph(), "mock3-random1"
for i in range(50):
    for j in range(10):
        targ = np.random.randint(-40,50)
        if targ > -1:
            mockG.add_edge(i, targ, trust=1)
mockGraphs[name], rndmGraphs[name], diGraphs[name] = mockG, mockG, mockG

#Medium-sized, randomly generated directed, randomly-weighted graph
mockG, name = nx.DiGraph(), "mock4-random2"
for i in range(50):
    for j in range(10):
        targ = np.random.randint(-40,50)
        if targ > -1:
            tru = np.random.uniform(0,1)
            mockG.add_edge(i, targ, trust=tru)
mockGraphs[name], rndmGraphs[name], diGraphs[name] = mockG, mockG, mockG

#Functional testing for new graphing method
"""
for gl in [realGraphs, mockGraphs]:
    for g in gl:
        print(g + ": " + str(gl[g].size()))
        print(g + ": " + str(len(gl[g])) + "\n")
#"""
print("")


# In[13]:


#Normalize (Min-Max) every value in a given dictionary
def mmNormalizeDict(dic, elMax, elMin):
    #for key, value in dic.items():
    #    dic[key] = ((value - elMin) / (elMax - elMin))
    #printResults("Assigned", dic.values())
    #print("Assigned normalization:\nMax = " + str(elMax) + "\nMin = " 
    #      + str(elMin) + "\nMean = " 
    #      + str(np.mean(list(dic.values()))))
    #return dic
    return {key: ((val - elMin)/(elMax - elMin)) for key,val in dic.items()}


# In[14]:


#Methods that assign probabilities for WC1 & WC2 to nodes or edges

#Calculate manipulated degree-reciprocals for all nodes in a graph, and
#  assign them as node attributes for the Weighted Cascade 1 model

#Log-scaling method - default if not specified
def assignRecips1(g):
    drs = {}
    for target in g:
        if not g.in_degree(target):
            continue
        drs[target] = log(1 / g.in_degree(target))
    elMax = drs[max(drs, key=drs.get)]
    elMin = drs[min(drs, key=drs.get)]
    drs = mmNormalizeDict(drs, elMax, elMin)
    nx.set_node_attributes(g, drs, "degRecip")

#Square-rooting method
def assignRecips2(g):
    drs = {}
    for target in g:
        if not g.in_degree(target):
            continue
        drs[target] = ((1 / g.in_degree(target)) ** (1/2))
    nx.set_node_attributes(g, drs, "degRecip")
    
#Cube-rooting method
def assignRecips3(g):
    drs = {}
    for target in g:
        if not g.in_degree(target):
            continue
        drs[target] = ((1 / g.in_degree(target)) ** (1/3))
    nx.set_node_attributes(g, drs, "degRecip")
    
#Calculate manipulated relational-degrees for all edges in a graph, and
#  assign them as edge attributes for the Weighted Cascade 2 model
    
#Log-scaling method
def assignRelDegs1(g):
    rds = {}
    for target in g:
        if not g.in_degree(target):
            continue
        snd = 0
        for targeting in g.predecessors(target):
            snd += g.out_degree(targeting)
        for targeting in g.predecessors(target):
            rds[(targeting, target)] = log((g.out_degree(targeting) / snd))
    #elMax = rds[max(rds, key=rds.get)]
    #elMin = rds[min(rds, key=rds.get)]
    rds = mmNormalizeDict(rds, max(rds.values()), min(rds.values()))
    nx.set_edge_attributes(g, rds, "relDeg")
    
#Square-rooting method
def assignRelDegs2(g):
    rds = {}
    for target in g:
        if not g.in_degree(target):
            continue
        snd = sum([(g.out_degree(neighbour)) 
                   for neighbour in g.predecessors(target)])
        for targeting in g.predecessors(target):
            rds[(targeting, target)] = (((g.out_degree(targeting)) / snd) ** (1/2))
    nx.set_edge_attributes(g, rds, "relDeg")

#Cube-rooting method
def assignRelDegs3(g):
    rds = {}
    for target in g:
        if not g.in_degree(target):
            continue
        snd = sum([(g.out_degree(neighbour)) 
                   for neighbour in g.predecessors(target)])
        for targeting in g.predecessors(target):
            rds[(targeting, target)] = (((g.out_degree(targeting)) / snd) ** (1/3))
    nx.set_edge_attributes(g, rds, "relDeg")

#Assign method dictionary for selection depending on parameters
assignMods = {'WC1': {1: assignRecips1, 2: assignRecips2, 3: assignRecips3}, 
              'WC2': {1: assignRelDegs1, 2: assignRelDegs2, 3:assignRelDegs3}}

#Selects and runs appropriate assigning method
def assignSelect(g, propMod, assignMod):
    if assignMod:
        assignMods[propMod][assignMod](g)


# In[15]:


#Manual method to assign relational degree for WC2 to edges

#Return list of relational degrees
def allRelDegs(g):
    #allRds = []
    allRds, allRdsDict = [], {}
    for target in g:
        if not g.in_degree(target):
            continue
        snd = sum([g.out_degree(neighbour) 
                   for neighbour in g.predecessors(target)])
        for targeting in g.predecessors(target):
            rdval = log(g[targeting][target]['trust']*(g.out_degree(targeting) / snd))
            allRds.append(rdval)
            allRdsDict[(targeting, target)] = log((g.out_degree(targeting) / snd))
    return allRds

#List of relational degrees, maximum & minimums are assigned
relDegsTest1 = allRelDegs(graphs['Facebook'])
elMax, elMin = max(relDegsTest1), min(relDegsTest1)

#Normalize a single element, given the max and min elements
def mmNormalizeSingle(val):
    return ((val - elMin)/(elMax - elMin))


# In[16]:


#  MISCALLANEOUS & UTILITY METHODS/FUNCTIONS ::
#
#  Methods & functions for various purposes, that are either required
#  in other sections of the program or optimize their performance.
#
#  Methods/Functions included:
#  1. Measure time/speed of a given function
#  2. Min-max normalize a given dictionary, scaling between 0 and 1.
#  3. Draw a histogram from a given dictionary of probabilities


# In[17]:


#Normalize (Min-Max) every value in a given dictionary
def mmNormalizeDict(dic, elMax, elMin):
    for key, value in dic.items():
        dic[key] = ((value - elMin) / (elMax - elMin))
    return dic


# In[18]:


#Time measuring functions

#Measure the time taken to perform a given function
def measureTime1(func, *pars):
    startT = time()
    print(func(*pars))
    print(str(round((time() - startT), 3)) + "\n")
    
#MeasureTime1 with no printing of function
def measureTime1NoPrint(func, *pars):
    startT = time()
    func(*pars)
    print(str(round((time() - startT), 3)) + " secs\n")

#Same as measureTime, but also returns the result from the given function
def measureTimeRet(func, *pars):
    startT = time()
    return func(*pars), round((time() - startT), 3)
    
#Same as measureTime1, but prints a given message initially
def measureTime2(msg, func, *pars):
    print(msg + ":")
    startT = time()
    print(func(*pars))
    print(str(round((time() - startT), 3)) + " secs\n")

#Given a seed selection model, and can also take parameters for that, selects a seed set and 
#  measures the time taken to do so. Checks this seed set hasn't already been propagated to, 
#  and if it hasnt't performs a given propagation model on it and measures the time it took.
#Also returns the seed set, so that it can be added to the set of propagated-to seed sets.
def measureTime3(seedSel, propMod, oldSeeds, g, qty, its, *params):
    print(seedSel[1] + " Seed Selection:")
    startT = time()
    S = set(seedSel[0](g, qty, *params))
    print(str(S) + "\n" + str(time() - startT) + "\n")
    found = False
    for oldSeedSet in oldSeeds:
        if S in oldSeedSet:
            found = True
            print("Same seed set as " + oldSeedSet[1] + 
                  ".\nNo need for propagation, check previous results.\n")
    if found:
        return S
    print(propMod[1] + ": (" + str(its) + " iterations)")
    startT = time()
    print(str(cascade(g, S, its, propMod[0])))
    print(str(time() - startT) + "\n\n")
    return S

#Same as measureTime3, but doesn't print strings and returns results
def measureTime3Ret(seedSel, propMod, vals, gname, g, qty, its, *params):
    #found = False
    startT = time()
    Seed = (set(seedSel[0](g, qty, *params)), round((time()-startT), 5))
    #endTime = time() - startT
    #if vals and findNestedDictVal(vals, S):
    #    found = True
    #vals[gname][seedSel[1]][params]['Seed'] = (S, endTime)
    #if found:
    #    return vals

def measureRetTup1(func, g, qty, *params):
    startT = time()
    return (set(seedSel[0](g, qty, *params)), round((time()-startT), 5))
    
def measureRetTup2(func, g, S, its):
    startT = time()
    return (cascade(g, S, its, propMod[0]), round((time()-startT), 5))

#Same as measureTime3 without the old seed checking
def measureTime4(seedSel, propMod, oldSeeds, g, qty, its, *params):
    print(seedSel[1] + " Seed Selection:")
    startT = time()
    S = set(seedSel[0](g, qty, *params))
    print(str(S) + "\n" + str(time() - startT) + "\n")
    print(propMod[1] + ": (" + str(its) + " iterations)")
    startT = time()
    print(str(cascade(g, S, its, propMod[0])))
    print(str(time() - startT) + "\n\n")


# In[19]:


#  SEED SELECTION MODEL PARAMETER FINE-TUNING ::
#
#  Seed selection functions with variable parameters, and
#  functions to compare the spreads of different parameter values.
#
#  Sections:
#  1. Seed selection models with variable parameters
#     -closeness centrality, -betweenness centrality, 
#     -approximate current flow betweenness centrality
#     -load centrality, -eigenvector centrality,
#     -katz centrality, -harmonic centrality, -page rank
#  2. Printing resultant spreads using different permutations of 
#     parameters to compare and choose the best.


# In[20]:


#Seed selection functions with variable parameters, for testing different
#  value parameters and comparing their resultant spread.

#Closeness-centrality w/ variable parameters
def ccSeedsTune(g, qty, wf, dis=None):
    ccs = nx.closeness_centrality(g, distance=dis, wf_improved=wf)
    return sorted(ccs, key=ccs.get, reverse=True)[:qty]

#Betweenness-centrality w/ variable parameters
def btwnCSeedsTune(g, qty, kp, s, w=None):
    btwns = nx.betweenness_centrality(g, k=kp, normalized=False, weight=w, seed=s)
    return sorted(btwns, key=btwns.get, reverse=True)[:qty]

#Approximate current-flow Betweenness-centrality w/ variable parameters
def approxCfBtwnCSeedsTune(g, qty, p):
    #A negative p indicates that k should be the graph's size
    #  divided by p, not p itself
    if p < 0:
        k = g.size()/abs(p)
    else:
        k = p
    acfBtwns = nx.approximate_current_flow_betweenness_centrality(nx.to_undirected(g), normalized=False, kmax=k, seed=10)
    return sorted(acfBtwns, key=acfBtwns.get, reverse=True)[:qty]

#Load-centrality w/ variable parameters
def loadCSeedsTune(g, qty, cut, w=None):
    lcs = nx.load_centrality(g, normalized=False, weight=w, cutoff=cut)
    return sorted(lcs, key=lcs.get, reverse=True)[:qty]

#Eigenvector-centrality w/ variable parameters
def evCSeedsTune(g, qty, mi, t, w=None):
    evcs = nx.eigenvector_centrality(g, weight=w, tol=t, max_iter=mi)
    return sorted(evcs, key=evcs.get, reverse=True)[:qty]

#Katz-centrality w/ variable parameters
def kCSeedsTune(g, qty, b, a, w=None):
    kcs = nx.katz_centrality_numpy(g, alpha=a, beta=b, normalized=False, weight=w)
    return sorted(kcs, key=kcs.get, reverse=True)[:qty]

#Harmonic-centrality w/ variable parameters
def harmCSeedsTune(g, qty, p=None):
    hcs = nx.harmonic_centrality(g, distance=p)
    return sorted(hcs, key=hcs.get, reverse=True)[:qty]

#PageRank w/ variable parameters
def pageRankSeedsTune(g, qty, al, w=None):
    prs = nx.pagerank(g, alpha=al, weight=w)
    return sorted(prs, key=prs.get, reverse=True)[:qty]


# In[21]:


#Seed selection model parameter fine-tuning

#Runs seed selection models that can take different parameter values, with
#  every possible permutation of values and prints the results of each.
def paramFineTune(gs, qty, its, sModsParams):
    for g in graphs:
        print(g + ":\n")
        #Set of tried seed sets is kept, to avoid unneccesary repeated propagations.
        oldSeeds = set()
        for seedSel in sModsTune:
            #Generates tuples for every possible permutation from the given tuple of variables
            #Special case for single parameters, as they need to be within an iterable
            singleParam = False
            if len(seedSel[2]) > 1:
                params = list(product(*seedSel[2]))
            else:
                params = list(*seedSel[2])
                singleParam = True
            for paramPerm in params:
                print(paramPerm)
                if singleParam:
                    paramPerm = [paramPerm]
                try:
                    currentSeeds = measureTime3(seedSel, propMods[0], 
                                                oldSeeds, graphs[g], 
                                                qty, its, *paramPerm)
                    #Seeds obtained are added as frozen set to a set of tried seeds, 
                    #  to avoid propagating the same seed set twice.
                    oldSeeds.add((frozenset(currentSeeds), 
                                 (seedSel[1] + ": " + str(paramPerm))))
                except Exception as e:
                    print(e)


# In[22]:


#Seed selection models to be fine-tuned, with tuples 
#  for each parameter containing every value to be tried.
sModsTune = [(btwnCSeedsTune, "BetweennessCentrality", ((25,50,100,200), (10, None))), 
             (approxCfBtwnCSeedsTune, "ApproxCF-BetweennessCentrality", [(10000,500,-50,-200)]), 
             (loadCSeedsTune, "LoadCentrality", [(1,2,3,4)]), 
             (evCSeedsTune, "EigenvectorCentrality", ((100,500,1000), (0.001,0.0025,0.005))), 
             (kCSeedsTune, "KatzCentrality", ((0.75,1,1.25,1.5), (0.05,0.1,0.15,0.2))),
             (harmCSeedsTune, "HarmonicCentrality", ([(None, 'trust')])),
             (pageRankSeedsTune, "PageRank", [(0.65,0.75,0.85,0.95)]), 
             (ccSeedsTune, "ClosenessCentrality", [(True, False)])]
#Parameter fine-tuning
"""
paramFineTune(graphs, 8, 25, sModsTune)
#"""
print("")


# In[23]:


#Text output from above function:
"""
BitcoinOTC:

(25, 10)
BetweennessCentrality Seed Selection:
{0, 34, 6, 904, 2027, 4558, 2641, 1809}
0.9559996128082275

Independent Cascade: (25 iterations)
579.48
3.3419723510742188


(25, None)
BetweennessCentrality Seed Selection:
{34, 6, 1351, 904, 2124, 1809, 2387, 3128}
0.9119899272918701

Independent Cascade: (25 iterations)
543.6
2.987001419067383


(50, 10)
BetweennessCentrality Seed Selection:
{0, 34, 6, 904, 4171, 2027, 2641, 1809}
1.7610361576080322

Independent Cascade: (25 iterations)
579.08
2.9539971351623535


(50, None)
BetweennessCentrality Seed Selection:
{0, 34, 6, 904, 12, 2641, 1809, 3734}
1.7709946632385254

Independent Cascade: (25 iterations)
578.68
3.026970386505127


(100, 10)
BetweennessCentrality Seed Selection:
{0, 34, 6, 904, 4171, 2027, 2641, 1809}
3.460965156555176

Same seed set as BetweennessCentrality: (50, 10).
No need for propagation, check previous results.

(100, None)
BetweennessCentrality Seed Selection:
{0, 34, 6, 904, 4171, 2027, 2641, 1809}
3.956001043319702

Same seed set as BetweennessCentrality: (50, 10).
No need for propagation, check previous results.

Same seed set as BetweennessCentrality: (100, 10).
No need for propagation, check previous results.

(200, 10)
BetweennessCentrality Seed Selection:
{0, 34, 6, 904, 4171, 2027, 1809, 2641}
7.684998512268066

Same seed set as BetweennessCentrality: (50, 10).
No need for propagation, check previous results.

Same seed set as BetweennessCentrality: (100, None).
No need for propagation, check previous results.

Same seed set as BetweennessCentrality: (100, 10).
No need for propagation, check previous results.

(200, None)
BetweennessCentrality Seed Selection:
{0, 34, 6, 904, 4171, 2124, 2641, 1809}
8.950001239776611

Independent Cascade: (25 iterations)
579.64
2.987016439437866


10000
ApproxCF-BetweennessCentrality Seed Selection:
{1952, 34, 904, 4171, 2124, 2027, 1809, 5851}
15.70596981048584

Independent Cascade: (25 iterations)
563.44
3.250000238418579


500
ApproxCF-BetweennessCentrality Seed Selection:
{1952, 34, 904, 4171, 2124, 2027, 1809, 5851}
14.256999969482422

Same seed set as ApproxCF-BetweennessCentrality: [10000].
No need for propagation, check previous results.

-50
ApproxCF-BetweennessCentrality Seed Selection:
{1952, 34, 904, 4171, 2124, 2027, 1809, 5851}
15.11800241470337

Same seed set as ApproxCF-BetweennessCentrality: [10000].
No need for propagation, check previous results.

Same seed set as ApproxCF-BetweennessCentrality: [500].
No need for propagation, check previous results.

-200
ApproxCF-BetweennessCentrality Seed Selection:
{1952, 34, 904, 4171, 2124, 2027, 1809, 5851}
14.81799840927124

Same seed set as ApproxCF-BetweennessCentrality: [10000].
No need for propagation, check previous results.

Same seed set as ApproxCF-BetweennessCentrality: [-50].
No need for propagation, check previous results.

Same seed set as ApproxCF-BetweennessCentrality: [500].
No need for propagation, check previous results.

1
LoadCentrality Seed Selection:
{0, 1, 2, 3, 4, 5, 14, 15}
0.08399629592895508

Independent Cascade: (25 iterations)
494.44
2.95900297164917


2
LoadCentrality Seed Selection:
{34, 6, 904, 2027, 2124, 4171, 2641, 1809}
5.055690288543701

Independent Cascade: (25 iterations)
572.0
3.5680010318756104


3
LoadCentrality Seed Selection:
{0, 34, 6, 904, 2027, 2124, 2641, 1809}
52.553258419036865

Independent Cascade: (25 iterations)
587.44
3.2359983921051025
#"""
print("")


# In[24]:


#  SEED SELECTION MODELS (post parameter fine-tuning)::
#
#  Functions for selecting a seed set from a given graph, using
#  various different strategies. Split into 4 sections.
#
#  Sections:
#  1. Random seed selection model (for comparison)
#  2. Models from prior papers
#     -random, -original greedy, -CELF,
#     -improved greedy, degree discount
#  3. Models based on network analysis metrics
#     -degree centrality, -out_degree centrality, -closeness centrality, 
#     -information centrality, -betweenness centrality,
#     -approximate current flow betweenness centrality,
#     -load centrality, -eigenvector centrality, -katz centrality
#     -subgraph centrality, -harmonic centrality,
#     -vote rank, -page rank, -HITS hubs, -HITS authorities
#  4. New models
#     -Mixed greedy 1.1, 1.2, 2.1 & 2.2
#     -customHeuristic
#     -disconnect (DOESN'T WORK)


# In[ ]:


#Random seed selection model
def randomSeeds(g, qty, *other):
    return set(np.random.choice(g, qty, replace=False))
randomTuple = (randomSeeds, 'random')


# In[ ]:


#Seed Selection Models from past research

#Original Greedy from Kempe et al 2003
#Calculates spread for every node not in the seed set, and adds the 
#  highest to the seed set. Repeat until seed set is full.
def ogGreedySeeds(g, qty, its, propFunc='IC'):
    S = set()
    for _ in range(qty):
        inf = {node: cascade(g, S.union({node}), its, model=propFunc) 
               for node in g if node not in S}
        S.add(max(inf, key=inf.get))
    return S

#Cost-effective Lazy-forward (CELF) from Leskovec 2007
#Uses submodularity of propagation to optimize - spread of every node
#  doesn't need to be calculated every time.
#Calculates the spread of every node and creates a sorted list, 
#  and extracts the highest to seed set. Then the new highest is 
#  recalculated and if it remains the highest is added to the
#  seed set, otherwise the list is resorted.
def celfSeeds(g, qty, its, propFunc='IC'):
    infs = sorted([(node, cascade(g, {node}, its, model=propFunc)) 
                   for node in g], key=itemgetter(1), reverse=True)
    S = {infs[0][0]}
    spread = infs[0][1]
    infs = infs[1:]
    for _ in range(qty-1):
        sameTop = False
        while not sameTop:
            check = infs[0][0]
            infs[0] = (check, cascade(g, S.union({check}), its, 
                                      model=propFunc)- spread)
            infs = sorted(infs, key=itemgetter(1), reverse=True)
            sameTop = (infs[0][0] == check)
        S.add(infs[0][0])
        spread += infs[0][1]
        infs = infs[1:]
    return S

#Improved Greedy from Chen 2009
#Creates a simulated copy of the graph, removing edges with the probability
#  (1 - pp). Then the reach within that graph is calculated for each node,
#  and the highest is added to the seed set.
def impGreedySeeds(g, qty, its):
    S = set()
    for _ in range(qty):
        for i in range(its):
            np.random.seed(i)
            remove = [(u, v) for (u, v, t) in g.edges.data('trust') 
                      if np.random.uniform(0, 1) > (pp*t)]
            newG = deepcopy(g)
            newG.remove_edges_from(remove)
            rSnewG = reachable(newG, S)
            infs = [(node, len(reachable(newG, {node}))) for node in newG 
                    if node not in rSnewG]
        #infs = sorted([(node, (val/its)) for (node, val) in infs], 
        #              key=lambda x:x[1], reverse=True)
        #S.add(infs[0][0])
        infs = [(node, val/its) for (node, val) in infs]
        S.add(max(infs, key=itemgetter(1))[0])
    return S

#Degree Discount from Chen 2009
#Calculates degree 'score' of each node, adds highest to seed set,
#  discounts score of every neighbour node of newly added seed, 
#  and repeats until seed set is filled.
def degDiscSeeds(g, qty, *other):
    S = set()
    nodes = {}
    for node in g:
        deg = g.degree(node)
        nodes[node] = (deg, 0, deg)
    for _ in range(qty):
        ddvmax = 0
        for node in nodes:
            if node not in S:
                if nodes[node][2] > ddvmax:
                    ddvmax = nodes[node][2]
                    u = node
        S.add(u)
        for neighbour in g.neighbors(u):
            if neighbour not in S:
                dv, tv, ddv = nodes[neighbour]
                tv += 1
                ddv = dv - (2*tv) - ((dv - tv)*tv*pp)
                nodes[neighbour] = dv, tv, ddv
    return S

#seeds compiled into list
priorSeeds = [(ogGreedySeeds, 'ogGreedy'), (celfSeeds, 'celf'),
              (impGreedySeeds, 'impGreedy'), (degDiscSeeds,'degDisc'), 
              (randomSeeds, 'random')]


# In[ ]:


#Network-Analysis-metric-based Seed Selection Models

#Degree-centrality
def degCSeeds(g, qty):
    dcs = nx.degree_centrality(g)
    return sorted(dcs, key=dcs.get, reverse=True)[:qty]

#In-degree-centrality
def inDegCSeeds(g, qty):
    dcs = nx.in_degree_centrality(g)
    return sorted(dcs, key=dcs.get, reverse=True)[:qty]

#Out-degree-centrality
def outDegCSeeds(g, qty):
    dcs = nx.out_degree_centrality(g)
    return sorted(dcs, key=dcs.get, reverse=True)[:qty]

#Closeness-centrality
def ccSeeds(g, qty):
    ccs = nx.closeness_centrality(g, wf_improved=True)
    return sorted(ccs, key=ccs.get, reverse=True)[:qty]

#Information-centrality
#(a.k.a. current-flow closeness-centrality)
def infCSeeds(g, qty):
    ics = nx.information_centrality(nx.to_undirected(g))
    return sorted(ics, key=ics.get, reverse=True)[:qty]

#Betweenness-centrality
def btwnCSeeds(g, qty):
    btwns = nx.betweenness_centrality(g, k=50, normalized=False, seed=10)
    return sorted(btwns, key=btwns.get, reverse=True)[:qty]

#Approximate current-flow betweenness-centrality
def approxCfBtwnCSeeds(g, qty):
    acfBtwns = nx.approximate_current_flow_betweenness_centrality(nx.to_undirected(g), normalized=False, kmax=200, seed=10)
    return sorted(acfBtwns, key=acfBtwns.get, reverse=True)[:qty]

#Load-centrality
def loadCSeeds(g, qty):
    lcs = nx.load_centrality(g, normalized=False, cutoff=2)
    return sorted(lcs, key=lcs.get, reverse=True)[:qty]

#Eigenvector-centrality
def evCSeeds(g, qty):
    evcs = nx.eigenvector_centrality(g, tol=0.005)
    return sorted(evcs, key=evcs.get, reverse=True)[:qty]

#Katz-centrality
def kCSeeds(g, qty):
    kcs = nx.katz_centrality_numpy(g, alpha=0.05, normalized=False)
    return sorted(kcs, key=kcs.get, reverse=True)[:qty]

#Subgraph-centrality
def subgCSeeds(g, qty):
    sgcs = nx.subgraph_centrality(nx.to_undirected(g))
    return sorted(sgcs, key=sgcs.get, reverse=True)[:qty]

#Harmonic-centrality
def harmCSeeds(g, qty):
    hcs = nx.harmonic_centrality(g, distance='distance')
    return sorted(hcs, key=hcs.get, reverse=True)[:qty]

#VoteRank
def voteRankSeeds(g, qty):
    return set(nx.voterank(nx.to_undirected(g), qty))

#PageRank
def pageRankSeeds(g, qty):
    prs = nx.pagerank(g, alpha=0.95)
    return sorted(prs, key=prs.get, reverse=True)[:qty]

#HITS Hubs
def hitsHubSeeds(g, qty):
    hhs, has = nx.hits(g)
    return sorted(hhs, key=hhs.get, reverse=True)[:qty]

#HITS Authorities
def hitsAuthSeeds(g, qty):
    hhs, has = nx.hits(g)
    return sorted(has, key=has.get, reverse=True)[:qty]

#NetworkX seeds compiled into list (w/ random)
netSeeds = [(degCSeeds, 'degC'), (inDegCSeeds, 'inDeg'), 
            (outDegCSeeds, 'outDeg'), (ccSeeds, 'closeC'), 
            (infCSeeds, 'info'), (btwnCSeeds, 'btwnC'), 
            (approxCfBtwnCSeeds, 'approxCfBtwnC'), (loadCSeeds, 'loadC'), 
            (subgCSeeds, 'subG'), (harmCSeeds, 'harmC'), 
            (voteRankSeeds, 'voteRank'), (pageRankSeeds, 'pageRank'), 
            (hitsHubSeeds, 'Hubs'), (hitsAuthSeeds, 'Auth'), 
            (randomSeeds, 'random')]


# In[ ]:


#Mixed greedy seed selection models
def mixedGreedy11(g, qty, its):
    S = set()
    for i in range(its):
        np.random.seed(i)
        remove = [(u, v) for (u, v, t) in g.edges.data('trust') 
                  if np.random.uniform(0, 1) > (pp*t)]
        newG = deepcopy(g)
        newG.remove_edges_from(remove)
        rSnewG = reachable(newG, S)
        infs = [(node, len(reachable(newG, {node}))) for node in newG 
                if node not in rSnewG]
    infs = sorted([(node, val/its) for (node, val) in infs], 
                  key=itemgetter(1), reverse=True)
    S.add(infs[0][0])
    reach, infs = infs[0][1], infs[1:]
    for _ in range(qty-1):
        rSnewG = reachable(newG, S)
        sameTop = False
        while not sameTop:
            check = infs[0][0]
            if check in rSnewG:
                infs = infs[1:]
                continue
            infs[0] = (check, len(reachable(newG, {check})) - reach)
            infs = sorted(infs, key=itemgetter(1), reverse=True)
            sameTop = (infs[0][0] == check)
        S.add(infs[0][0])
        reach += infs[0][1]
        infs = infs[1:]
    return S

def mixedGreedy12(g, qty, its):
    S = set()
    for i in range(its):
        np.random.seed(i)
        remove = [(u, v) for (u, v, t) in g.edges.data('trust') 
                  if np.random.uniform(0, 1) > (pp*t)]
        newG = deepcopy(g)
        newG.remove_edges_from(remove)
        rSnewG = reachable(newG, S)
        infs = [(node, len(reachable(newG, {node}))) for node in newG 
                if node not in rSnewG]
    infs = sorted([(node, val/its) for (node, val) in infs], 
                  key=itemgetter(1), reverse=True)
    S.add(infs[0][0])
    reach, infs = infs[0][1], infs[1:]
    firstrun = 1
    for _ in range(qty-1):
        if firstrun:
            rSnewG = reachable(newG, S)
            firstrun = 0
        else:
            rSnewG = reachable(newG, {Snew})
        newG.remove_nodes_from(rSnewG)
        sameTop = False
        while not sameTop:
            check = infs[0][0]
            if check in rSnewG or check not in newG:
                infs = infs[1:]
                continue
            infs[0] = (check, len(reachable(newG, {check})) - reach)
            infs = sorted(infs, key=itemgetter(1), reverse=True)
            sameTop = (infs[0][0] == check)
        Snew = infs[0][0]
        S.add(Snew)
        reach += infs[0][1]
        infs = infs[1:]
    return S

def mixedGreedy21(g, qty, its):
    S = set()
    edges, edgeCount = [], []
    for i in range(its):
        np.random.seed(i)
        newG = deepcopy(g)
        newG.remove_edges_from([(u,v) for (u,v,t) in g.edges.data('trust')
                              if np.random.uniform(0,1) > (pp*t)])
        newEdges = [e for e in newG.edges]
        edgeCount.append(len(newEdges))
        edges += newEdges
    counts = Counter(edges)
    finalEdges = (sorted(counts, key=counts.get, reverse=True))[:int(np.mean(edgeCount))]
    newGfinal = nx.DiGraph(finalEdges)
    rSnewG = reachable(newG, S)
    infs = sorted([(node, len(reachable(newG, {node}))) for node in newGfinal 
                   if node not in rSnewG], key=itemgetter(1), reverse=True)
    S.add(infs[0][0])
    reach, infs = infs[0][1], infs[1:]
    for _ in range(qty-1):
        rSnewG = reachable(newGfinal, S)
        sameTop = False
        while not sameTop:
            check = infs[0][0]
            if check in rSnewG:
                infs = infs[1:]
                continue
            infs[0] = (check, len(reachable(newGfinal, {check})) - reach)
            infs = sorted(infs, key=itemgetter(1), reverse=True)
            sameTop = (infs[0][0] == check)
        S.add(infs[0][0])
        reach += infs[0][1]
        infs = infs[1:]
    return S

def mixedGreedy22(g, qty, its):
    S = set()
    edges, edgeCount = [], []
    for i in range(its):
        np.random.seed(i)
        newG = deepcopy(g)
        newG.remove_edges_from([(u,v) for (u,v,t) in g.edges.data('trust')
                              if np.random.uniform(0,1) > (pp*t)])
        newEdges = [e for e in newG.edges]
        edgeCount.append(len(newEdges))
        edges += newEdges
    counts = Counter(edges)
    finalEdges = (sorted(counts, key=counts.get, reverse=True))[:int(np.mean(edgeCount))]
    newGfinal = nx.DiGraph(finalEdges)
    rSnewG = reachable(newG, S)
    infs = sorted([(node, len(reachable(newG, {node}))) for node in newGfinal 
                   if node not in rSnewG], key=itemgetter(1), reverse=True)
    S.add(infs[0][0])
    reach, infs = infs[0][1], infs[1:]
    firstrun = 1
    for _ in range(qty-1):
        if firstrun:
            rSnewG = reachable(newGfinal, S)
            firstrun = 0
        else:
            rSnewG = reachable(newGfinal, {Snew})
        newGfinal.remove_nodes_from(rSnewG)
        sameTop = False
        while not sameTop:
            check = infs[0][0]
            if check in rSnewG or check not in newGfinal:
                infs = infs[1:]
                continue
            infs[0] = (check, len(reachable(newGfinal, {check})) - reach)
            infs = sorted(infs, key=itemgetter(1), reverse=True)
            sameTop = (infs[0][0] == check)
        Snew = infs[0][0]
        S.add(Snew)
        reach += infs[0][1]
        infs = infs[1:]
    return S

#Custom Heuristic
def calculateRank(g, node, seeds):
    seedProb, nonSeedProb = 1, 1
    for neighbour in g.predecessors(node):
        if neighbour in seeds:
            seedProb *= (1 - (basicProb()*(g[neighbour][node]['trust'])))
    for neighbour in g.successors(node):
        if neighbour not in seeds:
            nonSeedProb += (basicProb()*(g[node][neighbour]['trust']))
    return seedProb * nonSeedProb

def customHeuristicSeeds(g, qty, *other):
    S = set()
    ranks = sorted({(node, calculateRank(g, node, S)) for node in g 
                    if node not in S}, key=lambda x:x[1], reverse=True)
    for _ in range(qty):
        topNode = ranks[0][0]
        S.add(topNode)
        ranks = ranks[1:]
        for neighbour in g.successors(topNode):
            for pos,(node,rank) in enumerate(ranks):
                if node == neighbour:
                    ranks[pos] = (node, calculateRank(g, node, S))
                    #changed.append(ranks[pos])
                    break
        ranks = sorted(ranks, key=lambda x:x[1], reverse=True)
    return S

#Disconnect Greedy
#Could not get it to work, so omitted from results
def disconnectSeeds(g, qty, its=500):
    S, infs = set(), {}
    for node in g:
        reached = cascade(g, {node}, its, ret=True)
        infs[node] = {'nodes': reached, 'inf': len(reached)}
        maxSeed = max(infs, key=lambda x:infs[x]['inf'])
        S.add(maxSeed)
    Gx = deepcopy(g)
    Gx.remove_nodes_from(infs[maxSeed]['nodes'])
    del infs[maxSeed]
    for _ in (range(qty - 1)):
        for node in Gx:
            newReach = cascade(Gx, {node}, its, ret=True)
            infs[node] = {'nodes':newReach, 'inf':len(newReach)}
        maxSeed = max(infs, key=lambda x:infs[x]['inf'])
        S.add(maxSeed)
        Gx.remove_nodes_from(infs[maxSeed]['nodes'])
        del infs[maxSeed]
    return S

ogSeeds = [(mixedGreedy11, 'mixedGreedy11'), 
           (mixedGreedy12, 'mixedGreedy12'), 
           (mixedGreedy21, 'mixedGreedy21'), 
           (mixedGreedy22, 'mixedGreedy22'),
           (customHeuristicSeeds, 'customHeuristic'), 
           (randomSeeds, 'random')]


# In[ ]:


allSeeds1 = [(ogGreedySeeds, 'ogGreedy'), 
             (celfSeeds, 'celf'),
             (impGreedySeeds, 'impGreedy'), 
             (degDiscSeeds,'degDisc'), 
             (inDegCSeeds, 'inDeg'), 
             (outDegCSeeds, 'outDeg'), 
             (ccSeeds, 'closeC'), 
             (infCSeeds, 'info'), 
             (btwnCSeeds, 'btwnC'), 
             (approxCfBtwnCSeeds, 'approxCfBtwnC'), 
             (loadCSeeds, 'loadC'), 
             (subgCSeeds, 'subG'), 
             (harmCSeeds, 'harmC'), 
             (voteRankSeeds, 'voteRank'), 
             (pageRankSeeds, 'pageRank'), 
             (hitsHubSeeds, 'Hubs'), 
             (hitsAuthSeeds, 'Auth'), 
             (mixedGreedy11, 'mixedGreedy11'), 
             (mixedGreedy12, 'mixedGreedy12'), 
             (mixedGreedy21, 'mixedGreedy21'), 
             (mixedGreedy22, 'mixedGreedy22'),
             (customHeuristicSeeds, 'customHeuristic'), 
             (randomSeeds, 'random')]
allSeeds2 = [(degDiscSeeds, 'degreeDiscount'), 
             (inDegCSeeds, 'inDegree'),
             (outDegCSeeds, 'outDegree'), 
             (mixedGreedy22, 'mixedGreedy22'),
             (customHeuristicSeeds, 'customHeuristic'), 
             (randomSeeds, 'randomSeeds')]


# In[ ]:


#  GRAPHING FUNCTIONS ::


# In[25]:


def horzBar(lis, vals, msg, topGap):
    #return nothing if lists aren't same size
    if len(lis[1]) != len(vals[1]) != len(vals[2]):
        print("Error, not the same size")
        return
    lis[1], vals[1], vals[2] = lis[1][::-1], vals[1][::-1], vals[2][::-1]
    #subplot set up, gridlines drawn, max value calculated and y-limits set
    fig, ax = plt.subplots(2, 1, figsize=(12,2*len(vals[1])))
    for g in range(2):
        ax[g].grid(zorder=0)
        height, pos = 0.4, np.arange(len(vals[1]))
        #bar chart plotted
        ax[g].barh(lis[1], vals[g+1], height=height,
                facecolor='lightsteelblue', edgecolor='black', 
                linewidth=2.5, zorder=3)
        #Subtitle, x-labels & y-labels are set for each axis
        ax[g].set_ylabel(lis[0], fontsize=20)
        ax[g].tick_params(axis='both', labelsize=15)
        ax[g].set_xlabel(vals[0][g], fontsize=20)
    #Titles are set and the layout (incl. padding/gaps) is set and adjusted
    fig.tight_layout(pad=5)
    fig.suptitle(msg + " Comparison:", fontsize=24, fontweight='bold')
    fig.subplots_adjust(top=topGap)


# In[26]:


#Prepares values for comparing seed selection models, and plots bar chart
def prepareBar(seedMods, gs, qty=4, its=100, its2=100, topGap=0.92, gqty=0, model='IC', timeFactor=0.01):
    if not gqty:
        gqty=len(gs)
    #Lists are intialized
    x = ['Models', seedMods]
    #Special cases where additional variables are required in 
    #  seed selection are noted
    fourParam = ['ogGreedy', 'celf']
    threeParam = ['impGreedy', 'mixedGreedy11', 'mixedGreedy12', 
                  'mixedGreedy21', 'mixedGreedy22']
    #For every graph in the given list,
    for gc, g in enumerate(gs):
        if gc+1 != gqty:
            continue
        y, seedTimes = [['Spread', 'Spread by Time'], [], []], []
        for c, seedSel in enumerate(x[1]):
            #Every seed selection model is run, seeds and the time 
            #  elapsed are added to a list
            if seedSel[1] in threeParam:
                t = time()
                seeds = seedSel[0](gs[g], qty, its2)
                t = time() - t
                if t < 0.001:
                    t = 0.001
                seedTimes.append(t)
            elif seedSel[1] in fourParam:
                t = time()
                seeds = seedSel[0](gs[g], qty, its2, model)
                t = time() - t
                if t < 0.001:
                    t = 0.001
                seedTimes.append(t)
            else:
                t = time()
                seeds = seedSel[0](gs[g], qty)
                t = time() - t
                if t < 0.001:
                    t = 0.001
                seedTimes.append(t)
            casc = cascade(gs[g], seeds, its)
            y[1].append(casc)
            if seedTimes[c] > 1:
                y[2].append(casc/(seedTimes[c]))
            else:
                y[2].append(casc)
        #Seed selection labels are compiled and bar chart is plotted
        xLabels = ['Models', [seedMod[1] for seedMod in x[1]]]
        horzBar(xLabels, y, (g + " Seed Select Models:"), topGap)


# In[75]:


#Past models - printing
print("Past models time testing (5 seeds)::\n")
for rndmGraph in rndmGraphs:
    print(rndmGraph +  ":\n")
    for seedSel in priorSeeds:
        measureTime2(seedSel[1], seedSel[0], rndmGraphs[rndmGraph], 5)


# In[89]:


#Past models random graphs - bar charts for spread & spread/time
#"""
measureTime1NoPrint(prepareBar, priorSeeds, rndmGraphs, 5)
#"""


# In[91]:


#NetworkX models - printing
print("Past models time testing (5 seeds)::\n")
for rndmGraph in rndmGraphs:
    print(rndmGraph +  ":\n")
    for seedSel in netSeeds:
        measureTime2(seedSel[1], seedSel[0], rndmGraphs[rndmGraph], 5)


# In[95]:


#NetworkX models random graphs - bar charts for spread & spread/time
#"""
measureTime1NoPrint(prepareBar, netSeeds, rndmGraphs, 5, 100, 100, 0.96)
#"""


# In[30]:


#NetworkX models real graphs - bar charts for spread & spread/time
#"""
measureTime1NoPrint(prepareBar, netSeeds, graphs, 5, 1, 1, 0.96, 1)
#"""
print("")


# In[109]:


#New models - printing
print("New models time testing (5 seeds)::\n")
for rndmGraph in rndmGraphs:
    print(rndmGraph +  ":\n")
    for seedSel in ogSeeds:
        measureTime2(seedSel[1], seedSel[0], rndmGraphs[rndmGraph], 5, 500)


# In[113]:


#New models real graphs - printing
print("New models time testing (5 seeds)::\n")
for graph in graphs:
    print(graph +  ":\n")
    for seedSel in ogSeeds:
        measureTime2(seedSel[1], seedSel[0], graphs[graph], 5, 5)
    print("")
    break


# In[115]:


#New models random graphs - bar charts for spread & spread/time
#"""
measureTime1NoPrint(prepareBar, ogSeeds, rndmGraphs, 5, 100)
#"""
print("")


# In[35]:


#New models real graphs - bar charts for spread & spread/time
#"""
measureTime1NoPrint(prepareBar, ogSeeds, graphs, 5, 5, 5, 0.92)
#"""
print("")


# In[134]:


#All models random graphs - bar charts for spread & spread/time
#"""
measureTime1NoPrint(prepareBar, allSeeds1, rndmGraphs, 5, 100, 100, 0.96)
#"""
print("")


# In[139]:


#All models random graphs - bar charts for spread & spread/time
#"""
measureTime1NoPrint(prepareBar, allSeeds2, graphs, 5, 50, 50, 0.92, 1)
#"""
print("")


# In[34]:


#All models random graphs - bar charts for spread & spread/time
#"""
measureTime1NoPrint(prepareBar, allSeeds2, graphs, 5, 50, 50, 0.92, 2)
#"""
print("")


# In[34]:


g, qty, its = rndmGraphs['mock3-random1'], 4, 250
testSeeds = [(randomSeeds, 'random'), 
             (degDiscSeeds, 'degreeDiscount'), 
             (degCSeeds, 'degreeCentrality'), 
             (customHeuristicSeeds, 'customHeuristics')]
def testRun(g, qty, seedMod, its):
    print("Seed selection model: " + seedMod[1])
    s, t = measureTimeRet(seedMod[0], g, qty)
    print("Seeds: " + str(s) + "\nTime taken: " + str(round(t, 3)))
    inf, t = measureTimeRet(cascade, g, s, its)
    print("Spread: " + str(inf) + "\nTime taken: " + str(round(t, 3)) + "\n")

for testSeed in testSeeds:
    testRun(g, qty, testSeed, its)


# In[ ]:




