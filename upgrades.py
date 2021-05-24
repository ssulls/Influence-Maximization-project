#!/usr/bin/env python
# coding: utf-8

# In[1]:


#  ALL NECESSARY IMPORTS ::


# In[3]:


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


# In[13]:


#Time measuring functions

#Measure the time taken to perform a given function
def measureTime1(func, *pars):
    startT = time()
    print(func(*pars))
    print(str(round((time() - startT), 3)) + " secs\n")

#Measures time, and returns the time unrounded
def measureTimeRet1(func, *pars):
    startT = time()
    return (time() - startT)

#Measures time, and returns the result and the time
def measureTimeRet2(func, *pars):
    startT = time()
    return func(*pars), round((time() - startT), 3)
    
#Same as measureTime1, but prints a given message initially
def measureTime2(msg, func, *pars):
    print(msg + ":")
    startT = time()
    print(func(*pars))
    print(str(time() - startT) + " secs\n")

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
            print(seedSel[1] + "has the same seed set as " + oldSeedSet[1] + 
                  ". No need for propagation, check previous results.\n")
    if found:
        return S
    print(propMod[1] + ": (" + str(its) + " iterations)")
    startT = time()
    print(str(cascade(g, S, its, propMod[0])))
    print(str(time() - startT) + "\n\n")
    return S

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


# In[9]:


#INFLUENCE MODEL #1

def IndependentCascade1(g, s, its, pp):
    #Graph, Seed set, Iterations, Propagation probability
    #Time measured and overall spread list initalized
    startTime = time()
    spread = []
    #every iteration...
    for it in range(its):
        #randomness seeded for consistency
        np.random.seed(it)
        influenced, tried = [], []
        #new nodes set to seed nodes
        newlyInf = s
        #while nodes were influenced this turn...
        #(stops when propagation cannot continue)
        while newlyInf:
            #targets are compiled into a list from newly influenced nodes
            targets = []
            for node in newlyInf:
                for neighbour in g.neighbors(node):
                    if neighbour not in influenced:
                        targets.append(neighbour)
            lastTurn = newlyInf
            newlyInf = []
            #targets are influenced depending on pp
            for target in targets:
                tried.append(target)
                if np.random.random() < pp:
                    newlyInf.append(target)
            #newly influenced nodes added to overall list
            influenced.append(newlyInf)
        #total number of influenced nodes added to list
        spread.append(len(influenced))
    #mean of all iterations returned, with time taken
    return np.mean(spread), (str(round((time()-startTime), 4)) + " secs")


# In[22]:


#propagation probability testing
print(IndependentCascade1(G3, [1,2], 100, 0.1))
print(IndependentCascade1(G3, [1,2], 100, 0.8)) 


# In[11]:


#INFLUENCE MODEL #2

#Success functions #1
def successVars2(sign, qf):
    q = qf
    #Modify quality factor for negative influence
    if not sign:
        q = (1-q)
    return q

def successIC(sign, g, target, targeting, pp, qf):
    return np.random.uniform(0,1) < (pp*successVars2(sign, qf)*g[targeting][target]['trust'])

def successWC1(sign, g, target, targeting, pp, qf):
    recip = 1 / g.in_degree(target)
    return np.random.uniform(0,1) < (recip*successVars2(sign, qf)*g[targeting][target]['trust'])
    
def successWC2(sign, g, target, targeting, pp, qf):
    snd = 0
    for neighbour in g.predecessors(target):
        snd += 1
    reldeg = g.out_degree(targeting) / snd
    return np.random.uniform(0,1) < (reldeg*successVars2(sign, qf)*g[targeting][target]['trust'])

def propagation2(g, posNew, negNew, tried, successMod, pp, qf):
    posCurrent, negCurrent = set(), set()
    for node in negNew:
        for neighbour in g.neighbors(node):
            if (node, neighbour) not in tried:
                #Negative influence to neighbours of negative nodes
                if successMod(False, g, neighbour, node, pp, qf):
                    negCurrent.add(neighbour)
                tried.add((node, neighbour))
    for node in posNew:
        for neighbour in g.neighbors(node):
            if (node, neighbour) not in tried:
                #Positive influence to neighbours of positive nodes
                if neighbour not in negCurrent and successMod(True, g, neighbour, node, pp, qf):
                    posCurrent.add(neighbour)
                #Negative influence to neighbours of positive nodes
                elif neighbour not in negCurrent and neighbour not in posCurrent and successMod(False, g, neighbour, node, pp, qf):
                    negCurrent.add(neighbour)
                tried.add((node, neighbour))
    return(posCurrent, negCurrent, tried)

#Cascade Model #2
def iteration2(g, s, its, pp=0.2, qf=1, model='IC'):
    #Graph, Seed set, Iterations, Propagation probability
    if model == 'WC1':
        successFunc = successWC1
    elif model == 'WC2':
        successFunc = successWC2
    else:
        successFunc = successIC
    #Time measured and overall spread list initalized
    startTime = time()
    spread = []
    #every iteration...
    for it in range(its):
        #randomness seeded for consistency
        np.random.seed(it)
        #Sets initialised for influenced/newly influenced nodes and tried edges
        positive, posNew, negative, negNew, tried = set(), set(s), set(), set(), set()
        #while nodes were influenced this turn...
        #(stops when propagation cannot continue)
        while posNew or negNew:
            #placeholder variables for new nodes
            posLastTurn, negLastTurn = posNew, negNew
            #propagation function is called
            posNew, negNew, tried = propagation2(g, posNew, negNew, tried, successFunc, pp, qf)
            #newly influenced nodes added to overall lists
            positive = (positive.union(posNew, posLastTurn) - negNew)
            negative = (negative.union(negNew, negLastTurn) - posNew)
        #total number of influenced nodes added to list
        spread.append(len(positive))
    #mean of all iterations returned, with time taken
    return np.mean(spread), (str(round((time()-startTime), 4)) + " secs")


# In[16]:


#Optimization testing for influence model method 1 -> 2
for infFunc in [IndependentCascade1, iteration2]:
    measureTime1(infFunc, G3, [1,2], 500, 0.5)


# In[47]:


#quality factor testing for every model, for every real graph
for model in ['IC', 'WC1', 'WC2']:
    for gc, g in enumerate([G1, G2]):
        for pp in [0.2]:
            for qf in [0.2, 0.8]:
                print("G" + str(gc+1) + ":\n" + model + " Vars Testing\nPP = " 
                      + str(pp) + ". QF = " + str(qf) + "\n" + 
                      str(iteration2(g, [1], 50, pp, qf, model)) + "\n")


# In[42]:


#FINAL INFLUENCE MODEL

#Determine propagation success for the various models
#(includes quality factor to differentiate positive/negative influence)
#(includes a switch penalty for nodes switching sign)

#Apply quality factor and switch factor variables
def successVars(sign, switch, qf, sf):
    if not switch:
        sf = 0
    if not sign:
        qf = (1-qf)
    return qf*(1-sf)

#Calculate whether propagation is successful (model-specific)
def success(successModel, sign, switch, timeDelay, g, target, targeting, pp, qf, sf, a):
    if successModel == 'ICu':
        succ = (pp*successVars(sign, switch, qf, sf)*timeDelay)
    elif successModel == 'IC':
        succ = (pp*successVars(sign, switch, qf, sf)*g[targeting][target]['trust']*timeDelay)
    elif successModel == 'WC1':
        if a:
            recip = g.nodes[target]['degRecip']
        else:
            recip = (1 / g.in_degree(target))
        succ = (recip*successVars(sign, switch, qf, sf)*timeDelay*g[targeting][target]['trust'])
    elif successModel == 'WC2':
        if a:
            relDeg = g[targeting][target]['relDeg']
        else:
            snd = sum([(g.out_degree(neighbour)) for neighbour in g.predecessors(target)])
            relDeg = (g.out_degree(targeting) / snd)
            #relDeg = mmNormalizeSingle(log(g.out_degree(targeting)/snd))
        succ = (relDeg*successVars(sign, switch, qf, sf)*timeDelay*g[targeting][target]['trust'])
    return np.random.uniform(0,1) < succ

#Returns probability with only the variables
#(no trust values, degree reciprocals or relational degrees)
def basicProb(weighted=False, *nodes):
    return pp * successVars(True, False)

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
                trv.add((node, neighbour))
    return(posCurrent, negCurrent, trv)

#Calculate average positive spread over a given number of iterations
def iterate(g, s, its, successFunc, pp, qf, sf, tf, retNodes, a):
    #If no number of iterations is given, one is calculated based on the
    #  ratio of nodes to edges within the graph, capped at 2000.
    if not its:
        neRatio = (len(g)/(g.size()))
        if neRatio > 0.555:
            its = 2000
        else:
            its = ((neRatio/0.165)**(1/1.75))*1000
    influence = []
    for i in range(its):
        #Randomness seeded per iteration for repeatability & robustness
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
            if timeFactor < 0:
                timeFactor = 0
            else:
                timeFactor -= tf
        if retNodes:
            #Positive nodes added to list
            for p in positive:
                influence.append(p)
            #Number of nodes added to list
            infCount.append(len(positive))
        else:
            #Number of positive nodes added to list
            influence.append(len(positive))
    #If nodes are being returned
    if retNodes:
        #Average list of positive nodes are returned
        counts = Counter(influence)
        result = (sorted(counts, key=counts.get, reverse=True))[:int(np.mean(infCount))]
    #If nodes aren't being returned
    else:
        #Mean is returned
        result = np.mean(influence)
    return result

#Determine the cascade model and run the iteration function 
#  with the appropriate success function
def cascade(g, s, its=0, 
            model='IC', assign=1, ret=False, 
            pp=0.2, qf=0.6, sf=0.7, tf=0.04):
    #g = graph, s = set of seed nodes, its = num of iterations
    #model = cascade model, #assign model, #return nodes?
    #pp = propagation probability, qf = quality factor
    #sf = switch factor, tf = time factor
    #Model is determined and appropriate success function is assigned
    #print(f'model = {model},  assign = {assign}  its = {its}\npp = {pp}, qf = {qf}, sf = {sf}, tf = {tf} \n')
    if model != 'IC' and model != 'ICu' and assign:
        assignSelect(g, model, assign)
    success = model
    return iterate(g, s, its, success, pp, qf, sf, tf, ret, assign)

#Propagation models and their names are compiled into a list
propMods = [('IC', "Independent Cascade"), 
            ('WC1', "Weighted Cascade 1"), 
            ('WC2', "Weighted Cascade 2")]

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
            rds[(targeting, target)] = log(g.out_degree(targeting) / snd)
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
        snd = 0
        for targeting in g.predecessors(target):
            snd += g.out_degree(targeting)
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

#Normalize (Min-Max) every value in a given dictionary (method 2 & 3)
def mmNormalizeDict(dic, elMax, elMin):
    #for key, value in dic.items():
    #    dic[key] = ((value - elMin) / (elMax - elMin))
    #printResults("Assigned", dic.values())
    #print("Assigned normalization:\nMax = " + str(elMax) + "\nMin = " 
    #      + str(elMin) + "\nMean = " 
    #      + str(np.mean(list(dic.values()))))
    #return dic
    return {key: ((val - elMin)/(elMax - elMin)) for key,val in dic.items()}


# In[41]:


#switch factor testing for every model in BitcoinOTC graph
g = graphs['BitcoinOTC']
for model in propMods:
    print(model[1] + ":\n")
    for test in [0, 0.3, 0.6, 0.9]:
        measureTime2("Switch factor: " + str(test), cascade, g, 
                     [1], 50, model[0], 1, False, 0.2, 0.6, test)
    print("")


# In[44]:


#switch factor testing for every model in BitcoinOTC graph
g = graphs['BitcoinOTC']
for model in propMods:
    print(model[1] + ":\n")
    for test in [0, 0.05, 0.1, 0.5]:
        measureTime2("Time factor: " + str(test), cascade, g, 
                     [1], 50, model[0], 1, False, 0.2, 0.6, 0.5, test)
    print("")


# In[22]:


#Printing Methods

#Method 1 - seperate print calls
#"""
def printResults1(msg, lis):
    print(msg)
    print("Mean = " + str(round(np.mean(lis), 5)))
    print("Median = " + str(round(np.median(lis), 5)))
    print("Max = " + str(round(max(lis), 5)))
    print("Min = " + str(round(min(lis), 5)))
    print("Range = " + str(round((max(lis)-min(lis)), 5)))
#"""
#Method 2 - single print call
#"""
def printResults2(msg, lis):
    print(msg)
    print("\nMean = " + str(round(np.mean(lis), 5)) + 
          "\nMedian = " + str(round(np.median(lis), 5)) + 
          "\nMax = " + str(round(max(lis), 5)) +
          "\nMin = " + str(round(min(lis), 5)) +
          "\nRange = " + str(round((max(lis)-min(lis)), 5)))
#"""
#Method 3 - .join()
#"""
def printResults3(msg, lis):
    print(msg)
    strs = [("Mean = " + str(round(np.mean(lis), 5))),
                 ("Median = " + str(round(np.median(lis), 5))),
                 ("Max = " + str(round(max(lis), 5))),
                 ("Min = " + str(round(min(lis), 5))),
                 ("Range = " + str(round((max(lis)-min(lis)), 5)) + '\n')]
    sep = '\n'
    print(sep.join(strs))
#"""

#Functionality Testing
"""
ab = [np.random.randint(0,200) for _ in range(200)]
for c, p in enumerate([printResults1, printResults2, printResults3]):
    p(("Method " + str(c+1)), ab)
#"""
#Results:
#Means, Medians, Maxs, Mins, Ranges --> all identical

#Time Testing
"""
its = 250
for c, p in enumerate([printResults1, printResults2, printResults3]):
    startT = time()
    for it in range(its):
        #ab = [np.random.randint(0,200) for _ in range(200)]
        p("", [0])
    print("Method " + str(c+1) + ": " + str(time()-startT))
#"""
print("")
#Results: (250 iterations)
#printResults1----0.129
#printResults2----0.067
#printResults3----0.092

#printResults2 is the fastest
#One single print call with '\n'


# In[26]:


#Dataset dictionary, needed for graph methods 3 & 4
#"""
#Title : directed, weighted, offset, filepath to .csv file
datasets = {
    #BitcoinOTC dataset (5881 nodes, 35592 edges)
    #(directed, weighted, signed)
    "BitcoinOTC": (True, True, 
         r"D:\Sully\Documents\Computer Science BSc\Year 3\Term 2\Individual Project\datasets\soc-sign-bitcoinotc.csv"),
    #Facebook dataset (4039 nodes, 88234 edges)
    #(undirected, unweighted, unsigned)
    "Facebook": (False, False,
         r"D:\Sully\Documents\Computer Science BSc\Year 3\Term 2\Individual Project\datasets\facebook.csv")
}
#"""


# In[27]:


#Used in graph generation
#Removes any unconnected components of a given graph
def removeUnconnected(g):
    components = sorted(list(nx.weakly_connected_components(g)), key=len)
    while len(components)>1:
        component = components[0]
        for node in component:
            g.remove_node(node)
        components = components[1:]


# In[28]:


#Network generation methods

#First method - manual
"""
#Generate network from soc-BitcoinOTC dataset
#(5881 nodes, 35592 edges)
#(directed, weighted, signed)
#Initliaise directed graph
G11 = nx.DiGraph(Graph = "BitcoinOTC")
#Open files from path
with open(r"D:\Sully\Documents\Computer Science BSc\Year 3\Term 2\Individual Project\datasets\soc-sign-bitcoinotc.csv") as csvfile1:
    #read file and seperate items by comma
    #  (file is in format: X, Y, W
    #  indicating an edge from node X to node Y with weight W)
    readFile = csv.reader(csvfile1, delimiter=',')
    #for every row in the file...
    for row in readFile:
        #add the edge listed to the graph
        #the edges are reversed to indicate influence
        G11.add_edge((int(row[1])-1), (int(row[0])-1), trust=(int(row[2])+10)/20)
removeUnconnected(G11)

#Generate network from ego-Facebook dataset
#(4039 nodes, 88234 edges)
#(undirected, unweighted, unsigned, no parallel edges)
#Initliaise standard graph
G21 = nx.DiGraph(Graph = "Facebook")
#Open file from path
with open(r"D:\Sully\Documents\Computer Science BSc\Year 3\Term 2\Individual Project\datasets\facebook.csv") as csvfile:
    #read file and seperate items by comma
    #  (file is in format: X, Y
    #  indicating an edge from node X to node Y)
    readFile = csv.reader(csvfile, delimiter=',')
    #for every row in the file...
    for row in readFile:
        #add the edge listed to the graph
        #  (edges are reversed to indicate influence)
        G21.add_edge(int(row[1]), int(row[0]), trust=1)
        G21.add_edge(int(row[0]), int(row[1]), trust=1)
        
#Small, custom directed, unweighted graph
G31 = nx.DiGraph()
testedges = [(1,2), (2,4), (2,5), (2,6), (3,5), (4,5), (5,9), (5,10), (6,8),
            (7,8), (8,9)]
G31.add_edges_from(testedges)
nx.set_edge_attributes(G3, 1, 'trust')
#"""
#Second method - modularized with functions
"""
#Generates NetworkX graph from given file path:
def generateNetwork(name, weighted, directed, offset, path):
    newG = nx.DiGraph(Graph = name)
    with open(path) as csvfile:
        #read file and seperate items by comma
        #  (file is in format: X, Y, W - but may not contain W, indicating an edge from node X to node Y with weight W)
        readFile = csv.reader(csvfile, delimiter=',')
        for row in readFile:
            tr, dis = 1, 1
            #add the edge listed to the graph (the edges are reversed to indicate influence, & nodes are added automatically)
            #allow for custom weights in the csv file, distance = weight's reciprocal
            if weighted:
                tr = (int(row[2])+10)/20
                dis = 1-tr
            newG.add_edge(int(row[1])-offset, int(row[0])-offset, trust=tr, distance=dis)
            if not directed:
                newG.add_edge(int(row[0])-offset, int(row[1])-offset, trust=tr, distance=dis)
        if directed:
            removeUnconnected(newG)
    return newG

#Generate graphs from real datasets:

#  Generate network graph from soc-BitcoinOTC dataset (5881 nodes, 35592 edges)
#  (directed, weighted, signed)
G12 = generateNetwork("BitcoinOTC Network", True, True, 1, r"D:\Sully\Documents\Computer Science BSc\Year 3\Term 2\Individual Project\datasets\soc-sign-bitcoinotc.csv")

#  Generate network from ego-Facebook dataset (4039 nodes, 88234 edges)
#  (undirected, unweighted, unsigned, no parallel edges)
G22 = generateNetwork("Facebook Network", False, False, 0, r"D:\Sully\Documents\Computer Science BSc\Year 3\Term 2\Individual Project\datasets\facebook.csv")

#Generate mock graphs for testing and debugging:

#Small, custom directed, unweighted graph
G32 = nx.DiGraph()
testedges = [(1,2), (2,4), (2,5), (2,6), (3,5), (4,5), (5,9), (5,10), (6,8),
            (7,8), (8,9)]
G32.add_edges_from(testedges)
nx.set_edge_attributes(G32, 1, 'trust')

#Medium-sized path graph
#(each node only has edges to the node before and/or after it)
G4 = nx.path_graph(100)
nx.set_edge_attributes(G4, 1, 'trust')

#Medium-sized, randomly generated directed, unweighted graph
G5 = nx.DiGraph(Graph = "G5: random, trust=1")
for i in range(50):
    for j in range(10):
        targ = np.random.randint(-40,50)
        if targ > -1:
            G5.add_edge(i, targ, trust=1)

#Medium-sized, randomly generated directed, randomly-weighted graph
G6 = nx.DiGraph(Graph = "G6: random, randomized trust vals")
for i in range(50):
    for j in range(10):
        targ = np.random.randint(-40,50)
        if targ > -1:
            tru = np.random.uniform(0,1)
            G6.add_edge(i, targ, trust=tru)
#"""

#Third method - modularized, using the itertuples iteration method and
#  dictionaries to allow for additional graphs to be added simply.
#Also, offest no longer needed as it is calculated in function.
#"""
def generateNetwork(name, weighted, directed, path):
    #graph is initialized and named, dataframe is initialized
    newG = nx.DiGraph(Graph = name)
    data = pd.DataFrame()
    #pandas dataframe is read from .csv file,
    #  with weight if weighted, without if not
    if weighted:
        data = pd.read_csv(path, header=None, usecols=[0,1,2],
                           names=['Node 1', 'Node 2', 'Weight'])
    else:
        data = pd.read_csv(path, header=None, usecols=[0,1],
                           names=['Node 1', 'Node 2'])
        data['Weight'] = 1
    #offset is calculated from minimum nodes
    offset = min(data[['Node 1', 'Node 2']].min())
    #each row of dataframe is added to graph as an edge
    for row in data.itertuples(False, None):
        #trust=weight, & distance=(1-trust)
        trustval = row[2]
        newG.add_edge(row[1]-offset, row[0]-offset, 
                      trust=trustval, distance=(1-trustval))
        #if graph is undirected, edges are added again in reverse
        if not directed:
            newG.add_edge(row[0]-offset, row[1]-offset, 
                          trust=trustval, distance=(1-trustval))
    #unconnected components are removed
    if directed:
        removeUnconnected(newG)
    return newG
#"""
print("")


# In[8]:


#Functionality testing for method 1
#"""
for test in [(1,5), (4,5), (14,0)]:
    present = (test in G1.edges)
    print(str(test) + ": " + str(present))
#"""


# In[22]:


test = [(G11, G21, G31), (G12, G22, G32)]
for gc in range(3):
    for graphlist in test:
        print(graphlist[gc].size())
        print(str(len(graphlist[gc])) + "\n")


# In[29]:


#Graph compilation methods

#method 2
"""
#All real graphs
graphs = [G1, G2]
#All real graphs with their names attached
namedGraphs = [(G1, 'G1'), (G2, 'G2')]
#All real graphs with their optimal number of iterations
graphits = [(G1, 1000), (G2, 500)]
#All mock graphs
mockGraphs = [G3, G4, G5, G6]
#All mock graphs with their names attached
namedMockGraphs = [(G3, 'G3'), (G4, 'G4'), (G5, 'G5'), (G6, 'G6')]
#Randomly generated mock graphs
rndmGraphs = [G5, G6]
#All directed graphs
diGraphs = [G1, G2, G3, G5, G6]
#All directed graphs with their names attached
namedDiGraphs = [(G1, 'G1'), (G2, 'G2'), (G3, 'G3'), (G5, 'G5'), (G6, 'G6')]
#All graphs - real & mock
allGraphs = [G1, G2, G3, G4, G5, G6]
#"""
#method 3
#"""
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
mockG, name = nx.DiGraph(), "mock1: Custom, small"
testedges = [(1,2), (2,4), (2,5), (2,6), (3,5), (4,5), (5,9), (5,10), (6,8),
            (7,8), (8,9)]
mockG.add_edges_from(testedges)
nx.set_edge_attributes(mockG, 1, 'trust')
mockGraphs[name], diGraphs[name] = mockG, mockG

#Medium-sized path graph
#(each node only has edges to the node before and/or after it)
mockG, name = nx.path_graph(100), "mock2: Path graph, 100 nodes"
nx.set_edge_attributes(mockG, 1, 'trust')
mockGraphs[name] = mockG

#Medium-sized, randomly generated directed, unweighted graph
mockG, name = nx.DiGraph(), "mock3: Random, trustvals=1"
for i in range(50):
    for j in range(10):
        targ = np.random.randint(-40,50)
        if targ > -1:
            mockG.add_edge(i, targ, trust=1)
mockGraphs[name], rndmGraphs[name], diGraphs[name] = mockG, mockG, mockG

#Medium-sized, randomly generated directed, randomly-weighted graph
mockG, name = nx.DiGraph(), "mock4: Random, trustvals=random"
for i in range(50):
    for j in range(10):
        targ = np.random.randint(-40,50)
        if targ > -1:
            tru = np.random.uniform(0,1)
            mockG.add_edge(i, targ, trust=tru)
mockGraphs[name], rndmGraphs[name], diGraphs[name] = mockG, mockG, mockG
#"""
print("")


# In[67]:


#Functional testing for graph method 3
#Print numbers of nodes & edges
#"""
for graphlist in [namedGraphs, namedMockGraphs]:
    for g in graphlist:
        print(g[1] + ": " + str(g[0].size()))
        print(g[1] + ": " + str(len(g[0])) + "\n")
    
for graphlist in [realGraphs, mockGraphs]:
    for g in graphlist:
        print(g + ": " + str(graphlist[g].size()))
        print(g + ": " + str(len(graphlist[g])) + "\n")
#"""
print("")


# In[13]:


#Functions needed for graph methods 2 & 3, for time testing

def removeUnconnected2(g):
    for component in list(nx.weakly_connected_components(g)):
        if len(component) < 3:
            for node in component:
                g.remove_node(node)
    
def removeUnconnected(g):
    components = sorted(list(nx.weakly_connected_components(g)), key=len)
    while len(components)>1:
        component = components[0]
        for node in component:
            g.remove_node(node)
        components = components[1:]
        
#Generates NetworkX graph from given file path:
def generateNetwork(name, weighted, directed, offset, path):
    NG = nx.DiGraph(Graph = name)
    with open(path) as csvfile:
        #read file and seperate items by comma
        #  (file is in format: X, Y, W - but may not contain W, indicating an edge from node X to node Y with weight W)
        readFile = csv.reader(csvfile, delimiter=',')
        for row in readFile:
            tr, dis = 1, 1
            #add the edge listed to the graph (the edges are reversed to indicate influence, & nodes are added automatically)
            #allow for custom weights in the csv file, distance = weight's reciprocal
            if weighted:
                tr = (int(row[2])+10)/20
                dis = 1-tr
            NG.add_edge(int(row[1])-offset, int(row[0])-offset, trust=tr, distance=(dis))
            if not directed:
                NG.add_edge(int(row[0])-offset, int(row[1])-offset, trust=tr, distance=(dis))
        if directed:
            removeUnconnected(NG)
    return NG


# In[11]:


#Graphing methods 1 vs 2 time testing
#  Manual --> Modular, slight time improvement
#"""
def compareGraphMethods(its):
    a, b = 0, 0
    for it in range(its):
    
        startT1 = time()
        #Generate network from soc-BitcoinOTC dataset
        #(5881 nodes, 35592 edges)
        #(directed, weighted, signed)
        #Initliaise directed graph
        G1 = nx.DiGraph(Graph = "BitcoinOTC")
        #Open files from path
        with open(r"D:\Sully\Documents\Computer Science BSc\Year 3\Term 2\Individual Project\datasets\soc-sign-bitcoinotc-EDITED.csv") as csvfile1:
            #read file and seperate items by comma
            #  (file is in format: X, Y, W
            #  indicating an edge from node X to node Y with weight W)
            readFile = csv.reader(csvfile1, delimiter=',')
            #for every row in the file...
            for row in readFile:
                #if the first node is not in the graph already,
                if (int(row[0])-1) not in G1:
                    #add it
                    G1.add_node((int(row[0]))-1)
                #if the second node is not in the graph already,
                if (int(row[1])-1) not in G1:
                    #add it
                    G1.add_node((int(row[1]))-1)
                #add the edge listed to the graph
                #(this happens every row without fail)
                #the edges are reversed to indicate influence
                G1.add_edge((int(row[1])-1), (int(row[0])-1), trust=(int(row[2])+10)/20)
        removeUnconnected(G1)

        #Generate network from ego-Facebook dataset
        #(4039 nodes, 88234 edges)
        #(undirected, unweighted, unsigned, no parallel edges)
        #Initliaise standard graph
        G2 = nx.DiGraph(Graph = "Facebook")
        #Open file from path
        with open(r"D:\Sully\Documents\Computer Science BSc\Year 3\Term 2\Individual Project\datasets\facebook_combined.csv") as csvfile:
            #read file and seperate items by comma
            #  (file is in format: X, Y
            #  indicating an edge from node X to node Y)
            readFile = csv.reader(csvfile, delimiter=',')
            #for every row in the file...
            for row in readFile:
                #if the first node is not in the graph already,
                if int(row[0]) not in G2:
                    #add it
                    G2.add_node(int(row[0]))
                #if the second node is not in the graph already,
                if int(row[1]) not in G2:
                    #add it
                    G2.add_node(int(row[1]))
                #add the edge listed to the graph
                #  (this happens every row without fail, and
                #  the edges are reversed to indicate influence)
                G2.add_edge(int(row[1]), int(row[0]), trust=1)
                G2.add_edge(int(row[0]), int(row[1]), trust=1)
        a += (time()-startT1)

        startT2 = time()
        #  Generate network graph from soc-BitcoinOTC dataset (5881 nodes, 35592 edges)
        #  (directed, weighted, signed)
        G1 = generateNetwork("BitcoinOTC Network", True, True, 1, r"D:\Sully\Documents\Computer Science BSc\Year 3\Term 2\Individual Project\datasets\soc-sign-bitcoinotc-EDITED.csv")

        #  Generate network from ego-Facebook dataset (4039 nodes, 88234 edges)
        #  (undirected, unweighted, unsigned, no parallel edges)
        G2 = generateNetwork("Facebook Network", False, False, 0, r"D:\Sully\Documents\Computer Science BSc\Year 3\Term 2\Individual Project\datasets\facebook_combined.csv")
        b += (time() - startT2)
    
    return (("First manual method: ", a), ("Second modular method: ", b))

#Function to run them a given number of times, and return running times to compare
#1. 64.046   2. 58.004
#Second modular approach is faster, as expected, but not by much.

#print(compareGraphMethods(100))


# In[56]:


#Graph method 3 testing different iteration methods
methods = {'index','loc','iloc','iterrows','itertuples'}

#Generates NetworkX graph from given file path with a given method:
def generateNetwork2(name, weighted, directed, offset, path, itermethod):
    #graph is initialized and named, dataframe is initialized
    newG = nx.DiGraph(Graph = name)
    data = pd.DataFrame()
    
    #pandas dataframe is created from .csv file,
    #  with weight if weighted, without if not
    if weighted:
        data = pd.read_csv(path, header=None, usecols=[0,1,2],
                           names=['Node 1', 'Node 2', 'Weight'])
    else:
        data = pd.read_csv(path, header=None, usecols=[0,1],
                           names=['Node 1', 'Node 2'])
        data['Weight'] = 1
        
    #offset is calculated from minimum nodes
    offset = min(data[['Node 1', 'Node 2']].min())
    
    #if graph is undirected, edges are added twice in parallel
    #different iteration methods below:
    #index itermethod
    if itermethod == 'index':
        for i in data.index:
            trustval = data['Weight'][i]
            newG.add_edge(data['Node 2'][i]-offset, 
                          data['Node 1'][i]-offset, 
                          trust=trustval, distance=1-trustval)
            if not directed:
                newG.add_edge(data['Node 1'][i]-offset, 
                              data['Node 2'][i]-offset, 
                              trust=trustval, 
                              distance=1-trustval)
    #loc ietrmethod
    elif itermethod == 'loc':
        for i in range(len(data)):
            trustval = data.loc[i, 'Weight']
            newG.add_edge(data.loc[i, 'Node 2']-offset, 
                          data.loc[i, 'Node 1']-offset, 
                          trust=trustval, distance=1-trustval)
            if not directed:
                newG.add_edge(data.loc[i, 'Node 1']-offset, 
                              data.loc[i, 'Node 2']-offset, 
                              trust=trustval, distance=1-trustval)
    #iloc itermethod
    elif itermethod == 'iloc':
        for i in range(len(data)):
            trustval = data.iloc[i, 2]
            newG.add_edge(data.iloc[i, 1]-offset, 
                          data.iloc[i, 0]-offset, 
                          trust=trustval, distance=1-trustval)
            if not directed:
                newG.add_edge(data.iloc[i, 0]-offset, data.iloc[i, 1]-offset, 
                              trust=trustval, distance=1-trustval)
    
    #iterrows itermethod
    elif itermethod == 'iterrows':
        for i, row in data.iterrows():
            trustval = row['Weight']
            newG.add_edge(row['Node 2']-offset, row['Node 1']-offset, 
                          trust=trustval, distance=1-trustval)
            if not directed:
                newG.add_edge(row['Node 1']-offset, 
                              row['Node 2']-offset, 
                              trust=trustval, distance=1-trustval)
    
    #itertuples itermethod
    elif itermethod == 'itertuples':
        for row in data.itertuples(False, None):
            trustval = row[2]
            newG.add_edge(row[1]-offset, row[0]-offset, 
                          trust=trustval, distance=(1-trustval))
            if not directed:
                newG.add_edge(row[0]-offset, row[1]-offset, 
                              trust=trustval, distance=(1-trustval))

    #unconnected components are removed
    if directed:
        removeUnconnected(newG)
    return newG


#Functionality testing function
def itermethodEqual(method1, methodlist, cleared):
    for count, g in enumerate(datasets):
        if not len(cleared[count]):
            cleared[count].append(g + ": ")
        networks = [(generateNetwork2((g + " Network"), 
                                      datasets[g][0], datasets[g][1], 
                                      datasets[g][2], datasets[g][3], 
                                      method1), method1)]
        for method in methodlist:
            if method not in cleared[count]:
                networks.append((generateNetwork2((g + " Network"), 
                                                  datasets[g][0], 
                                                  datasets[g][1], 
                                                  datasets[g][2], 
                                                  datasets[g][3], 
                                                  method), method))
        missingnodes = [(g + " " + method1 + " missing nodes:")]
        clear = True
        for c, network in enumerate(networks[1:]):
            if set(network[0].nodes) == set(networks[0][0].nodes):
                continue
            unequal = False
            for node in network[0]:
                if node not in networks[0][0]:
                    if not unequal:
                        missingnodes[c].append(method1 + " - " + network[1])
                        unequal = True
                    missingnodes[c].append(node)
                    clear = False
        if clear:
            cleared[count].append(method1)
            print("Cleared methods so far:\n" + str(cleared[0]) 
                  + "\n" + cleared[1] + "\n")
        else:
            for l in missingnodes:
                print(l)
            print("")
    return cleared                     

#Functionality testing area
"""
clearmethods = [[], []]
for method in methods:
    clearmethods = itermethodEqual(method, (methods - set(method)), clearmethods)
#"""
#Results:
#none had equal sets of nodes -> led me to typo in offset
#iloc was unequal to all others -> led me to typo in iloc
#when typos were fixed -> all were identical


#Time testing function to generate all real graphs for a given method
#  and return the time taken to do so + the time taken so far.
def itermethodTime(method, timeSoFar):
    startTime = time()
    testGraphs = {}
    for g in datasets:
        testGraphs[g] = generateNetwork2((g + " Network"), 
                                         datasets[g][0], datasets[g][1], 
                                         datasets[g][2], datasets[g][3], 
                                         method)
    return timeSoFar + (time()-startTime)

#Time testing area - Repeatedly generates graphs for a number of iterations, 
#  for each method, and prints the time elapsed for each
"""
for method in methods:
    t = 0
    for i in range(10):
        t = itermethodTime(method, t)
    print(method + " = " + str(t))
#"""
#Results: (10 iterations)
#index--------38.352
#iterrows-----95.219
#loc----------45.991
#itertuples---5.541
#iloc---------130.186
#Itertuples is the fastest by far, so was implemented

print("")


# In[73]:





# In[ ]:


#Failed section: method 3 for graph iteration method functionality testing
"""
        checknodes = {}
        checkedges = {}
        m1 = methodlist[0]
        test = networks[m1]
        testnodes = {node for node in test}
        testedges = {edge for edge in test}
        for m2 in networks:
            if m2 == networks[m1]:
                continue
            check = networks[m2]
            if m2 == m1:
                print("Same graph\n")
                continue
            for node in check:
                if node not in testnodes:
                    if node not in checknodes[m2 + " " + m1] and node not in checknodes[m1 + " " + m2]:
                        checknodes[m1 + " " + m2].append(node)
            for edge in check:
                if edge not in testedges:
                    if edge not in checkedges[m2 + " " + m1] and edge not in checknodes[m1 + " " + m2]:
                        checkedges[m1 + " " + m2].append(edge)
        
        print(g + ":\n" + "Node dict, key/values pair by pair:\n")
        for nodepair in checknodes:
            print(nodepair + "\n" + str(checknodes[nodepair]) + "\n")
        print("\n" + g + ":\n" +"Edge dict, key/values pair by pair:\n")
        for edgepair in checkedges:
            print(edgepair + "\n" + str(checkedges[edgepair]) + "\n")
"""          
"""
            for g2 in networks:
                check = networks[g2]
                checknodes = {node for node in check}
                checkedges = {edge for edge in check}
                if g2 == g1:
                    "Same graph\n"
                #    continue
                for node in networks[g2]:
                    if node not in testnodes:
                        checknodes[(g1 + " " + g2)].append(node)
                if not (testnodes == checknodes):
                    print("Nodes different in " + g + "!\n" 
                          + g1 + " - " + g2 + "\n")
                if not (testedges == checkedges):
                    print("Edges different in " + g + "!\n" 
                          + g1 + " - " + g2 + "\n")
"""


# In[ ]:


#Accessing degree recips & rel degs for probabilities

#After deciding on log-scaling -> requires minMaxNormalizing,
#  but that requires knowing the min and max of all logged probs.
#So I implemented a manual method of calculating the normalized prob
#  during the cascade process. 
#Then I improved upon this with a method of assigning them to dictionaries, 
#  made other improvements/optimizations and ran multiple tests

#Method 1 - manually while cascading, everytime when needed
#  (requires entire list to be calculated first for mmNormalize)
"""
def mmNormalizeLis(lis):
    elMax, elMin = max(lis), min(lis)
    return list(map(lambda x : ((x - elMin)/(elMax - elMin)), lis))

def allRelDegs(g):
    #allRds = []
    allRds, allRdsDict = [], {}
    for target in g:
        if not g.in_degree(target):
            continue
        snd = 0
        for neighbour in g.predecessors(target):
            snd += 1
        for targeting in g.predecessors(target):
            rdval = log(g[targeting][target]['trust']*(g.out_degree(targeting) / snd))
            allRds.append(rdval)
            allRdsDict[(targeting, target)] = log((g.out_degree(targeting) / snd))
    return allRds, allRdsDict

relDegsTest1 = allRelDegs(graphs['Facebook'])
#relDegsTest1, relDegsTestDict = allRelDegs(graphs['Facebook'])
elMax, elMin = max(relDegsTest1), min(relDegsTest1)
relDegsTest2 = mmNormalizeLis(relDegsTest1)
#relDegsTest3 = mmNormalizeDict(relDegsTestDict, 
#                              max(relDegsTestDict.values()), 
#                              min(relDegsTestDict.values()))

def mmNormalizeSingle(val):
    #elMax, elMin = max(relDegsTest1), min(relDegsTest1)
    #normLis = list(map(lambda x: ((x-elMin)/(elMax-elMin)), relDegsTest1))
    #print("Single test normalization:\nMaximum = " + str(elMax) + 
    #      "\nMinimum = " + str(elMin) + "Mean = " + 
    #      str(np.mean(relDegsTest1)) + "\n")
    return ((val - elMin)/(elMax - elMin))

#printResults("Test list: ", relDegsTest1)
#printResults("Test normalized list: ", relDegsTest2)
#printResults("Test normalized dict: ", list(relDegsTestDict.values()))

#startT = time()
#print("Test normalized dict: " + str(np.mean(list(relDegsTest3.values()))) 
#             + "\n" + str(time()-startT) + " secs\n")


#Functionality & quality testing of assignment functions
#"""
for assignTest in [0,1]:
    print('assign method: ' + str(assignTest))
    measureTime1(cascade, graphs['Facebook'], [1], 15, 'WC2', assignTest, 
                 0.5, 0.7, 0.7, 0.08)
    print("")
#"""
print("")

#Results: (S=[1], 75its, pp=0.5, qf=0.7, sf=0.7, tf=0.04)
#Manual log-scaling----------1.427 spread, 0.368secs
#Pre-assigned log-scaling----888.773 spread, 77.491secs
#Initially not equal --> typo in allRelDegs (return line indented so no loop)

#Lowered iterations due to it taking so long to process the manual method
#Results: (S=[1])



#"""
#Method 2 - assign to dictionary, at the start of WC1 or WC2
#  3 different functions: logscale, squareroot, cuberoot
"""
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
    print("bloop")
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
            rds[(targeting, target)] = log(g[targeting][target]['trust']*(g.out_degree(targeting) / snd))
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
        snd = sum([(g[neighbour][target]['trust']*g.out_degree(neighbour)) 
                   for neighbour in g.predecessors(target)])
        for targeting in g.predecessors(target):
            rds[(targeting, target)] = (((g[targeting][target]['trust']*g.out_degree(targeting)) / snd) ** (1/2))
    nx.set_edge_attributes(g, rds, "relDeg")

#Cube-rooting method
def assignRelDegs3(g):
    rds = {}
    for target in g:
        if not g.in_degree(target):
            continue
        snd = sum([(g[neighbour][target]['trust']*g.out_degree(neighbour)) 
                   for neighbour in g.predecessors(target)])
        for targeting in g.predecessors(target):
            rds[(targeting, target)] = (((g[targeting][target]['trust']*g.out_degree(targeting)) / snd) ** (1/3))
    nx.set_edge_attributes(g, rds, "relDeg")
    
#if,elif,else statement added to cascade
    if model == 'WC1':
        assignRecips1(g)
        success = model
    elif model == 'WC2':
        assignRelDegs1(g)
        success = model
    else:
        success = model
#"""
#Method 3 - method 2 with AssignSelect func, to select which
#  assign function from variable in cascade (logscale=default)
#Same code as Method 2, with the following addition & change to cascade
"""
#Assign method dictionary for selection depending on parameters
assignMods = {'WC1': {1: assignRecips1, 2: assignRecips2, 3: assignRecips3}, 
              'WC2': {1: assignRelDegs1, 2: assignRelDegs2, 3:assignRelDegs3}}
#Selects and runs appropriate assigning method
def assignSelect(g, propMod, assignMod):
    if assignMod:
        assignMods[propMod][assignMod](g)
        
#assign paramater added to cascade, along with those 3 lines
def cascade(g, s, it=0, model='IC', assign=1, pp=0.2, qf=1, sf=1, tf=0):
    if model != 'IC' and assign:
        assignSelect(g, model, assign)
    success = model
#"""


# In[ ]:


#Access/Assign Methods 1 & 3 Functionality&Time Testing

"""

"""


#Access/Assign Methods 1 & 3 Time Testing

"""

"""
#Results: (S=[1], 75its, pp=0.5, qf=0.7, sf=0.7, tf=0.04)
#Manual log-scaling----------1.427 spread, 0.368secs
#Pre-assigned log-scaling----888.773 spread, 77.491secs
#Initially not equal --> typo in allRelDegs (return line indented so no loop)
#Due to the typo these results are erroneous

#Lowered iterations due to it taking so long to process the manual method
#Results: (S=[1], 75its, pp=0.5, qf=0.7, sf=0.7, tf=0.04)
#Manual log-scaling---------1188.533 spread, 328.085secs
#Pre-assigned log-scaling---1188.533 spread, 13.405secs


# In[ ]:


#Assign Method 2/3 Optimization

#Optimization for calculating, normalizing & assigning probabilities
#  (log-scaled RelDegs - WC2 here, but applicable to all methods)
#Specifically optimizing the way in which the sum of all a target's 
#  neighbours' degrees or maximums/minimums of a dictionary are obtained.

#Sum neighbour degree
#method 1 - Integer & For-loop Method
"""
def assignRelDegs11(g):
    rds = {}
    for target in g:
        if not g.in_degree(target):
            continue
        snd = sum([(g[neighbour][target]['trust']*g.out_degree(neighbour)) 
                   for neighbour in g.predecessors(target)])
        for targeting in g.predecessors(target):
            rds[(targeting, target)] = log((g[targeting][target]['trust']*g.out_degree(targeting)) / snd)
    rds = mmNormalizeDict(rds, max(rds.values()), min(rds.values()))
    nx.set_edge_attributes(g, rds, "relDeg")
"""
#method 2 - List Comprehension method
"""
def assignRelDegs12(g):
    rds = {}
    for target in g:
        if not g.in_degree(target):
            continue
        snd = 0
        for neighbour in g.predecessors(target):
            snd += (g[neighbour][target]['trust']*g.out_degree(neighbour))
        for targeting in g.predecessors(target):
            rds[(targeting, target)] = log((g[targeting][target]['trust']*g.out_degree(targeting)) / snd)
    rds = mmNormalizeDict(rds, max(rds.values()), min(rds.values()))
    nx.set_edge_attributes(g, rds, "relDeg")
"""

##Dictionary Maximum & Minimum
#method 1 - .values()
"""
def assignRelDegs21(g):
    rds = {}
    for target in g:
        if not g.in_degree(target):
            continue
        snd = 0
        for targeting in g.predecessors(target):
            snd += (g[targeting][target]['trust']*g.out_degree(targeting))
        for targeting in g.predecessors(target):
            rds[(targeting, target)] = log((g[targeting][target]['trust']*g.out_degree(targeting)) / snd)
    rds = mmNormalizeDict(rds, max(rds.values()), min(rds.values()))
    nx.set_edge_attributes(g, rds, "relDeg")
"""
#method 2 - index and key.get Method
"""
def assignRelDegs22(g):
    rds = {}
    for target in g:
        if not g.in_degree(target):
            continue
        snd = 0
        for targeting in g.predecessors(target):
            snd += (g[targeting][target]['trust']*g.out_degree(targeting))
        for targeting in g.predecessors(target):
            rds[(targeting, target)] = log((g[targeting][target]['trust']*g.out_degree(targeting)) / snd)
    elMax = rds[max(rds, key=rds.get)]
    elMin = rds[min(rds, key=rds.get)]
    rds = mmNormalizeDict(rds, elMax, elMin)
    nx.set_edge_attributes(g, rds, "relDeg")
"""
#method 3 - itemgetter(1)
"""
def assignRelDegs23(g):
    rds = {}
    for target in g:
        if not g.in_degree(target):
            continue
        snd = 0
        for targeting in g.predecessors(target):
            snd += (g[targeting][target]['trust']*g.out_degree(targeting))
        for targeting in g.predecessors(target):
            rds[(targeting, target)] = log((g[targeting][target]['trust']*g.out_degree(targeting)) / snd)
    elMax = max(rds.items(), key=itemgetter(1))[1]
    elMin = min(rds.items(), key=itemgetter(1))[1]
    rds = mmNormalizeDict(rds, elMax, elMin)
    nx.set_edge_attributes(g, rds, "relDeg")  
"""

#SumNeighbourDegree Time Testing
"""
methods, its = [("SumListComp", assignRelDegs11), 
                ("IntegerForLoop", assignRelDegs12)], 20
for method in methods:
    startT = time()
    for _ in range(its):
        method[1](gr)
    print(method[0] + ": " + str(time()-startT) + " secs")
#"""
#Results: (20 iterations)
#SumListComp-------29.476
#IntegerForLoop----28.361

#IntegerForLoop was marginally quicker, probably due to the lack of
#  creating a new list each time.
    
#MaxMinDictionary Time Testing
"""
methods, its = [("values()", assignRelDegs21), 
                (".get() & index", assignRelDegs22), 
                ("itemgetter(1)", assignRelDegs23)], 20
for method in methods:
    startT = time()
    for _ in range(its):
        method[1](gr)
    print(method[0] + ": " + str(time()-startT) + " secs")
"""
print("")
#Results: (20 iterations)
#.values()------------36.638
#.get()&index---------38.029
#itemgetter(1)&[1]----38.418

#.values() was marginally faster than the others


# In[ ]:


#Printing 


# In[ ]:





# In[ ]:





# In[ ]:


#Comparing histograms of normalized probabilites
#Original method
"""
a = calcRelDegs(G1, False)
figs, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axs[0].hist(a)
axs[0].set(xlabel="Probabilities", ylabel="Base relational degrees")
axs[1].hist(varsList(a))
axs[1].set(xlabel="Probabilities", ylabel="Base relational degrees w/ probability variables")

b = rootList(a, (1/2))
figs, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axs[0].hist(b)
axs[0].set(xlabel="Probabilities", ylabel="Square rooted")
axs[1].hist(varsList(b))
axs[1].set(xlabel="Probabilities", ylabel="Square rooted w/ probability variables")

b = rootList(a, (1/3))
figs, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axs[0].hist(b)
axs[0].set(xlabel="Probabilities", ylabel="Cube rooted")
axs[1].hist(varsList(b))
axs[1].set(xlabel="Probabilities", ylabel="Cube rooted w/ probability variables")

b = mmNormalize(a)
figs, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axs[0].hist(b)
axs[0].set(xlabel="Probabilities", ylabel="min-max normalized")
axs[1].hist(varsList(b))
axs[1].set(xlabel="Probabilities", ylabel="min-max normalized w/ probability variables")

b = zNormalize(a)
figs, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axs[0].hist(b)
axs[0].set(xlabel="Probabilities", ylabel="z-score normalized")
axs[1].hist(varsList(b))
axs[1].set(xlabel="Probabilities", ylabel="z-score normalized w/ probability variables")

b = robustNormalize(a)
figs, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axs[0].hist(b)
axs[0].set(xlabel="Probabilities", ylabel="robust normalized")
axs[1].hist(varsList(b))
axs[1].set(xlabel="Probabilities", ylabel="robust normalized w/ probability variables")

b = logList(a)
figs, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axs[0].hist(b)
axs[0].set(xlabel="Probabilities", ylabel="logList")
axs[1].hist(varsList(b))
axs[1].set(xlabel="Probabilities", ylabel="logList w/ probability variables")
"""

#Scaled values between 0 and 1
"""
a = calcRelDegs(G1, False)
if max(b) > 1 or min(b) < -1:
    b = mmNormalize(b)
figs, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axs[0].hist(a)
axs[0].set(xlabel="Probabilities", ylabel="Base relational degrees")
axs[1].hist(varsList(a))
axs[1].set(xlabel="Probabilities", ylabel="Base relational degrees w/ probability variables")

b = rootList(a, (1/2))
if max(b) > 1 or min(b) < -1:
    b = mmNormalize(b)
figs, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axs[0].hist(b)
axs[0].set(xlabel="Probabilities", ylabel="Square rooted")
axs[1].hist(varsList(b))
axs[1].set(xlabel="Probabilities", ylabel="Square rooted w/ probability variables")

b = rootList(a, (1/3))
if max(b) > 1 or min(b) < -1:
    b = mmNormalize(b)
figs, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axs[0].hist(b)
axs[0].set(xlabel="Probabilities", ylabel="Cube rooted")
axs[1].hist(varsList(b))
axs[1].set(xlabel="Probabilities", ylabel="Cube rooted w/ probability variables")

b = mmNormalize(a)
if max(b) > 1 or min(b) < -1:
    b = mmNormalize(b)
figs, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axs[0].hist(b)
axs[0].set(xlabel="Probabilities", ylabel="min-max normalized")
axs[1].hist(varsList(b))
axs[1].set(xlabel="Probabilities", ylabel="min-max normalized w/ probability variables")

b = zNormalize(a)
if max(b) > 1 or min(b) < -1:
    b = mmNormalize(b)
figs, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axs[0].hist(b)
axs[0].set(xlabel="Probabilities", ylabel="z-score normalized")
axs[1].hist(varsList(b))
axs[1].set(xlabel="Probabilities", ylabel="z-score normalized w/ probability variables")

b = robustNormalize(a)
if max(b) > 1 or min(b) < -1:
    b = mmNormalize(b)
figs, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axs[0].hist(b)
axs[0].set(xlabel="Probabilities", ylabel="robust normalized")
axs[1].hist(varsList(b))
axs[1].set(xlabel="Probabilities", ylabel="robust normalized w/ probability variables")

b = logList(a)
if max(b) > 1 or min(b) < -1:
    b = mmNormalize(b)
figs, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
axs[0].hist(b)
axs[0].set(xlabel="Probabilities", ylabel="logList")
axs[1].hist(varsList(b))
axs[1].set(xlabel="Probabilities", ylabel="logList w/ probability variables")
"""

#Modular approach
"""

"""


# In[ ]:


#Previous methdos for variable comparison / graph plotting:
#Manual approach
"""
qty, its = 5, 50

for g in graphs:
    S = randomSeeds(graphs[g], qty)
    print(g + "\nseed set: " + str(S) + "\n")
    for propMod in propMods:
        res, t = measureTimeRet(cascade, graphs[g], S, its, propMod[0])
        print(propMod[1] + " " + str(its) + " iterations")
        print(str(res) + " " + str(t) + " secs")
    print("\n")

qty, its = 8, 250

S = randomSeeds(G1, qty)
print(str(G1.graph) + "\nseed set: " + str(S) + "\n")
g1values = [cascade(G1, S, its, 'IC', qf=q*0.1) for q in range(0, 11, 1)]
print("G1 results: " + str(g1values) + "\n")

S = randomSeeds(G2, qty)
print(str(G2.graph) + "\nseed set: " + str(S) + "\n")
g2values = [cascade(G2, S, its, 'IC', qf=q*0.1) for q in range(0, 11, 1)]
print("G2 results: " + str(g2values) + "\n")

#g1values = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]
#g2values = [50, 100, 150, 200, 250, 300, 460, 600, 720, 900, 1050]
xValues = [x*0.1 for x in range(0, 11, 1)]

fig, axs = plt.subplots(figsize=(10,6))
axs.set_xlabel("Quality factor")
axs.set_ylabel("Spread")
axs.set_title("Effect of quality factor on spread within a network")
axs.plot(xValues, g1values, label="G1: Bitcoin")
axs.plot(xValues, g2values, label="G2: Facebook")
axs.legend()

plt.show()
#"""
#Modular approach
"""#
#Compare positive influence spreads for a given list of graphs with 
#  a range of different quality factors, and plot a line graph to show.
def comparePP(gs, qty, its, seedFunc, model, pps):
    values = []
    for i,g in enumerate(gs):
        startTime = time()
        S = seedFunc(gs[g], qty)
        print(g + "\nseed set: " + str(S) + "\n" + 
              str(round((time() - startTime), 5)) + " secs\n")
        startTime = time()
        values.append([cascade(gs[g], S, its, model, pp=p) for p in pps])
        print(g + ":\n" + str(values[i]) + "\n" + 
              str(round((time() - startTime), 5)) + " secs\n")
    figs, axs = plt.subplots(figsize=(10,6))
    axs.set_xlabel("Propagation probability")
    axs.set_ylabel("Spread")
    axs.set_title("Effect of propagation probability on spread within a network\n" + 
                  "Cascade model: " + model)
    for i,g in enumerate(gs):
        axs.plot(pps, values[i], label=g)
    axs.legend()
    plt.show()
    
comparePP(graphs, 4, 1000, randomSeeds, 'IC', [pp*0.05 for pp in range(1,20)])
    
#Compare positive influence spreads for a given list of graphs with 
#  a range of different quality factors, and plot a line graph to show.
def compareQF(gs, qty, its, seedFunc, model, qfs, sw):
    values = []
    for i,g in enumerate(gs):
        startTime = time()
        S = seedFunc(gs[g], qty)
        print(g + "\nseed set: " + str(S) + "\n" + 
              str(round((time() - startTime), 5)) + " secs\n")
        startTime = time()
        values.append([cascade(gs[g], S, its, model, qf=q, sf=sw) for q in qfs])
        print(g + ":\n" + str(values[i]) + "\n" + 
              str(round((time() - startTime), 5)) + " secs\n")
    figs, axs = plt.subplots(figsize=(10,6))
    axs.set_xlabel("Quality factor")
    axs.set_ylabel("Spread")
    axs.set_title("Effect of Quality & Switch Factors\n" + 
                  "Switch factor = " + str(sw))
    for i,g in enumerate(gs):
        axs.plot(qfs, values[i], label=g)
    axs.legend()
    plt.show()
    
for sw in [b*0.1 for b in range(1)]:
    compareQF(graphs, 5, 2, randomSeeds, 'IC', [q*0.1 for q in range(0,11,1)], sw)
#compareQF(graphs, 8, 50, randomSeeds, 'WC1', [q*0.1 for q in range(0,11,1)])
#compareQF(graphs, 8, 50, randomSeeds, 'WC2', [q*0.1 for q in range(0,11,1)])

#Compare positive influence spreads for a given list of graphs with 
#  a range of different switch factors, and plot a line graph to show.
def compareSF(gs, qty, its, seedFunc, model, sfs, qual):
    values = []
    for i,g in enumerate(gs):
        startTime = time.time()
        S = seedFunc(g, qty)
        print(str(g.graph) + "\nseed set: " + str(S) + "\n" + 
              str(round((time.time() - startTime), 5)) + " secs\n")
        startTime = time.time()
        values.append([cascade(g, S, its, model, qf=qual, sf=sw) for sw in sfs])
        print(str(g.graph) + ":\n" + str(values[i]) + "\n" + 
              str(round((time.time() - startTime), 5)) + " secs\n")
    figs, axs = plt.subplots()
    axs.set_xlabel("Switch factor")
    axs.set_ylabel("Spread")
    axs.set_title("Effect of switch factor on spread within a network\n" + 
                  "Cascade model: " + model + ". Quality factor: " + str(qual))
    for i,g in enumerate(gs):
        axs.plot(sfs, values[i], label=str(g.graph))
    axs.legend()
    plt.show()
    
#Compare positive influence spreads for a given list of graphs with 
#  a range of different switch factors, and plot a line graph to show.
def compareSF2(g, s, its, sfs, qfs, col):
    values, labels = [], [str(q) for q in qfs]
    #values, labels = [[] for _ in range(len(qfs))], [str(q) for q in qfs]
    for q in (range(len(qfs))):
        #print("Quality factor: " + str(qfs[q]))
        for sw in range(len(sfs)):
            startTime = time()
            values.append(cascade(graphs['BitcoinOTC'], S, its, pp=0.6, qf=qfs[q], sf=sfs[sw]))
            #print("Switch factor: " + str(sfs[sw]) 
            #      + ":\n" + str(values[q]) + "\n" + 
            #      str(round((time() - startTime), 5)) + " secs\n")
    figs, axs = plt.subplots()
    for q in range(len(qfs)):
        axs.plot(sfs, values, label=g, color=col)
    axs.set_xlabel("Switch factor")
    axs.set_ylabel("Spread")
    axs.set_title(g + "\nQuality factor: " + str(qual))
    axs.legend()
    
compareSF(graphs, 8, 25, randomSeeds, 'IC', [sf*0.1 for sf in range(0,11,1)], 0.8)
compareSF(graphs, 8, 50, randomSeeds, 'WC1', [sf*0.1 for sf in range(0,11,1)], 0.8)
compareSF(graphs, 8, 50, randomSeeds, 'WC2', [sf*0.1 for sf in range(0,11,1)], 0.8)
compareSF(graphs, 8, 50, randomSeeds, 'IC', [sf*0.1 for sf in range(0,11,1)], 0.5)
compareSF(graphs, 8, 50, randomSeeds, 'WC1', [sf*0.1 for sf in range(0,11,1)], 0.5)
compareSF(graphs, 8, 50, randomSeeds, 'WC2', [sf*0.1 for sf in range(0,11,1)], 0.5)
compareSF(graphs, 8, 50, randomSeeds, 'IC', [sf*0.1 for sf in range(0,11,1)], 0.3)
compareSF(graphs, 8, 50, randomSeeds, 'WC1', [sf*0.1 for sf in range(0,11,1)], 0.3)
compareSF(graphs, 8, 50, randomSeeds, 'WC2', [sf*0.1 for sf in range(0,11,1)], 0.3)

#Compare positive influence spreads for a given list of graphs with 
#  a range of different time factors, and plot a line graph to show.
def compareTF(gs, qty, its, seedFunc, model, tfs):
    values = []
    for i,g in enumerate(gs):
        startTime = time.time()
        S = seedFunc(g, qty)
        print(str(g.graph) + "\nseed set: " + str(S) + "\n" + 
              str(round((time.time() - startTime), 5)) + " secs\n")
        startTime = time.time()
        values.append([cascade(g, S, its, model, tf=t) for t in tfs])
        print(str(g.graph) + ":\n" + str(values[i]) + "\n" + 
              str(round((time.time() - startTime), 5)) + " secs\n")
    figs, axs = plt.subplots()
    axs.set_xlabel("Time factor")
    axs.set_ylabel("Spread")
    axs.set_title("Effect of time factor on spread within a network\n" + 
                  "Cascade model: " + model)
    for i,g in enumerate(gs):
        axs.plot(tfs, values[i], label=str(g.graph))
    axs.legend()
    plt.show()

compareTF(graphs, 8, 50, randomSeeds, 'IC', [tf*0.01 for tf in range(0,11,1)])
compareTF(graphs, 8, 50, randomSeeds, 'WC1', [tf*0.01 for tf in range(0,11,1)])
compareTF(graphs, 8, 50, randomSeeds, 'WC2', [tf*0.01 for tf in range(0,11,1)])
"""
#Unfinished comparison function
"""
def compareVariables(g, s, its, compareVars, compare):
    startTT, values = time(), []
    for c1, var1 in enumerate(compareVars[0]):
        for c2, var2 in enumerate(compareVars[1]):
            for c3, var3 in enumerate(compareVars[2]):
                startT = time()
                casc = cascade(gs[g], s, its, pp=var1[1], qf=var2[1], sf=var3[1])
                endT = round((time()-startT), 5)
                print(str(c1+c2+c3+1) + " cascades completed so far!\n" + 
                      str(round((time()-startT), 5)) + " secs\n" + 
                      str(round((time()-startTT), 5)) + " secs total\n")
                values.append(())
                
            measureTimeRet2(cascade, g, S, its)
            
def plotComparison(vals, cols):
    labels = [str(u*0.1) for u in range(11)]
    figs, axs = plt.subplots(figsize=(12,6))
    for v in range(len(vals)):
        axs.plot(vals[v], label="QF=" + labels[v], color=cols[v])
    axs.legend()
#"""
print("")


# In[ ]:


#seed selection model comparison unfinished/erroneous methods

#old method of comparing seed selection models' printed results
"""
#Selects seeds with a given model, uses those seeds with each cascade model
#  and prints the resultant spreads.
def compareSeedSelMods(qty, seedMods):
    for g in graphits:
        doneSeeds = set()
        for seedMod in seedMods:
            S, t = measureTimeRet(seedMod[0], g[0], qty)
            print(str(g[0].graph) + " " + seedMod[1] + 
                  "\n" + str(S) + " " + str(t) + " secs\n")
            found = False
            for check in doneSeeds:
                if S in check:
                    print(seedMod[1] + " has the same seed set as " + check[1])
                    found = True
            doneSeeds.add((frozenset(S), seedMod[1]))
            if not found:
                for propMod in propMods:
                    try:
                        measureTime2(propMod[1], propMod[0], g[0], S, g[1])
                    except Exception as e:
                        print(e)
                        
qty = 8
seedMods = [(degDiscSeeds1, "degreeDiscount1"), (degDiscSeeds2, "degreeDiscount2"),
            (degCSeeds, "degreeCentrality"), (inDegCSeeds, "inDegreeCentrality"), 
            (outDegCSeeds, "outDegreeCentrality"), (ccSeeds, "ClosenessCentrality"), 
            (infCSeeds, "infoCentrality"), (btwnCSeeds, "BetweennessCentrality"),
            (approxCfBtwnCSeeds, "approxCF-BetweennessCentrality"), (loadCSeeds, "loadCentrality"), 
            (evCSeeds, "eigenvector"), (kCSeeds, "katz"), 
            (subgCSeeds, "subgraph"), (harmCSeeds, "harmonic"),
            (voteRankSeeds, "voteRank"), (pageRankSeeds, "pageRank"),
            (hitsHubSeeds, "HITS Hubs"), (hitsAuthSeeds, "HITS Auths")]
compareSeedSelMods(qty, seedMods)

#Special case for mixed greedy models
#25 iterations for MixedGreedy 1.1, 1.2, 2.1 & 2.2 took a very long time
#  in G2, so it was run with both 10 and 25 iterations for comparison.
qty, its1, its2 = 8, 25, [10, 25]
graphits1, graphits2 = [(G1, 1000)], [(G2, 500)]
seedMods = [(mixedGreedy11, "Mixed1.1"), (mixedGreedy12, "Mixed1.2"), 
            (mixedGreedy21, "Mixed2.1"), (mixedGreedy22, "Mixed2.2")]

#Seed selection and propagation with those seeds for MixedGreedy models in G1
for g in graphits1:
    for seedMod in seedMods:
        S, t = measureTimeRet(seedMod[0], g[0], qty, its)
        print(str(g[0].graph) + " " + seedMod[1] + 
              "\n" + str(S) + " " + str(t) + " secs\n")
        for propMod in propMods:
            measureTime2(propMod[1], propMod[0], g[0], S, g[1])

#Seed selection and propagation with those seeds for MixedGreedy models in G2
its = [10, 25]
for g in graphits2:
    for it in its:
        for seedMod in seedMods:
            S, t = measureTimeRet(seedMod[0], g[0], qty, it)
            print(str(g[0].graph) + " " + seedMod[1] + " " + str(it) + 
                  " iterations\n" + str(S) + " " + str(t) + " secs\n")
            for propMod in propMods:
                measureTime2(propMod[1], propMod[0], g[0], S, g[1])
#"""
#unfinished comparison of all models on one graph
"""
def compareAllBar(lis, vals):
    labels = [label[1] for label in lis[1]]
    fig, ax = plt.subplots()
    y = np.arange(len(lis[1]))
    height = 0.4
    ax.grid(zorder=0)
    spreads = ax.barh(y - height/2, seedModels[0], height=height, 
                      label='Spread', zorder=3)
    spreadsT = ax.barh(y + height/2, seedModels[1], height=height, 
                       label='Spread / (Time*0.1)', zorder=3)
    
    ax.set_xlabel('Spread')
    ax.set_yticks(y)
    ax.set_yticklabels(lis[1])
    ax.set_title('Spreads of various models')
    ax.legend(loc=0)
        
    fig.tight_layout()
    
allSeeds = priorSeeds + netSeeds + ogSeeds
prepareBar(allSeeds, rndmGraphs)
allSeeds = (degDiscSeeds, 'degDisc') + netSeeds
#"""
#old comparison methods that were improved upon
"""
x, y = ['Graphs', ['ogGreedy','celf','impGreedy','degDisc']], ['Spread', []]
gtest, seedsels = graphs['BitcoinOTC'], [ogGreedySeeds(mockGraphs["mock4-random2"],  4, 100, 'IC'),
                                         celfSeeds(mockGraphs["mock4-random2"], 4, 100, 'IC'),
                                         impGreedySeeds(mockGraphs["mock4-random2"], 4, 100),
                                         degDiscSeeds(mockGraphs["mock4-random2"], 4)]
for s in range(len(seedsels)):
    st = time()
    y[1].append(cascade(mockGraphs["mock4-random2"], seedsels[s], 1000))
    print(x[1][s] + " = " + str(round((time()-st), 4)))
vertBar(x, y, "Seed Select Models")

x = ['Models', ]
qty, its, model, its2, timeFactor = 4, 1000, 'IC', 1000, 1

for g in mockGraphs:
    y, seedTimes = ['Spread', []], []
    for c, seedSel in enumerate(x[1]):
        if c > 2:
            t = time()
            seeds = seedSel[0](mockGraphs[g], qty)
            t = time() - t
            if t < 0.001:
                t = 0.001
            seedTimes.append(t)
        elif c > 1:
            t = time()
            seeds = seedSel[0](mockGraphs[g], qty, its2)
            t = time() - t
            if t < 0.001:
                t = 0.001
            seedTimes.append(t)
        else:
            t = time()
            seeds = seedSel[0](mockGraphs[g], qty, its, model)
            t = time() - t
            if t < 0.001:
                t = 0.001
            seedTimes.append(t)
        y[1].append(cascade(mockGraphs[g], seeds, its))
    
    xLabels = ['Models', [seedMod[1] for seedMod in x[1]]]
    print(y)
    vertBar(xLabels, y, (g + " Seed Select Models: Spread"))

    print(seedTimes)
    for seedSpread in range(len(y[1])):
        y[1][seedSpread] = y[1][seedSpread] / (timeFactor * seedTimes[seedSpread])
    print(y)
    vertBar(xLabels, y, (g + " Seed Select Models: Spread / Time"))
    

#"""


# In[ ]:


#Plotting bar charts for seed selection models

#Vertical bar chart for each seed select model on one graph
"""
def vertBar(lis, vals, msg):
    #return nothing if lists aren't same size
    if len(lis[1]) != len(vals[1]):
        print("Error, not the same size")
        return
    #subplot set up, gridlines drawn, max value calculated and y-limits set
    fig, ax = plt.subplots(1, 1, figsize=(16,len(vals[1])))
    ax.grid(zorder=0)
    topVal = max(vals[1])
    ax.set_ylim([0, topVal*1.25])
    #bar chart plotted
    bars = ax.bar(lis[1], vals[1], width=0.4, facecolor='lightsteelblue', 
                  edgecolor='black', linewidth=2.5, zorder=3)
    #ax.bar_label(bars, fmt='%.3f')
    #Subtitle, x-labels & y-labels are set for each axis
    ax.set_xlabel(lis[0], fontsize=20)
    ax.set_ylabel(vals[0], fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
    #Titles are set and the layout (incl. padding/gaps) is set and adjusted
    fig.tight_layout(pad=5)
    fig.suptitle(msg + " Comparison:", fontsize=24, fontweight='bold')
    fig.subplots_adjust(top=0.88)
#"""


# In[ ]:


#Time testing simple inequality functions, as practice for
#  timing other functions

"""
#Time function
def timeFunc(func, its, a, b, count):
    startTime = time()
    for _ in range(its):
        count = func(a, b, count)
    return count, (time() - startTime)

#Method1
def notEq1(a, b, count):
    if a != b:
        count += 1
    return count

#Method2
def notEq2(a, b, count):
    if not a == b:
        count += 1
    return count

#Method3
def notEq3(a, b, count):
    if a is not b:
        count += 1
    return count

#Method4
def notEq4(a, b, count):
    if a == b:
        cdefg=None
    else:
        count+=1
    return count

#Dictionary for results
eqMethod = {}
for i in range(4):
    eqMethod[i+1] = [['True', 0], 
                     ['False', 0], 
                     ['Uniterable', 0], 
                     ['Iterable', 0]]
#Testing loops
for num1, (aa,bb) in enumerate([(1200, 7),
                              ("Blah", "Yoyoyoyoyoyoyoyo"), 
                              (None, True), 
                              ([1,2,3],[7,8,9]), 
                              ({1,2,3,4,5}, {4,5,6,7}), 
                              (('abc', 123), ('abc', 125))]):
    for x in range(2):
        #print("Params #" + str(num1+1) + " " + str(aa==bb) + ":\n" 
        #      + str(aa) + " " + str(bb) + "\n")
        for num2, f in enumerate([notEq1, notEq2, notEq3, notEq4]):
            qty = 0
            qty, t = timeFunc(f, 10000000, aa, bb, qty)
            index = 0
            if x:
                index += 1
            if num1 > 1:
                index += 2
            eqMethod[num2+1][index][1] += t
        bb = aa
        
#Print results
for a in eqMethod:
    print("Method " + str(a))
    print(eqMethod[a][0])
    print(eqMethod[a][1])
    print(eqMethod[a][2])
    print(eqMethod[a][3])
    print('\n')
for b in eqMethod:
    print("Types of tests:")
    print(eqMethod[0][b])
    print(eqMethod[1][b])
    print(eqMethod[2][b])
    print(eqMethod[3][b])
"""
print("")


# In[ ]:




