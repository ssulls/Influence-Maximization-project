#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#  ALL NECESSARY IMPORTS ::


# In[ ]:


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
    #BitcoinOTC dataset (5881 nodes, 35592 edges)
    #(directed, weighted, signed)
    "BitcoinOTC": (True, True, 
         r"D:\Sully\Documents\Computer Science BSc\Year 3\Term 2\Individual Project\datasets\soc-sign-bitcoinotc.csv"),
    #Facebook dataset (4039 nodes, 88234 edges)
    #(undirected, unweighted, unsigned)
    "Facebook": (False, False, 
         r"D:\Sully\Documents\Computer Science BSc\Year 3\Term 2\Individual Project\datasets\facebook.csv")
}


# In[ ]:


#  NETWORK GRAPH SETUP ::
#
#  Functions to generate network graphs from various csv files,
#  and assign meaningful attributes to the nodes/edges to save
#  processing time during propagation.
#
#  Datasets/graphs included:
#  1. soc-BitcoinOTC
#  2. ego-Facebook


# In[ ]:


#Removes any unconnected components of a given graph
def removeUnconnected(g):
    components = sorted(list(nx.weakly_connected_components(g)), key=len)
    while len(components)>1:
        component = components[0]
        for node in component:
            g.remove_node(node)
        components = components[1:]


# In[ ]:


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


# In[ ]:


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
mockG, name = nx.DiGraph(), "mock1"
testedges = [(1,2), (2,4), (2,5), (2,6), (3,5), (4,5), (5,9), (5,10), (6,8),
            (7,8), (8,9)]
mockG.add_edges_from(testedges)
nx.set_edge_attributes(mockG, 1, 'trust')
mockGraphs[name], diGraphs[name] = mockG, mockG

#Medium-sized, randomly generated directed, unweighted graph
mockG, name = nx.DiGraph(), "mock2"
for i in range(50):
    for j in range(10):
        targ = np.random.randint(-40,50)
        if targ > -1:
            mockG.add_edge(i, targ, trust=1)
mockGraphs[name], rndmGraphs[name], diGraphs[name] = mockG, mockG, mockG

#Medium-sized, randomly generated directed, randomly-weighted graph
mockG, name = nx.DiGraph(), "mock3"
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


# In[ ]:


#Calculate the logs of the degree-reciprocals for all nodes in a graph,
#  and assign them as node attributes to that graph (WC1)
def assignRecips(g):
    drs = {}
    for target in g:
        if not g.in_degree(target):
            continue
        drs[target] = log(1 / g.in_degree(target))
    elMax = drs[max(drs, key=drs.get)]
    elMin = drs[min(drs, key=drs.get)]
    drs = mmNormalizeDict(drs, elMax, elMin)
    nx.set_node_attributes(g, drs, "degRecip")

def assignRecips2(g):
    drs = {}
    for target in g:
        if not g.in_degree(target):
            continue
        drs[target] = ((1 / g.in_degree(target)) ** (1/2))
    nx.set_node_attributes(g, drs, "degRecip")
    
def assignRecips3(g):
    drs = {}
    for target in g:
        if not g.in_degree(target):
            continue
        drs[target] = ((1 / g.in_degree(target)) ** (1/3))
    nx.set_node_attributes(g, drs, "degRecip")
    
#Calculate the logs of the relational-degrees for all edges in a graph,
#  and assign them as edge attributes to that graph (WC2)
def assignRelDegs(g):
    rds = {}
    for target in g:
        if not g.in_degree(target):
            continue
        snd = 0
        for targeting in g.predecessors(target):
            snd += g.out_degree(targeting)
        for targeting in g.predecessors(target):
            rds[(targeting, target)] = log(g.out_degree(targeting) / snd)
    elMax = rds[max(rds, key=rds.get)]
    elMin = rds[min(rds, key=rds.get)]
    rds = mmNormalizeDict(rds, elMax, elMin)
    nx.set_edge_attributes(g, rds, "relDeg")
    
def assignRelDegs2(g):
    rds = {}
    for target in g:
        if not g.in_degree(target):
            continue
        snd = 0
        for targeting in g.predecessors(target):
            snd += g.out_degree(targeting)
        for targeting in g.predecessors(target):
            rds[(targeting, target)] = ((g.out_degree(targeting) / snd) ** (1/2))
    nx.set_edge_attributes(g, rds, "relDeg")

def assignRelDegs3(g):
    rds = {}
    for target in g:
        if not g.in_degree(target):
            continue
        snd = 0
        for targeting in g.predecessors(target):
            snd += g.out_degree(targeting)
        for targeting in g.predecessors(target):
            rds[(targeting, target)] = ((g.out_degree(targeting) / snd) ** (1/3))
    nx.set_edge_attributes(g, rds, "relDeg")


# In[ ]:


#  MISCALLANEOUS & UTILITY METHODS/FUNCTIONS ::
#
#  Functions for printing some/all of the maximum, minimum, mean,
#  median and range of a given list. For comparison of normalization
#  techniques.


# In[ ]:


#Print the mean, median, maximum and minimum of a given list
def printResults(lis, msg, space):
    print(msg)
    print("Mean = " + str(round(np.mean(lis), 5)))
    print("Median = " + str(round(np.median(lis), 5)))
    print("Max = " + str(round(max(lis), 5)))
    print("Min = " + str(round(min(lis), 5)))
    print("Range = " + str(round((max(lis)-min(lis)), 5)))
    if space:
        print("")
        
#Print the maximum, minimum and averages of a given list 
#(more concisely; for larger comparisons)
def printResults1(lis, msg, space):
    print(msg)
    print("Mean = " + str(round(np.mean(lis), 5))
          + ". Median = " + str(round(np.median(lis), 5)) 
          + "\nMax = " + str(round(max(lis), 5)) + ". Min = "
          + str(round(min(lis), 5)) + ". Range = " 
          + str(round((max(lis)-min(lis)), 5)) 
          + "\n---------------------------------")
    if space:
        print("")
        
#Print the mean and median of a given list
#(more concisely; for more specific, larger comparisons)
def printResults2(lis, msg, space):
    print(msg)
    print("Mean = " + str(round(np.mean(lis), 5))
          + ". Median = " + str(round(np.median(lis), 5)))
    if space:
        print("")


# In[ ]:


#  NETWORK-WIDE ANALYSIS ::
#
#  Analyses:
#  1. Strongly/Weakly Connected components - these are subgraphs or sections of the network where:
#     -every node can reach every other node (strongly connected)
#     -every node is reachable from some other node (weakly connected)
#  2. Mutuality percentage (fraction of edges that are bidirectional)
#  3. Density percentage (actual edges / possible edges)
#  4. Percentage of nodes with no incoming/outgoing edges


# In[ ]:


#Returns the mutuality-percentage of a given graph
#(how many of the edges are parallel/bi-directional)
def strongWeak(g):
    weak = len(list(nx.weakly_connected_components(g)))
    strong = len(list(nx.strongly_connected_components(g)))
    return round((weak/strong)*100, 5)

#Results are printed and compared for every directed graph
"""
for g in diGraphs:
    print(g + "\n# of weak components / # of weak components:\n" 
          + str(strongWeak(diGraphs[g])) + "%\n")
#"""
print("")


# In[ ]:


#Returns the mutuality-percentage of a given graph
#(how many of the edges are parallel/bi-directional)
def mutuality(g):
    edgeSet = set(g.edges)
    count = 0
    for (u,v) in edgeSet:
        if (v,u) in edgeSet:
            count += 1
    return round((count/g.size())*100, 5)

#Results are printed and compared for every directed graph
"""
for g in diGraphs:
    print(g + ": mutuality: " + str(mutuality(diGraphs[g])) + "%")
#"""
print("")


# In[ ]:


#Returns the density-percentage of a given graph
#(how many possible edges are actually present)
def density(g):
    nodeCount = len(g)
    return round((g.size()/((nodeCount*(nodeCount-1))/2))*100, 5)

#Results are printed and compared for every graph
"""
for g in allGraphs:
    print(g + " density: " + str(density(allGraphs[g])) + "%")
#"""
print("")


# In[ ]:


#Returns percentage of nodes that have no incoming edges
def noIncoming(g):
    return round((len([node for node in g if not g.in_degree(node)])/len(g))*100, 5)

#Returns percentage of nodes that have no outgoing edges
def noOutgoing(g):
    return round((len([node for node in g if not g.out_degree(node)])/len(g))*100, 5)

#Results are printed and compared for every directed graph
"""
for g in diGraphs:
    print(g + " nodes with no incoming edges: " 
          + str(noIncoming(diGraphs[g])) + "%")
print("")    
for g in diGraphs:
    print(g + " nodes with no outgoing edges: " 
          + str(noOutgoing(diGraphs[g])) + "%")
#"""
print("")


# In[ ]:


#  DEGREE-RELATED ANALYSIS FUNCTIONS ::
#
#  These tests specifically analyse the networks' degrees, degree reciprocals 
#  and relational degrees (used to calculate certain propagation probabilities), 
#  as well as various normalization or scaling techniques applied to them.
#
#  This was to rectify the issue I encountered with my WC1 & WC2 models, 
#  that the propagation probabilities were too low across the whole dataset, 
#  due to the high number of edges and the wide range of nodes' degrees.
#
#  My aim was to find a spread of the probabilities whereby:
#  the mean falls between 0.25-0.75, but the key relationships are kept intact.
#
#  The key relationships being between the propagation probability and:
#  -the in_degree of the target node (WC1).
#  -the out_degree of the targeting node, relative to the out_degrees of all of the target node's neighbours (WC2).
#
#  Functions:
#  1. In-degrees & Out-degrees
#  2. Degree reciprocals for WC1
#  3. Relational degrees for WC2 (incl. sum of all neighbours' degrees)
#  4. Incorporating propagation variables (pp & qf)
#  5. Root normalization/scaling
#  6. Min-Max normalization/scaling
#  7. Max-normalization/scaling
#  8. Z-score normalizations (incl. adjust-scaling)
#  9. Robust/interquartile normalization
#  10. Log-scaling


# In[ ]:


#Return the in_degrees and out_degrees for all nodes in a graph:
def degsList(g, weighted=False):
    inDegs = []
    outDegs = []
    for node in g:
        if weighted:
            inDegs.append(g.in_degree(node, weight='trust'))
            outDegs.append(g.out_degree(node, weight='trust'))
        else:
            inDegs.append(g.in_degree(node))
            outDegs.append(g.out_degree(node))
    return (inDegs, outDegs)

#Calculate and return the degree-reciprocals for all nodes in a graph (WC1):
def calcRecips(g):
    recips = []
    for node in g:
        if not g.in_degree(node):
            continue
        recips.append(1/(g.in_degree(node)))
    return recips

#Calculate and return the relational-degrees for all edges in a graph (WC2):
def calcRelDegs(g, weighted=False):
    relDegs = []
    for target in g:
        if not g.in_degree(target):
            continue
        #sum of target's neighbours' out_degrees
        snd = 0
        for neighbour in g.predecessors(target):
            snd += g.out_degree(neighbour)
        #relational degrees calculated
        for targeting in g.predecessors(target):
            if weighted:
                relDegs.append((g.out_degree(targeting)/snd)*g[targeting][target]['trust'])
            else:
                relDegs.append(g.out_degree(targeting)/snd)
    return relDegs

#Probability calculating functions are compiled into a list
probFuncs = [(calcRecips, "Degree Reciprocals"), (calcRelDegs, "Relational Degrees")]

#Averages, maximums and minimums are printed
#"""
for g in graphs:
    print(str(g))
    #indegs, outdegs = degsList(diGraphs[g])
    #printResults1(indegs, "Unweighted in-degrees", False)
    #printResults1(outdegs, "Unweighted out-degrees", False)
    #indegs, outdegs = degsList(diGraphs[g], True)
    #printResults1(indegs, "Weighted in-degrees", False)
    #printResults1(outdegs, "Weighted out-degrees", False)
    printResults1(calcRecips(diGraphs[g]), "Degree reciprocals", False)
    printResults1(calcRelDegs(diGraphs[g]), "Unweighted relational degrees", False)
    printResults1(calcRelDegs(diGraphs[g], True), "Weighted relational degrees", True)
    print("")
#"""
print("")


# In[ ]:


#Multiply all values in a list by a quality factor qf
def varsList(lis, qf=0.7):
    return list(map(lambda x : x*qf, lis))

#Returns a list of all the trust values from all edges of a given graph, 
#  multiplied by pp.
#For comparison with Independent Cascade probabilities
def icProb(g, pp=0.2):
    icprobs = []
    for (u,v,t) in g.edges.data('trust'):
        if not t:
            continue
        icprobs.append(t * pp)
    return icprobs


# In[ ]:


#Convert all elements in a list to a given root
def rootList(lis, root):
    return list(map(lambda x : x**root, lis))

#Direct rooting of degree-reciprocals in a given graph, up to a given number k:
def recipsRoots(g, k):
    recips = calcRecips(g)
    probs = []
    for i in range(1, k+1):
        probs.append(round(np.mean(rootList(recips, (1/i))), 5))
        #print("Average degRecip prob to the power of " + str(i) + " = " + str(prob))
    return probs

#Direcct rooting of relational-degrees in a given graph, up to a given number k:
def relDegsRoots(g, k):
    relDegs = calcRelDegs(g)
    probs = []
    for i in range(1, k+1):
        probs.append(round(np.mean(rootList(relDegs, (1/i))), 5))
        #print("Average RelDeg prob to the power of " + str(i) + " = " + str(prob))
    return probs

#Averages, maximums and minimums are printed
"""
for probFunc in probFuncs:
    print(probFunc[1] + ":\n")
    for g in diGraphs:
        print(g)
        probs = probFunc[0](diGraphs[g])
        for i in range(1,5):
            printResults1(rootList(probs, 1/i), ("Rooted by " + str(i)), True)
    print("")
"""
print("")


# In[ ]:


#Min-max normalization of a given list
#(a normalization technique itself, but can also be combined with
# other techniques to scale the values between 0 and 1.)
def mmNormalize(lis):
    elMax = max(lis)
    elMin = min(lis)
    return list(map(lambda x : ((x - elMin) / (elMax - elMin)), lis))

#Direct min-max normalization of degree-reciprocals:
def mmNormalizeDegRec(g):
    degRecs = calcRecips(g)
    norDegRecs = []
    for dr in degRecs:
        norDegRecs.append((dr - min(degRecs)) / (max(degRecs) - min(degRecs)))
    return (degRecs, norDegRecs)

#Direct min-max normalization of relational-degrees:
def mmNormalizeRelDeg(g):
    relDegs = calcRelDegs(g)
    norRelDegs = []
    for rd in relDegs:
        norRelDegs.append((rd - min(relDegs)) / (max(relDegs) - min(relDegs)))
    return (relDegs, norRelDegs)

#Averages, maximums and minimums are printed
"""
for probFunc in probFuncs:
    print(probFunc[1] + ":\n")
    for g in diGraphs:
        print(g)
        printResults1(mmNormalize(probFunc[0](diGraphs[g])), "Min-max normalized", True)
    print("")
"""
print("")


# In[ ]:


#Normalization by dividing every element in a given list by the maximum value
def maxNormalize(lis):
    elMax = max(lis)
    if not elMax:
        elMax = 0.000000001
    return list(map(lambda x : x/elMax, lis))

#Averages, maximums and minimums are printed
"""
for probFunc in probFuncs:
    print(probFunc[1] + ":\n")
    for g in diGraphs:
        print(g)
        printResults1(maxNormalize(probFunc[0](diGraphs[g])), 
                      "Max normalized", True)
    print("")
"""
print("")


# In[ ]:


#Z-score normalization of a given list
def zNormalize(lis):
    mean = np.mean(lis)
    meanSqs = list(map(lambda x : ((x - mean) ** 2), lis))
    stanDev = np.mean(meanSqs) ** (1/2)
    zScores = list(map(lambda x : ((x - mean) / stanDev), lis))
    return zScores

#Returns mean and standard deviation of a given list
def standardDev(lis):
    mean = np.mean(lis)
    meanSqs = list(map(lambda x : ((x - mean) ** 2), lis))
    stanDev = np.mean(meanSqs) ** (1/2)
    return mean, stanDev

#Scale a given list to between 0 and 1
def adjust(lis):
    elMax = max(lis)
    elMin = abs(min(lis))
    return list(map(lambda x : ((x + elMin) / (elMax + elMin)), lis))

#Averages, maximums and minimums are printed
"""
for probFunc in probFuncs:
    print(probFunc[1] + ":\n")
    for g in diGraphs:
        print(g)
        printResults1(zNormalize(probFunc[0](diGraphs[g])), "Z-score normalized", True)
    print("")
"""
print("")


# In[ ]:


#Robust normalization using interquartile range
def robustNormalize(lis):
    median = np.median(lis)
    q75, q25 = np.percentile(lis, [75, 25])
    iqr = q75 - q25
    return list(map(lambda x : ((x - median) / iqr), lis))

#Averages, maximums and minimums are printed
"""
for probFunc in probFuncs:
    print(probFunc[1] + ":\n")
    for g in graphs:
        print(g)
        printResults1(robustNormalize(probFunc[0](diGraphs[g])), "Robust normalized", True)
    print("")
#"""
print("")


# In[ ]:


#Log-scale a given list
def logList(lis):
    return list(map(lambda x : log(x), lis))

#Averages, maximums and minimums are printed
"""
for probFunc in probFuncs:
    print(probFunc[1] + ":\n")
    for g in diGraphs:
        print(g)
        printResults1(logList(probFunc[0](diGraphs[g])), "Log-scaled", True)
    print("")
"""
print("")


# In[ ]:


#  GRAPHING & COMPAING NETWORK-WIDE PROBABILITIES ::
#
#  Functions:
#  1. Plot pie chart
#  2. Plot histograms comparing probabilities from normalization techniques
#     for a single graph.
#  3. Plot histograms comparing probabilities for a given list of graphs,
#     comparing at each normalization technique.
#  4. Plot histograms comparing probabilities for a given list of graphs,
#     comparing at each normalization technique, with a given list of 
#     normalization functions.


# In[ ]:


#Plot pie chart for probability distribution
def compareNetworksPie1(g, gname, func):
    probs, frac, explode = func[0](g), [[], [], [], [], []], (0.1, 0.1, 0.1, 0.1, 0.1)
    for prob in probs:
        if prob < 0.2:
            frac[0].append(round(prob, 1))
        elif prob < 0.4:
            frac[1].append(round(prob, 1))
        elif prob < 0.6:
            frac[2].append(round(prob, 1))
        elif prob < 0.8:
            frac[3].append(round(prob, 1))
        else:
            frac[4].append(round(prob, 1))
    fracnames, values = [(str(round(i*0.2, 1)) + " -- " + str((round(((i*0.2)+0.2), 1)))) for i in range(5)], [len(p) for p in frac]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.grid(zorder=0)
    ax.pie(values, labels=fracnames, explode=explode, autopct='%1f%%')
    fig.suptitle(gname + " - " + func[1] + " Probability Distribution:", fontsize=20, fontweight='bold')
    fig.subplots_adjust(top=0.88)

#"""
for gs in [graphs]:
    for g in gs:
        for probFunc in probFuncs:
            compareNetworksPie1(gs[g], g, probFunc)
#"""
print("")


# In[ ]:


#Plot pie chart for probability distribution #2
def compareNetworksPie2(g, gname, func):
    probs, frac, explode = func[0](g), [[], [], [], [], []], (0.1, 0.1, 0.1, 0.1, 0.1)
    for prob in probs:
        if prob < 0.2:
            frac[0].append(round(prob, 1))
        elif prob < 0.4:
            frac[1].append(round(prob, 1))
        elif prob < 0.6:
            frac[2].append(round(prob, 1))
        elif prob < 0.8:
            frac[3].append(round(prob, 1))
        else:
            frac[4].append(round(prob, 1))
    fracnames, values = [(str(round(i*0.2, 1)) + " -- " + str((round(((i*0.2)+0.2), 1)))) for i in range(5)], [len(p) for p in frac]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(zorder=0)
    wedges = ax.pie(values, labels=fracnames, 
                           wedgeprops=dict(width=0.5),
                           autopct='%1f%%', startangle=-30)
    #
    bbox_props = dict(boxstyle="square,pad=0.3", 
                      fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"), 
              bbox=bbox_props, zorder=0, va='center')
    
    ax.legend(wedges, ingredients)
    
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)
    
    fig.suptitle(gname + " - " + func[1] + " Probability Distribution:", fontsize=16, fontweight='bold')
    fig.subplots_adjust(top=0.88)

#"""
for gs in [graphs]:
    for g in gs:
        for probFunc in probFuncs:
            compareNetworksPie2(gs[g], g, probFunc)
#"""
print("")


# In[ ]:


#General function to plot a bar chart for a 
#  list of names against a list of values
def generalBar(lis, vals):
    if len(lis[1]) != len(vals[1]):
        print("Error, not the same size")
        return
    fig, ax = plt.subplots(1, 1, figsize=((len(lis[1])*2.5),6))
    ax.grid(zorder=0)
    topVal = max(vals[1])
    ax.set_ylim([0, topVal*1.25])
    ax.bar(lis[1], vals[1], width=0.4, facecolor=colors, 
           edgecolor='black', linewidth=2.5, zorder=3)
    for v in range(len(vals[1])):
        if vals[1][v] == max(vals[1]):
            try:
                label = AnchoredText(("Maximum = " + lis[1][v]) + 
                                     "\nMean = " + str(round(np.mean(vals[1]), 3)))
                ax.add_artist(label)
            except Exception as e:
                print(e)
    for count, (xbar,ybar) in enumerate(zip(lis[1], vals[1])):
        if ybar/topVal < 0.3:
            y = ybar + (max(values) * 0.05)
        else:
            y = ybar*0.5
        ax.annotate(ybar, xy=((0.5)*(2*count), y), 
                        rotation=90, ha='center', fontsize=16)
    #Subtitle, x-labels & y-labels are set for each axis
    ax.set_xlabel(lis[0], fontsize=15)
    ax.set_ylabel(func[0] + " (%)", fontsize=15)
    #Titles are set and the layout (incl. padding/gaps) is set and adjusted
    fig.tight_layout(pad=5)
    fig.suptitle(func[1] + " Comparison:", fontsize=24, fontweight='bold')
    fig.subplots_adjust(top=0.88)


# In[ ]:


#Plot bar charts for every metric for one list of graphs
def compareNetworksBar1(graphlist, func):
    #
    labels, values = [], []
    for g in graphlist:
        labels.append(g)
        values.append(func[0](graphlist[g]))
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    ax.grid(zorder=0)
    topVal = max(values)
    ax.set_ylim([0, topVal*1.38])
    ax.bar(labels, values, width=0.3, 
           facecolor='lightsteelblue', edgecolor='black', 
           linewidth=2.5, zorder=3)
    label = AnchoredText(("Mean = " + str(round(np.mean(values), 3)) + 
                          "\nMedian = " + str(round(np.median(values), 3))), 
                         loc=1, prop=dict(size=10))
    ax.add_artist(label)
            #An annotation displaying each bar's value is created and
            #  relatively positioned in each column
    for count, (xbar,ybar) in enumerate(zip(labels, values)):
        if ybar/topVal < 0.3:
            y = ybar + (max(values) * 0.05)
        else:
            y = ybar*0.5
        ax.annotate(round(ybar, 2), xy=((0.5)*(2*count), ybar+(topVal*0.05)), 
                    #rotation=90, 
                    ha='center', fontsize=14)
    #Subtitle, x-labels & y-labels are set for each axis
    ax.set_xlabel('Graphs', fontsize=15)
    ax.set_ylabel(func[1] + " (%)", fontsize=15)
    #Titles are set and the layout (incl. padding/gaps) is set and adjusted
    fig.tight_layout(pad=5)
    fig.suptitle(func[1] + " Comparison:", fontsize=20, fontweight='bold')
    fig.subplots_adjust(top=0.88)


#List of lists of graphs with labels are compiled
glistlist = [graphs, diGraphs]
#Lists of functions with labels are compiled
fs = [(mutuality, "Mutuality"), (density, "Density"), 
      (strongWeak, "# of weak comps / # of strong comps"),
      (noIncoming, "Nodes with no incoming edges"), 
      (noOutgoing, "Nodes with no outgoing edges")]

#Plotting graphs to compare network-wide metrics of all graphs
#"""
for gs in glistlist:
    for f in fs:
        compareNetworksBar1(gs, f)
#"""
print("")


# In[ ]:


#Plot and display bar charts of given network-wide metrics, 
#  comparing a given list of lists of graphs
def compareNetworksBar2(graphlistlist, funclist):
    #Network metric values and labels are calculated for every graph in a given
    #  list of lists of graphs and compiled into 2 lists
    labels, values = [[g for g in graphlist] for graphlist in graphlistlist], []
    for graphlist in graphlistlist:
        values.append([[func[0](graphlist[g]) for g in graphlist] for func in funclist])
    #Subplots are created, as wide as the number of metrics and as tall as
    #  the number of different lists of graphs
    figs, axs = plt.subplots(len(funclist), len(graphlistlist), figsize=(15,30), sharey=False)
    for f in range(len(funclist)):
        for g in range(len(graphlistlist)):
            #Gridlines are drawn behind the graph
            axs[f, g].grid(zorder=0)
            #Width of bars is adjusted based on the length of the current graph list
            barwidth = (len(graphlistlist[g]))*0.1
            while barwidth > 1:
                barwidth *= 0.5
            #Values are assigned from the array to prevent repetitive nested array access
            vals = values[g][f]
            #Maximum value is calculated and y-limits are adjusted to more than that,
            #  to reserve space for a text boxt in the upper-right
            topVal = max(vals)
            axs[f, g].set_ylim([0, topVal*1.38])
            #Bar chart is plotted with customised visual settings
            axs[f, g].bar(labels[g], vals, width=(barwidth), 
                          facecolor='lightsteelblue', edgecolor='black', 
                          linewidth=2.5, zorder=3)
            #Mean and median are calculated and displayed in a text-box
            label = AnchoredText(("Mean = " + str(round(np.mean(vals), 3)) + 
                                  "\nMedian = " + str(round(np.median(vals), 3))), 
                                 loc=1, prop=dict(size=10))
            axs[f, g].add_artist(label)
            #An annotation displaying each bar's value is created and
            #  relatively positioned in each column
            for count, (xbar,ybar) in enumerate(zip(labels[g], vals)):
                if ybar/topVal < 0.3:
                    y = ybar + (max(vals) * 0.05)
                else:
                    y = ybar*0.5
                axs[f, g].annotate(round(ybar, 2), xy=((0.5)*(2*count), ybar+(topVal*0.05)), 
                                   ha='center', fontsize=12)
            #Subtitle, x-labels & y-labels are set for each axis
            axs[f, g].set_title((funclist[f][1] + ": "), fontsize=20)
            axs[f, g].set_xlabel('Graphs', fontsize=15)
            axs[f, g].set_ylabel("Metric (%)", fontsize=15)
    #Titles are set and the layout (incl. padding/gaps) is set and adjusted
    figs.tight_layout(pad=5)
    figs.suptitle("Network-wide metrics", fontsize=24, fontweight='bold')
    figs.subplots_adjust(top=0.95)
    
#List of lists of graphs with labels are compiled
gs = [diGraphs, graphs]
#Lists of functions with labels are compiled
fs = [(mutuality, "Mutuality"), (density, "Density"), 
      (strongWeak, "# of weak comps / # of strong comps"),
      (noIncoming, "Nodes with no incoming edges"), 
      (noOutgoing, "Nodes with no outgoing edges")]

#Plotting graphs to compare network-wide metrics of all graphs
#"""
compareNetworksBar2(gs, fs)
#"""
print("")


# In[ ]:


rooted = []
for g in graphs:
    rooted.append(rootList(calcRelDegs(graphs[g], (1/2))))


# In[ ]:


#Histograms of probabilties & normalization techniques #1
#Every normalization technique for one graph
def compareProbsHist1(g, gs, baseFunc):
    #Probabilities are calculated with every technique and compiled into a list
    baseProbs = baseFunc[0](gs[g])
    probs = [(baseProbs, "Base values"), 
             (rootList(baseProbs, (1/2)), "Square rooted"), 
             (rootList(baseProbs, (1/3)), "Cube rooted"), 
             (rootList(baseProbs, (1/4)), "Fourth root"), 
             (mmNormalize(baseProbs), "Min-Max normalized"), 
             (maxNormalize(baseProbs), "Max normalized"),
             (zNormalize(baseProbs), "Z-score normalized"), 
             (robustNormalize(baseProbs), "Interquartile normalized"), 
             (logList(baseProbs), "Log-scaled")]
    #Subplots are created, 2 wide and as tall as the number of different probabilities
    figs, axs = plt.subplots(len(probs), 2, figsize=(15, 50), sharey=False)
    for f in range(len(probs)):
        #Values are scaled to within 0 and 1, if they are not already
        if max(probs[f][0]) > 1 or min(probs[f][0]) < 0:
            probs[f] = (mmNormalize(probs[f][0]), probs[f][1])
        for i in range(2):
            #Values and label are assigned from the array to prevent repetitive nested array access
            #Axes are plotted with two sets of probabilities - once for these values and then 
            #  these values multiplied by a quality factor, alternatively.
            if not i:
                vals, title = probs[f]
                axs[f, i].set_title(g + ": " + title, fontsize=20)
            else:
                vals, title = varsList(probs[f][0]), probs[f][1]
                axs[f, i].set_title(g + ": " + title + 
                                    " (multiplied by qf)", fontsize=20)
            #Gridlines are drawn behind the graph
            axs[f, i].grid(zorder=0)
            #Histogram is plotted with customised visual settings
            bars, bins, _ = axs[f, i].hist(vals, bins=[num*0.1 for num in range(11)], 
                                     facecolor='lightsteelblue', edgecolor='dimgrey', 
                                     linewidth=2.5, rwidth=0.75, zorder=3)
            #Maximum y-value is calculated and y-limits are adjusted to more than that,
            #  to reserve space for a text boxt in the upper-right
            maxBar = max(bars)
            axs[f, i].set_ylim([0, maxBar*1.25])
            #Mean and median are calculated and displayed in a text-box
            label = AnchoredText(("Mean = " + str(round(np.mean(vals), 3)) + 
                                  "\nMedian = " + str(round(np.median(vals), 3))), 
                                 loc=1, prop=dict(size=10))
            axs[f, i].add_artist(label)
            #An annotation displaying each histogram bar's value is created and
            #  relatively positioned in each column
            for count, (xbar,ybar) in enumerate(zip(bins, bars)):
                if ybar/maxBar < 0.3:
                    y = ybar + (maxBar * 0.05)
                elif ybar/maxBar > 0.7:
                    y = ybar*0.75
                else:
                    y = ybar*0.5
                axs[f, i].annotate(round(ybar, 0), xy=(xbar+0.05, y), 
                                   rotation=90, ha='center', fontsize=12)
            #Histogram bins, x-labels & y-labels are set for each axis
            axs[f, i].set_xticks([num*0.1 for num in range(11)])
            axs[f, i].set_xlabel("Probabilities", fontsize=15)
            axs[f, i].set_ylabel("Frequencies", fontsize=15)
    #Titles are set and the layout (incl. padding/gaps) is set and adjusted
    figs.tight_layout(pad=5)
    figs.subplots_adjust(top=0.96, bottom=0.02)
    figs.suptitle(baseFunc[1] + ": Various Normalization Techniques", 
                  fontsize=24, fontweight='bold')
    

#Plotting histograms to compare probabilities with various 
#  normalization techniques for one graph
#"""
for gs in [graphs]:
    for g in gs:
        for probFunc in probFuncs:
            compareProbsHist1(g, gs, probFunc)
#compareProbsHist1(namedGraphs[0], probFuncs[0])
#"""
print("")


# In[ ]:


#Histograms of probabilities & normalization techniques #2a
#Every graph in list, one-after-another
def compareProbsHist2(graphlist, baseFunc):
    #Probabilities are calculated with every technique and compiled into a list
    probs = []
    for g in graphlist:
        baseProbs = baseFunc[0](graphlist[g])
        probs.append([(baseProbs, "Base values"), 
                      (rootList(baseProbs, (1/2)), "Square rooted"), 
                      (rootList(baseProbs, (1/3)), "Cube rooted"), 
                      (rootList(baseProbs, (1/4)), "Fourth root"), 
                      (mmNormalize(baseProbs), "Min-Max normalized"), 
                      (maxNormalize(baseProbs), "Max normalized"),
                      (zNormalize(baseProbs), "Z-score normalized"), 
                      (robustNormalize(baseProbs), "Interquartile normalized"), 
                      (logList(baseProbs), "Log-scaled")])
    #Subplots are created, 2 wide and as tall as the number of different probabilities
    figs, axs = plt.subplots(len(probs)*len(probs[0]), 2, figsize=(15, 130), sharey=False)
    for f in range(len(probs[0])):
        for gc, g in enumerate(graphlist):
            index = (f*len(probs))+gc
            #Values are scaled to within 0 and 1, if they are not already
            if max(probs[gc][f][0]) > 1 or min(probs[gc][f][0]) < 0:
                probs[gc][f] = (mmNormalize(probs[gc][f][0]), (probs[gc][f][1] + " (adjusted)"))
            #Values and label are assigned from the array to prevent repetitive nested array access
            vals, title = probs[gc][f]
            #Axes are plotted twice - once for these values and then these values 
            #  multiplied by a quality factor, alternatively.
            for i in range(2):
                if not i:
                    axs[index, i].set_title(g + ": " + title, fontsize=20)
                else:
                    vals = varsList(vals)
                    axs[index, i].set_title(g + ": " + title + 
                                        " multiplied by qf", fontsize=20)
                #Gridlines are drawn behind the graph
                axs[index, i].grid(zorder=0)
                #Histogram is plotted with customised visual settings
                bars, bins, _ = axs[index, i].hist(vals, bins=[num*0.1 for num in range(11)], 
                                             facecolor='lightsteelblue', edgecolor='dimgrey', 
                                             linewidth=2.5, rwidth=0.75, zorder=3)
                #Maximum y-value is calculated and y-limits are adjusted to more than that,
                #  to reserve space for a text boxt in the upper-right
                maxBar = max(bars)
                axs[index, i].set_ylim([0, maxBar*1.25])
                #Mean and median are calculated and displayed in a text-box
                label = AnchoredText(("Mean probability = " + str(round(np.mean(vals), 3)) + 
                                      "\nMedian probability = " + str(round(np.median(vals), 3))), 
                                     loc=1, prop=dict(size=10))
                axs[index, i].add_artist(label)
                #An annotation displaying each histogram bar's value is created and
                #  relatively positioned in each column 
                for count, (xbar,ybar) in enumerate(zip(bins, bars)):
                    frac = ybar/maxBar
                    if frac < 0.3:
                        y = ybar + (maxBar * 0.05)
                    elif frac > 0.7:
                        y = ybar*0.75
                    else:
                        y = ybar*(frac)
                    axs[index, i].annotate(ybar, xy=(xbar+0.05, y), 
                                           rotation=90, ha='center', fontsize=12)
                #Histogram bins, x-labels & y-labels are set for each axis
                axs[index, i].set_xticks([num*0.1 for num in range(11)])
                axs[index, i].set_xlabel("Probabilities", fontsize=15)
                axs[index, i].set_ylabel("Frequencies", fontsize=15)
    #Titles are set and the layout (incl. padding/gaps) is set and adjusted
    figs.tight_layout(pad=5)
    figs.subplots_adjust(top=0.97, bottom=0.3)
    figs.suptitle(baseFunc[1] + ": Various Normalization Techniques", 
                  fontsize=24, fontweight='bold')    

#Plotting histograms to compare probabilities with various 
#  normalization techniques of all graphs
#"""
for probFunc in probFuncs:
    compareProbsHist2(graphs, probFunc)
#"""
print("")


# In[ ]:


#Histograms of probabilities & normalization techniques #2b
#Graphs alternating
def compareProbsHist2Alternate(graphlist, baseFunc, funclist):
    #Probabilities are calculated with every technique from a given list 
    #  and compiled into a list
    probs = []
    for g in graphlist:
        graphProbs = []
        baseProbs = baseFunc[0](graphlist[g])
        graphProbs.append((baseProbs, "Base values"))
        for func in funclist:
            if len(func) > 2:
                graphProbs.append((func[0](baseProbs, func[1]), func[2]))
            else:
                graphProbs.append((func[0](baseProbs), func[1]))
        probs.append(graphProbs)
    
    #Subplots are created, 2 wide and as tall as the number of different probabilities
    figs, axs = plt.subplots(len(probs)*len(probs[0]), 2, figsize=(15, 130), sharey=False)
    for f in range(len(probs[0])):
        for g, graph in enumerate(graphlist):
            index = (f*len(probs))+g
            #Values are scaled to within 0 and 1, if they are not already
            if max(probs[g][f][0]) > 1 or min(probs[g][f][0]) < 0:
                probs[g][f] = (mmNormalize(probs[g][f][0]), (probs[g][f][1] + " (adjusted)"))
            #Values and label are assigned from the array to prevent repetitive nested array access
            vals, title = probs[g][f]
            #Axes are plotted with two sets of probabilities - once for these values and then 
            #  these values multiplied by a quality factor, alternatively.
            for i in range(2):
                if not i:
                    axs[index, i].set_title(graph + ": " + title, fontsize=20)
                else:
                    vals = varsList(vals)
                    axs[index, i].set_title(graph + ": " + title + 
                                        " multiplied by qf", fontsize=20)
                #Gridlines are drawn behind the graph
                axs[index, i].grid(zorder=0)
                #Histogram is plotted with customised visual settings
                bars, bins, _ = axs[index, i].hist(vals, bins=[num*0.1 for num in range(11)], 
                                             facecolor='lightsteelblue', edgecolor='dimgrey', 
                                             linewidth=2.5, rwidth=0.75, zorder=3)
                #Maximum y-value is calculated and y-limits are adjusted to more than that,
                #  to reserve space for a text boxt in the upper-right
                maxBar = max(bars)
                axs[index, i].set_ylim([0, maxBar*1.25])
                #Mean and median are calculated and displayed in a text-box
                label = AnchoredText(("Mean probability = " + str(round(np.mean(vals), 3)) + 
                                      "\nMedian probability = " + str(round(np.median(vals), 3))), 
                                     loc=1, prop=dict(size=10))
                axs[index, i].add_artist(label)
                #An annotation displaying each histogram bar's value is created and
                #  relatively positioned in each column
                for count, (xbar,ybar) in enumerate(zip(bins, bars)):
                    frac = ybar/maxBar
                    if frac < 0.3:
                        y = ybar + (maxBar * 0.05)
                    elif frac > 0.7:
                        y = ybar*0.8
                    else:
                        y = ybar*(frac)
                    axs[index, i].annotate(ybar, xy=(xbar+0.05, y), 
                                           rotation=90, ha='center', fontsize=12)
                #Histogram bins, x-labels & y-labels are set for each axis
                axs[index, i].set_xticks([num*0.1 for num in range(11)])
                axs[index, i].set_xlabel("Probabilities", fontsize=15)
                axs[index, i].set_ylabel("Frequencies", fontsize=15)
    #Titles are set and the layout (incl. padding/gaps) is set and adjusted
    figs.tight_layout(pad=5)
    figs.subplots_adjust(top=0.97, bottom=0.3)
    figs.suptitle(baseFunc[1] + ": Various Normalization Techniques", 
                  fontsize=24, fontweight='bold')

#List of functions with labels are compiled
fs = [(rootList, (1/2), "Square rooted"), 
      (rootList, (1/3), "Cube rooted"), 
      (rootList, (1/4), "Fourth root"), 
      (mmNormalize, "Min-Max normalized"), 
      (zNormalize, "Z-score normalized"), 
      (robustNormalize, "Interquartile normalized"), 
      (logList, "Log-scaled")]

#Plotting histograms of probabilities with various given 
#  normalization techniques alternating between graphs
#"""
for probFunc in probFuncs:
    compareProbsHist2Alternate(graphs, probFunc, fs)
#"""
print("")


# In[ ]:


#Line-graphs of probabilty spreads & normalization techniques #3
#
def compareProbsLine1(g, gs, baseFunc, funclist, colours):
    #
    probs, ind = [(baseFunc[0](gs[g]), "Base values")], 0
    for c, func in enumerate(funclist):
        if len(func) > 2:
            for par in range(len(func[1])):
                probs.append((func[0](probs[0][0], func[1][par]), func[2][par]))
                ind += 1
        else:
            probs.append((func[0](probs[0][0]), func[1]))
            ind += 1
        #
        if max(probs[ind][0]) > 1 or min(probs[ind][0]) < 0:
            probs[ind] = (mmNormalize(probs[ind][0]), 
                           (probs[ind][1] + " (adjusted)"))
    #
    fracs = [[[] for _ in range(12)] for _ in range(len(probs))]
    for f in range(len(probs)):
        for prob in probs[f][0]:
            added, i = False, 0
            while added == False:
                if prob <= i*0.1:
                    #try:
                    fracs[f][i].append(prob)
                    added = True
                else:
                    i+=1
        #
        for frac in range(len(fracs[f])):
            fracs[f][frac] = len(fracs[f][frac])
    #
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharey=False)
    #
    ax.grid(zorder=0)
    xlabels = [i*0.1 for i in range(12)]
    #
    for i in range(len(probs)):
        ax.plot(xlabels, fracs[i], label=probs[i][1], marker='o', 
                color=colours[i], linewidth=4, zorder=3)
    #
    ax.legend()
    fig.suptitle(g + " " + probFunc[1] + 
                 " Probability Normalization Comparisons:", 
                 fontsize=20, fontweight='bold')

#    
fs = [(rootList, [(1/2), (1/3), (1/4)], 
       ["Square rooted", "Cube rooted", "Fourth root"]), 
      (mmNormalize, "Min-Max normalized"), 
      (zNormalize, "Z-score normalized"), 
      (robustNormalize, "Interquartile normalized"), 
      (logList, "Log-scaled")]
cs = ['red', 'orange', 'lawngreen', 'green', 'aqua', 
      'lightskyblue', 'blue', 'magenta', 'mediumpurple', 'darkkhaki']

#
#"""
for g in graphs:
    for probFunc in probFuncs:
        compareProbsLine1(g, graphs, probFunc, fs, cs)
#"""
print("")


# In[ ]:


def compareProbsLine2(graphlist, baseFunc, funclist, colours, lines):
    #
    allProbs = []
    for g in graphlist:
        gProbs, ind = [(baseFunc[0](graphlist[g]), "Base Values")], 0
        for func in funclist:
            if len(func) > 2:
                for par in range(len(func[1])):
                    gProbs.append((func[0](gProbs[0][0], func[1][par]), func[2][par]))
                    ind += 1
            else:
                gProbs.append((func[0](gProbs[0][0]), func[1]))
                ind += 1
            #
            if max(gProbs[ind][0]) > 1 or min(gProbs[ind][0]) < 0:
                gProbs[ind] = (mmNormalize(gProbs[ind][0]), 
                               (gProbs[ind][1] + " (adjusted)"))
        allProbs.append(gProbs)
    #
    fracs = [[[[] for _ in range(11)] for _ in range(len(allProbs[g]))]
             for g in range(len(allProbs))]
    for g in range(len(allProbs)):
        for f in range(len(allProbs[g])):
            for prob in allProbs[g][f][0]:
                added, i = False, 0
                while added == False:
                    if prob < 0 or prob > 1:
                        print("Error encountered with " + allProbs[g][f][1] + "!")
                        return
                    elif prob <= i*0.1:
                        fracs[g][f][i].append(prob)
                        added = True
                    else:
                        i+=1
            #
            for frac in range(len(fracs[g][f])):
                fracs[g][f][frac] = len(fracs[g][f][frac])
    #
    figs, axs = plt.subplots(len(allProbs), 1, figsize=(10, 8), sharey=False)
    xlabels = [i*0.1 for i in range(11)]
    for c, g in enumerate(graphlist):
        #
        axs[c].grid(zorder=0, which='both')
        #
        maxProbFreq = 0
        for f in range(len(allProbs[c])):
            if maxProbFreq < max(fracs[c][f]):
                maxProbFreq = max(fracs[c][f])
            lstyle = f
            while lstyle > 2:
                lstyle -= 3
            axs[c].plot(xlabels, fracs[c][f], label=allProbs[c][f][1], 
                        linestyle=lines[f], marker='o', markersize=7.5, 
                        alpha=0.7, color=colours[f], linewidth=4, zorder=3)
        #
        axs[c].set_ylim([0, maxProbFreq*1.25])
        axs[c].set_title(probFunc[1] + " Comparisons:")
        axs[c].legend(ncol=2)
        figs.suptitle(g + " " + probFunc[1] + ":", 
                      fontsize=20, fontweight='bold')

#    
fs = [(rootList, [(1/2), (1/3), (1/4)], 
       ["Square rooted", "Cube rooted", "Fourth root"]), 
      (mmNormalize, "Min-Max normalized"), 
      (zNormalize, "Z-score normalized"), 
      (robustNormalize, "Interquartile normalized"), 
      (logList, "Log-scaled")]
cs = ['red', 'orange', 'lawngreen', 'green', 'aqua', 
      'lightskyblue', 'blue', 'magenta', 'mediumpurple', 'darkkhaki']
ls = ['-', (0, (5, 5)), (0, (5, 5)), (0, (5, 5)), '--', 'dotted', (0, (5, 10)), (0, (5, 5)),]

#
#"""
for gs in [graphs]:
    for probFunc in probFuncs:
        compareProbsLine2(gs, probFunc, fs, cs, ls)
#"""
print("")


# In[ ]:


def compareProbsLine3(graphlist, baseFunc, funclist, colours, lines):
    #
    allProbs = []
    for g in graphlist:
        gProbs, ind = [(baseFunc[0](graphlist[g]), "Base Values")], 0
        for func in funclist:
            if len(func) > 2:
                for par in range(len(func[1])):
                    gProbs.append((func[0](gProbs[0][0], func[1][par]), func[2][par]))
                    ind += 1
            else:
                gProbs.append((func[0](gProbs[0][0]), func[1]))
                ind += 1
            #
            if max(gProbs[ind][0]) > 1 or min(gProbs[ind][0]) < 0:
                gProbs[ind] = (mmNormalize(gProbs[ind][0]), 
                               (gProbs[ind][1] + " (adjusted)"))
        allProbs.append(gProbs)
    #
    fracs = [[[[] for _ in range(11)] for _ in range(len(allProbs[g]))]
             for g in range(len(allProbs))]
    for g in range(len(allProbs)):
        for f in range(len(allProbs[g])):
            for prob in allProbs[g][f][0]:
                added, i = False, 0
                while added == False:
                    if prob < 0 or prob > 1:
                        print("Error encountered with " + allProbs[g][f][1] + "!")
                        return
                    elif prob <= i*0.1:
                        fracs[g][f][i].append(prob)
                        added = True
                    else:
                        i+=1
            #
            for frac in range(len(fracs[g][f])):
                fracs[g][f][frac] = len(fracs[g][f][frac])
    #
    figs, axs = plt.subplots(len(allProbs), 1, figsize=(10, 8), sharey=False)
    xlabels = [i*0.1 for i in range(11)]
    for c, g in enumerate(graphlist):
        #
        axs[c].grid(zorder=0, which='both')
        #
        maxProbFreq = 0
        for f in range(len(allProbs[c])):
            if maxProbFreq < max(fracs[c][f]):
                maxProbFreq = max(fracs[c][f])
            lstyle = f
            while lstyle > 2:
                lstyle -= 3
            axs[c].plot(xlabels, fracs[c][f], label=allProbs[c][f][1], 
                        linestyle=lines[f], marker='o', markersize=7.5, 
                        alpha=0.7, color=colours[f], linewidth=4, zorder=3)
        #
        axs[c].set_ylim([0, maxProbFreq*1.25])
        axs[c].set_title(probFunc[1] + " Comparisons:")
        axs[c].legend(ncol=2)
        figs.suptitle(g + " " + probFunc[1] + ":", 
                      fontsize=20, fontweight='bold')

#    
fs = [(rootList, [(1/2), (1/3), (1/4)], 
       ["Square rooted", "Cube rooted", "Fourth root"]), 
      (mmNormalize, "Min-Max normalized"), 
      (zNormalize, "Z-score normalized"), 
      (robustNormalize, "Interquartile normalized"), 
      (logList, "Log-scaled")]
cs = ['red', 'orange', 'lawngreen', 'green', 'aqua', 
      'lightskyblue', 'blue', 'magenta', 'mediumpurple', 'darkkhaki']
ls = ['-', (0, (5, 5)), (0, (5, 5)), (0, (5, 5)), '-', '-', (0, (5, 5)), (0, (5, 5)),]

#
#"""
for gs in [graphs]:
    for probFunc in probFuncs:
        compareProbsLine3(gs, probFunc, fs, cs, ls)
#"""
print("")


# In[ ]:




