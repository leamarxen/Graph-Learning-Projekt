import networkx as nx
from collections import Counter
import scipy.sparse as sp
import numpy as np
from multiset import FrozenMultiset


# Exercise 3: Weisfeiler-Leman-Kernel

def wl_kernel(orig_graphs, labelname="node_label", rounds=4):
    '''
    Implementation of the Weisfeiler-Leman-Kernel

    Keyword Arguments
    orig_graphs: original list of graphs
    label_name: initial node labels/colors (can be None, default value: "node_label")
    rounds: number of rounds of color refinement

    return: f_vecs -> list of histograms, one for each graph 
        (each histogram: sparse coo-matrix of shape (1, total_number_of_colors))

    Key ideas:
        - store the colors as node attributes of the respective graphs
        - use a hash function to compute new colors, but assign each new hashcolor to an integer 
            color (starting from 0) and store the pairs in a dictionary (keys: hashcolors, values:
            respective integer colors) 
        - use integer colors as indices in the final histograms (e.g. the number of occurences of 
            color 4 is stored at fvecs[4])
    '''

    #copy graphs because they are modified later
    graphs = [graph.copy() for graph in orig_graphs]
    
    ##### COLOR REFINEMENT ############
    idx_counter = 0
    coldict = dict() #save all colors in a dictionary (keys: hash values, values: index in the final histograms)
    
    #initial colors: if there is a initial color scheme, use it in round 0
    if labelname:
        for graph in graphs:
            init_labels = nx.get_node_attributes(graph, labelname) #dict {node: label}
            hash_labels = {key: hash(value) for key,value in init_labels.items()} #hash label values (-hashcolors) so that they are the same for all coming graphs and rounds
            colors = list(set(hash_labels.values())) #list of the different colors in this graph
            for hashcol in colors:
                #check if colors already have been saved in coldict and save them if not
                if hashcol not in coldict.keys():
                    coldict[hashcol] = idx_counter
                    idx_counter += 1 #counts total number of colors
            #change from hashed colors to final integer colors which will be used afterwards
            new_labels = {key: coldict[hashvalue] for key,hashvalue in hash_labels.items()}
            nx.set_node_attributes(graph, new_labels, str(0))
    # no initial color scheme -> every node gets same color
    else:
        for graph in graphs:
            nx.set_node_attributes(graph, 0, str(0))
        #save color in coldict and increment idx_counter (which counts total number of colors)
        coldict[0] = idx_counter #here: 0
        idx_counter += 1

    #next rounds of color refinement
    for k in range(1, rounds+1):
        for graph in graphs:
            #attribute dictionaries
            attrs_last_round = nx.get_node_attributes(graph, str(k-1)) #dictionary with nodes as keys and corresponding attributes of last round as values
            attrs_this_round = dict() #where you save attributes of this round
            
            #compute current color of each node
            for node in graph.nodes():
                #get colors of neighbors and hash them together with the node's color
                colset = FrozenMultiset(attrs_last_round.get(neighbor) for neighbor in list(graph[node]))
                hashcol = hash((attrs_last_round.get(node), colset))
                #if hash produces a new color:
                if hashcol not in coldict.keys():
                    coldict[hashcol] = idx_counter
                    idx_counter += 1
                attrs_this_round[node] = coldict[hashcol]
            #save current colors of the graph as node attributes
            nx.set_node_attributes(graph, attrs_this_round, name=str(k))

            
    ####### CONSTRUCT FEATURE VECTORS ###############
    f_vecs = list() #where feature vectors (histograms) will be stored
    for graph in graphs:
        c = Counter()
        for k in range(rounds):
            #count number of colors that appeared in each round, 
            #e.g. c = {0:302, 1:4} if color 0 appeared 302 times and color 1 appeared 4 times
            c.update(nx.get_node_attributes(graph, str(k)).values()) 
        #create feature vector as sparse matrix in format 1 x idx_counter
        data = np.array(list(c.values()))
        col = np.array(list(c.keys()))
        row = np.zeros(len(col)) #only one row for each histogram
        f_vec = sp.coo_matrix((data, (row,col)), shape=(1, idx_counter)) #feature vector with histogram entries 
        f_vecs.append(f_vec)

    return f_vecs
