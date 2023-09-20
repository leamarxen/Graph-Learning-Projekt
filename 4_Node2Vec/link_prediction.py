from sklearn.linear_model import LogisticRegression
import pickle
import torch 
import networkx as nx
import random
import numpy as np

from train_node2vec import train_node2vec



def link_prediction(network_path):
    """
    Samples evaluation edges from a graph such that the connected components of the graph
    stay the same without the sampled evaluation edges. The remaining edges are returned
    as training edges.

    :param network_path:graph
    :return:list of train_edges,list of eval_edges
    """

    with open(network_path, 'rb') as f:
        Graph = pickle.load(f)
    Graph = Graph[0]

    #make sure the operation on the copied graph does not change the original graph
    Graph_copy = Graph.copy()
    num_edges = Graph.number_of_edges()

    #number of 20% of all the edges
    perc20_edges = int(num_edges*0.2)

    #number of connected components
    num_con_components = nx.number_connected_components(Graph)

    E_eval = []
    i=0

    #Make sure that the original connected components are still connected after removing the eval_edges.
    while i < perc20_edges:
        # to make it more effective,remove 100 edges everytime and check the number of connected components.
        if (perc20_edges-i) >= 100:
            sampled_edges = random.sample(Graph_copy.edges,100)
        else:
            sampled_edges = random.sample(Graph_copy.edges, perc20_edges-i)

        Graph_copy.remove_edges_from(sampled_edges)

        #if the number of connected components changed,then do not remove those edges.
        if nx.number_connected_components(Graph_copy) != num_con_components:
            # print("Bridge edge was used. Number connected components changed.")
            Graph_copy.add_edges_from(sampled_edges)

        else:
            E_eval += sampled_edges
            i += 100

    #all the not selected edges are train_edges
    E_train = list(set(Graph.edges).difference(set(E_eval)))
    return E_train, E_eval

def sample_non_edges(network_path, len_train, len_eval):
    """
    Samples non-edges, i.e. from the set of possible, non-existing edges in the graph, such that
    the sizes of the samples correspond to those of the training and evaluation edges

    :param network_path: graph
    :param len_train: size of train_edges
    :param len_eval: size of eval_edges
    :return:list of train_neg_samples and eval_train_samples
    """
    with open(network_path, 'rb') as f:
        Graph = pickle.load(f)
    Graph = Graph[0]

    #number of edges and negative samples should be same
    number_to_sample = len_train + len_eval

    non_edges = list(nx.non_edges(Graph))

    sampled_non_edges = random.sample(non_edges, number_to_sample)

    #seperate nagative samples into train and eval
    return sampled_non_edges[:len_train], sampled_non_edges[len_train:]

def element_wise_product(X, edges):
    """
    Computes the Hadamard product as an edge embedding 

    :param X: node embeddings
    :param edges: edges that need to be computed edge embeddings
    :return:edge embeddings
    """
    
    #merge all out_nodes into one array,and all in_nodes into one array
    node_1, node_2 =  list(zip(*edges))
    node_1 = np.array(node_1)
    node_2 = np.array(node_2)
    hadamard_prod = np.multiply(X[node_1-1], X[node_2-1])

    return hadamard_prod


