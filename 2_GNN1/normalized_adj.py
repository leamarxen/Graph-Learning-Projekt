import networkx as nx
import scipy.sparse as sp
import numpy as np


def normalized_adj(graph):
    """
    Normalizes the adjecency of a given graph as was required in exercise 1.
    :param graph: one networkx graph
    :return: Numpy array result
    """
    ADJ = nx.adjacency_matrix(graph).todense()
    result = np.zeros(ADJ.shape)
    degrees = ADJ.sum(axis=1)+1
    for x in range(ADJ.shape[0]):
        for y in range(ADJ.shape[1]):
            if (x==y) or (ADJ[x,y] != 0):
                result[x,y] = 1/np.sqrt(degrees[x,0]*degrees[y,0])
    return result

def get_padded_normalized_adjacency(graphs):
    """
    Changed version of get_padded_adjacency from data_utils.py
    Computes a 3D Tensor A of shape (k,n,n) that stacks all normalized adjacency matrices.
    Here, k = |graphs|, n = max(|V|) and A[i,:,:] is the padded normalized adjacency matrix of the i-th graph.
    :param graphs: A list of networkx graphs
    :return: Numpy array A
    """
    max_size = np.max([g.order() for g in graphs])
    A_list = [normalized_adj(g) for g in graphs]
    A_padded = [np.pad(A, [0, max_size-A.shape[0]]) for A in A_list]
    return np.float32(A_padded)


# test functionality of normalized_adj
#import pickle
#ENZ = pickle.load(open("datasets/ENZYMES/data.pkl", "rb"))
#print(normalized_adj(ENZ[0]))
#print(nx.path_graph(4))
#print(normalized_adj(nx.path_graph(4)))
