import math
import numpy as np

from data_utils import get_padded_node_labels, get_padded_node_attributes, get_padded_adjacency

def norm_adj_matrix(graph_list):

    """
    Compute the normalized adjacency matrix of every graph in the list.
    """
    
    # get padded adj
    pad_adj_matrix=get_padded_adjacency(graph_list)
    
    print(pad_adj_matrix[0][0][1])

    # get the node degrees of each graph.  
    node_deg=[]
    for i in range(len(pad_adj_matrix)):
    
        a= pad_adj_matrix[i].sum(axis=1)+1
        node_deg.append(a)

    # Get the normalized adjacency matrices and add them in a list
    list=[]
    for k in range(len(pad_adj_matrix)):
        norm_matrix = np.zeros((len(pad_adj_matrix[k]),len(pad_adj_matrix[k])))
        for i in range(len(pad_adj_matrix[k])):
            for j in range(len(pad_adj_matrix[k])):
                if pad_adj_matrix[k][i][j] !=0:
                    norm_matrix[i][j] = 1/math.sqrt(node_deg[k][i]*node_deg[k][j])
                elif i == j:
                    norm_matrix[i][j] = 1/math.sqrt(node_deg[k][i]*node_deg[k][i])
               # else:
               #     norm_matrix[i][j] = 0
        list.append(norm_matrix)
    return list