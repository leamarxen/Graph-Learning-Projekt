import networkx as nx
import numpy as np

# Exercise 1
def closed_walk_kernel(graph_list):
    """Implemention of the Closed Walk Kernel.
    
    Keyword argument: 
    graph_list -- Dataset representing list of graphs
    
    Key idea:
    Computation through eigenvalues with help of the eigenvalue decomposition.
    
    Returns: List of Histograms (one histogram for every graph of the dataset) """

    # Compute the mean number of nodes over all graphs
    l = int(np.mean([len(g.nodes) for g in graph_list])) 
    print("mean of number of nodes:", l)

    
    # Compute the histogram of closed walks of different length up to the mean number of nodes
    feature_vectors = []
    for graph in graph_list:
        number = []
        A = nx.adjacency_matrix(graph) 
        A =A.todense() 
        lambdas = np.linalg.eigvalsh(A)
        for j in range(1, l+1):
            power_lambdas= [x**(j) for x in lambdas ]
            sum_lambdas=int(np.round(sum(power_lambdas)))
            number.append(sum_lambdas) 
        feature_vectors.append(number)
        
    return feature_vectors
