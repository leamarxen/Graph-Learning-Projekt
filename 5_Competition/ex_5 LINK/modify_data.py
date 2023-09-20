import torch
import networkx as nx
import random as rd

def concatenation(X, edges, N_out, N_in):
    """
    Make edge embeddings of the different node embeddings and concatenate them

    :X: node embedding computed in node2vec, size: |V|x128
    :edges: edges that have to be embedded
    :N_out: node embedding based on outgoing edges (and their labels) of each node, size: |V|=30
    :N_in: node embedding based on incoming edges (and their labels) of each node, size: |V|=30

    :return:
        edge_embedding: edge embedding for the edges from the input, size: |E|x316 (316=128+30+138+30)
    """
    #convert edge information from list((e_out, e_in)) to list(e_out), list(e_in)
    out_edge, in_edge = list(zip(*edges))
    out_edge, in_edge = list(out_edge), list(in_edge)

    #derive edge embedding for each matrix and concatenate (if edge=(e1,e2), first for e1, then e2)
    edge_embedding = torch.cat((X[out_edge], N_out[out_edge], X[in_edge], N_in[in_edge]), dim=1)
    return edge_embedding

def edge_split(graph, input_percentage, validation_percentage):
    """
    Split edges of a graph in input edges (input for classification), training and validation edges
    and unlabeled edges.

    :graph: graph for which edges should be split
    :input_percentage: percentage of input edges (i.e. how many edges should be used for input)
    :validation_percentage: percentage of validation edges (i.e. how many edges should be used for 
                                    validation during the classification) -> rest of edges used for training

    :return:
        :input_edges: edges to be used as input data, i.e. for edge embedding during classification; 
                            form: list((e1,e2), label)
        :training_edges: edges to be used as training data during classification; form: list((e1,e2))
        :train_edge_label: edge labels for training_edges
        :validation_edges: edges to be used as validation data during classification; form: list((e1,e2))
        :val_edge_label: edge labels for validation_edges
        :unlabeled: test edges for classification; form: list((e1,e2))
    """
    #get edge information
    edge_attrs = nx.get_edge_attributes(graph, "edge_label")
    #sort edges through their id to ensure that test data will be classified in the right order
    sorted_edges = sorted(list(graph.edges(data=True)), key= lambda edge: edge[2]["id"])
    tuples = [((i,j), attrs["edge_label"]) for i,j,attrs in sorted_edges]

    #split in labeled and unlabeled edges
    unlabeled = tuples[:1000]
    labeled = tuples[1000:]

    #first split all labeled edges in two groups: 
    # those which provide information as input and those which are used for classification
    
    #first get multiedges in order to ensure that at most one edge is used for classification 
    #double_edges_1: edge two of a double edge -> will be used as input_edges
    double_edges_1 = [((t1,t2),value) for (t1,t2,t3), value in nx.get_edge_attributes(graph, "edge_label").items() if t3!=0]
    #double_edges 0: edge one of a double edge -> either used for classification or input
    double_edges_0 = [((t1,t2),edge_attrs[(t1,t2,0)]) for (t1,t2),_ in double_edges_1 if edge_attrs[(t1,t2,0)] is not None]
    for entry in double_edges_0+double_edges_1:
        labeled.remove(entry)

    #sample input, training and validation edges from remaining data
    labeled_size = len(labeled)
    #get number of edges to sample for input and validation data
    num_input = int(labeled_size*input_percentage)
    num_validation = int(labeled_size*validation_percentage)
    
    #sample input edges
    input_edges = rd.sample(labeled, num_input)
    #remaining edges: for classification
    classification_edges = list(set(labeled).difference(set(input_edges)))
    #add double edges to input and labeled data respectively
    input_edges += double_edges_1
    classification_edges += double_edges_0

    #contiue with validation and training data sampling
    validation_edges = rd.sample(classification_edges, num_validation)
    #training edges: remaining edges
    training_edges = list(set(classification_edges).difference(set(validation_edges)))
    
    #split in edges and labels (not for input_edges, as this is easier for handling later on)
    training_edges, train_edge_label = list(zip(*training_edges))
    validation_edges, val_edge_label = list(zip(*validation_edges))
    unlabeled, _ = list(zip(*unlabeled))

    return input_edges, training_edges, train_edge_label, validation_edges, val_edge_label, unlabeled


