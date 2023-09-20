from torch.utils.data import Dataset
import networkx as nx
import pickle
import numpy as np
import torch


class CustomDataset(Dataset):
    """
    present each graph in the dataset by using sparse representation

    Dataset:list of graphs

    self.graphs:list of sparse presentation of each graph
    """

    def __init__(self, data):
        self.graphs = list()

        for graph in data:
            edge_begin = [i for i,j in graph.edges()]
            edge_end = [j for i,j in graph.edges()]

            #use start nodes and end nodes to present all edges and do it reverse
            edge_list = torch.tensor([edge_begin+edge_end, edge_end+edge_begin])

            #get node_features and edge_features 
            node_features = list(nx.get_node_attributes(graph, "node_label").values())
            edge_attrs = list(nx.get_edge_attributes(graph, "edge_label").values())
            
            #convert features from number to one_hot
            node_features = np.array(node_features)
            node_one_hot = np.zeros((node_features.size, 21))
            node_one_hot[np.arange(node_features.size),node_features] = 1
            node_features = torch.tensor(np.array(node_one_hot), dtype=torch.float32)

            # Because minimal edge attr =1, but we want 0 for one_hot
            edge_attrs = np.array(edge_attrs)-1 
            #checked that there are 3 different edge attributes
            edge_one_hot = np.zeros((edge_attrs.size, 3))
            edge_one_hot[np.arange(edge_attrs.size),edge_attrs] = 1
            edge_features = torch.tensor(np.array(edge_one_hot), dtype=torch.float32)
            #double the size
            edge_features = edge_features.repeat(2,1)

            graph_label = torch.tensor(graph.graph["label"])
            self.graphs.append((edge_list, node_features, edge_features, graph_label))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self,i):
        return self.graphs[i]


    
