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

    def __init__(self, data, test_data= False):
        self.graphs = list()

        for graph in data:
            edge_begin = [i for i,j in graph.edges()]
            edge_end = [j for i,j in graph.edges()]

            #use start nodes and end nodes to present all edges and do it reverse
            edge_list = torch.tensor([edge_begin+edge_end, edge_end+edge_begin])

            #get node_features and edge_features 
            node_features = list(nx.get_node_attributes(graph, "node_label").values())
            edge_attrs = list(nx.get_edge_attributes(graph, "edge_label").values())
            node_attrs = list(nx.get_node_attributes(graph, "node_attributes").values())
            node_attrs = np.vstack(node_attrs)
            
            #convert features from number to one_hot
            node_features = np.array(node_features)
            node_one_hot = np.zeros((node_features.size, 36))
            node_one_hot[np.arange(node_features.size),node_features] = 1
            node_features = torch.tensor(np.array(node_one_hot), dtype=torch.float32)

            # compute edge attributes
            edge_attrs = np.array(edge_attrs) 
            edge_one_hot = np.zeros((edge_attrs.size, 4))
            edge_one_hot[np.arange(edge_attrs.size),edge_attrs] = 1
            edge_features = torch.tensor(np.array(edge_one_hot))#, dtype=torch.float32)
            # add distance of edge nodes to the edge vector
            dist = np.linalg.norm(node_attrs[edge_begin]-node_attrs[edge_end], axis=1)
            dist = torch.unsqueeze(torch.tensor(dist), dim=1)
            edge_features = torch.hstack((edge_features,dist))            
            #double the size
            edge_features = edge_features.repeat(2,1)

            if test_data:
                graph_label = torch.zeros(1)
            
            else:
                graph_label = torch.tensor([(graph.graph["label"])]).to(dtype=torch.float32)
            #graph_label = torch.tensor(graph.graph["label"])
            
            self.graphs.append((edge_list, node_features, edge_features, graph_label))

            

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self,i):
        return self.graphs[i]

#with open("../datasets/HOLU/data.pkl", "rb") as f:
#    HOLU = pickle.load(f)

#HOLU_labeled = HOLU[1000:]
#dataset = CustomDataset(HOLU_labeled)
#print(dataset[0][3])
