import torch
import pickle
from custom_dataset import CustomDataset

def collate_graphs(data):
    """
    Collation function for the customDataset class, to use as argument with collate_fn in the torch DataLoader.

    :data: several graphs in a list, in the format from __getitem__ of the customDataset, i.e. list of tuples
        where each tuple has following entries: (edge_list, node_features, edge_features, graph_label)
    :return: one tuple which concatenated the info of all graphs, of the form
         (edge_list, node_features, edge_features, graph_label, batch_idx)
         batch idx: torch array which contains information which row of the node features 
            belongs to which original graph
    """
    #initialization of the tensors which should be returned in the end
    edge_list = torch.tensor([]).long()
    node_features = torch.tensor([])
    edge_features = torch.tensor([])
    graph_label = torch.tensor([])
    batch_idx = torch.tensor([], dtype=torch.int64)
    #number of total nodes seen so far
    n_nodes = 0 

    #go through each graph in data and concatenate the graph info to the final tensors
    for i, graph in enumerate(data):
        i_edge_list, i_node_features, i_edge_features, i_graph_label = graph
        
        #add the number of nodes which have been seen in the for loop before to the edge list 
        # and concatenate the result to the final tensor
        edge_list = torch.cat((edge_list, torch.add(i_edge_list, n_nodes)), dim=1)
        #add node features, edge features and the graph label
        node_features = torch.cat((node_features, i_node_features), dim=0)
        edge_features = torch.cat((edge_features, i_edge_features), dim=0)
        graph_label = torch.cat((graph_label, i_graph_label))
        #update the total number of nodes seen so far
        n_nodes += i_node_features.size(0)
        #add i (enumeration of the graph) |i_node_features| times to the batch_idx, 
        # once for each node in the current graph
        batch_idx = torch.cat((batch_idx, torch.tensor([i]*i_node_features.size(0), dtype=torch.int64)))
             
    return (edge_list, node_features, edge_features, graph_label, batch_idx)

#test functionality
#with open("../datasets/HOLU/data.pkl", "rb") as f:
#    HOLU = pickle.load(f)

#HOLU_labeled = HOLU[1000:]
#dataset = CustomDataset(HOLU_labeled)
#print(dataset[0][3])
#print(collate_graphs(dataset[:4]))