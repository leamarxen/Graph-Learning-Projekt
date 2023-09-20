import pickle

import numpy as np
import torch
from sklearn.preprocessing import normalize

from normalized_adj import get_padded_normalized_adjacency
from data_utils import get_padded_node_labels, get_padded_node_attributes

def load_data(path):
    """
    Loads the data from a path and extracts and pads A, H and y_one_hot (see below).

    :path: path where data can be found
    :return: 
        :A: padded and stacked adjacency matrices of all graphs in data 
                (torch tensor of size #graphs x #vertices x #vertices) 
        :H: padded and stacked first vertex embedding, consisting of node labels (and attributes)
                (torch tensor of size #graphs x #vertices x dim_node_labels+attributes)
        :y_one_hot: class labels one-hot-encoded
                (torch tensor of size #graphs x #labels)
    """
    # load the given data and cast to torch tensors
    data = pickle.load(open(path, "rb"))
    
    # get list of padded & normalized adjecency matrices and cast to torch tensors
    A = get_padded_normalized_adjacency(data)
    A = torch.tensor(A, dtype=torch.float32)
    
    # get classification labels, one-hot-encode and cast to torch
    y_label = np.array([g.graph["label"] for g in data])
    # labels should start with 0
    if not 0 in y_label:
        y_label = y_label -1
    y_one_hot = np.zeros((y_label.size, y_label.max()+1))
    y_one_hot[np.arange(y_label.size),y_label] = 1
    y_one_hot = torch.tensor(np.array(y_one_hot))

    # get padded node labels gets one-hot-encoded node labels, then cast to torch
    H = get_padded_node_labels(data)
    H = torch.tensor(H)

    # if there are node attributes, concatenate them to the node labels
    if "node_attributes" in data[0].nodes(data=True)[1].keys():
        node_as = get_padded_node_attributes(data)
        node_as = [normalize(x,norm='l2') for x in node_as]
        node_as = torch.tensor(node_as)
        H = torch.cat((H, node_as), dim=2)

    return A, H, y_one_hot



# test output of load_data
#A, H, y = load_data("datasets/ENZYMES/data.pkl", node_attrs=True)
#print(A.size(), H.size(), y.size())