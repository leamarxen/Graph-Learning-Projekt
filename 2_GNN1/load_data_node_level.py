import pickle
import torch
from torch.nn.functional import one_hot
import numpy as np
from adj_matrix import norm_adj_matrix
import numpy as np

from data_utils import get_padded_node_attributes

def load_data_node(train_path, test_path, node_attrs=False):

    """
    Loads the training and test data from paths and extracts and pads: 
        - A, H and y_label (from the training dataset)
        - A_test, H_test and y_label_test (from the test dataset)

    :path: path where data can be found
    :return: 
        :A: padded and stacked adjacency matrices of all graphs in data 
                (torch tensor of size #graphs x #vertices x #vertices) 
        :H: padded and stacked first vertex embedding, consisting of node labels (and attributes)
                (torch tensor of size #graphs x #vertices x dim_node_labels+attributes)
        :y_label: class labels one-hot-encoded
                (torch tensor of size #graphs x #labels)
        :A_test 
        :H_test
        :y_label
        
    """


    # load the given data and cast to torch tensors
    data_train = pickle.load(open(train_path, "rb"))   
    data_test = pickle.load(open(test_path, "rb"))
    
    # get list of padded & normalized adjecency matrices and cast to torch tensors
    A = norm_adj_matrix(data_train)
    A = torch.tensor(A, dtype=torch.float32)
    
    A_test = norm_adj_matrix(data_test)
    A_test = torch.tensor(A_test, dtype=torch.float32) 
    
    # get classification labels
    y_label = [np.int32([node[1]["node_label"] for node in data_train[0].nodes(data=True)])]
    y_label = torch.nn.functional.one_hot(torch.tensor(y_label).long())
    y_label= torch.tensor(y_label).float()
    
    y_label_test = [np.int32([node[1]["node_label"] for node in data_test[0].nodes(data=True)])]
    y_label_test = torch.nn.functional.one_hot(torch.tensor(y_label_test).long())
    y_label_test= torch.tensor(y_label_test).float()

    # get padded node attributes
    H = get_padded_node_attributes(data_train)
    H = torch.tensor(H)

    H_test = get_padded_node_attributes(data_test)
    H_test = torch.tensor(H_test)
    return A, A_test, H, H_test, y_label, y_label_test