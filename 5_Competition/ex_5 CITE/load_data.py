import pickle
import torch
from torch.nn.functional import one_hot
import numpy as np
from adj_matrix import get_adj_matrix,get_node_attributes
import numpy as np



def load_data_node(train_path):
    """
    Loads the training and validation data from given graph

    :path: path where data can be found
    :return:
        :A: adjacency matrix of the graph
        :H: node embeddings
        :y_label: class labels one-hot-encoded except None-class (first 1000 nodes)

    """

    # load the given data and cast to torch tensors
    data_train = pickle.load(open(train_path, "rb"))

    # get list of padded & normalized adjecency matrices and cast to torch tensors
    try:
        with open('adj_matrix.pickle','rb') as f:
            A = pickle.load(f)
    except:
        A = get_adj_matrix(data_train)
        A = torch.tensor(A, dtype=torch.float32)
        with open('adj_matrix.pickle','wb') as f:
            pickle.dump(A,f)


    # get not-None labels
    y_label = np.float32([node[1]["node_label"] for node in data_train.nodes(data=True)][1000:])
    y_label = torch.nn.functional.one_hot(torch.tensor(y_label).long()).float()

    # get node attributes
    H = get_node_attributes(data_train)
    H = torch.tensor(H)

    return A, H, y_label


