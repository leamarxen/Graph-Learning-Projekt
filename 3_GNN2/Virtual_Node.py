import torch
from torch_scatter import scatter_sum
import pickle

from customDataset import CustomDataset
from collate_graphs import collate_graphs


class Virtual_Node(torch.nn.Module):

    def __init__(self, hidden_dim):
        """
        Initializes a Virtual Node.

        :hidden_dim: hidden dimension of previous layer
        """
        super(Virtual_Node, self).__init__()

        #we dont use a total matrix here,because we need trainable matrix for each graph.
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)


        

    def forward(self, H, batch_idx):
        """
        Forward pass for adding the global information of each graph to each node.

        :H: vertex embedding of last layer
        :return: vertex embedding after adding the global information
        """
        #sum all the node attributes that belong to the same graph
        y = scatter_sum(H, batch_idx, dim=0) #|G| x hidden dim
        
        #'hash' the summed node attributes to present the global graph attribute
        y =  self.linear(y)
        
        # apply activation
        y = torch.relu(y)

        # add information to each row of H (different information for each graph)
        y = H + y[batch_idx]
        
        return y



