import torch
import torch.nn as nn

from GCN_modul import GCN_Layer

class graph_level_GCN(torch.nn.Module):

    def __init__(self, input_dim, output_dim, num_vertices, hidden_dim, num_layers):
        """
        Initializes the graph level GCN with several GCN layers.

        :input_dim: input dimension (3rd dimension of batched H0)
        :output_dim: output dimension (number of classification classes)
        :num_vertices: number of vertices of the graphs (2nd dimension of batched adjecency matrix)
        :hidden_dim: size of the hidden layers
        :num_layers: number of GCN-layers
        """
        super(graph_level_GCN, self).__init__()
        self.num_layers = num_layers

        #add sub-modules as attribute
        self.input_layer = GCN_Layer(input_dim, hidden_dim, num_vertices)
        # Store multiple submodules in 'ModuleList'
        self.hidden_layers = torch.nn.ModuleList(
            [GCN_Layer(hidden_dim, hidden_dim, num_vertices) for _ in range(num_layers-1)]
        )

        # add linear modules for subsequent classification
        self.MLP_layer = nn.Linear(num_vertices, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        # add dropout against overfit
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        

        

    def forward(self, A, H):
        """
        Forward pass for the Graph Level GCN.

        :A: Adjecency matrix
        :H: vertex embedding of last layer
        :return: torch vector of size batch_size x output_dim, with one-hot-encoded output
        """
        # apply GCN-layers
        y = self.input_layer(A, H)
        for i in range(self.num_layers-1):
            y = self.hidden_layers[i](A, y)

        #sum over all rows respectively, resulting dimension: batchx|V|x1
        y = torch.sum(y, dim=2)
        
        # MLP with one hidden layer + relu, then a linear output layer, with dropout
        y = self.dropout1(y)
        y = self.MLP_layer(y)
        y = self.dropout2(y)
        y = torch.relu(y)
        y = self.output_layer(y)

        return y
