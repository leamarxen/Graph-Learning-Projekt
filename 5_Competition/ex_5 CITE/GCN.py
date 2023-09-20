import torch
from GCN_Layer import GCN_Layer


class GCN_node(torch.nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        """
        Initializes the node level GCN with several GCN layers.

        :input_dim: input dimension (len of node_arrtibutes)
        :output_dim: output dimension (number of labels)
        :hidden_dim: size of the hidden layers
        :num_layers: number of GCN-layers
        """
        super(GCN_node, self).__init__()
        self.num_layers = num_layers

        # add sub-modules as attribute
        self.input_layer = GCN_Layer(input_dim, hidden_dim,is_linear=True)

        # add linear output
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)

        # store multiple submodules in "ModuleList"
        self.hidden_layers = torch.nn.ModuleList(
            [GCN_Layer(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )

        # add dropout against overfit
        self.dropout1 = torch.nn.Dropout(0.2)

    def forward(self, A, H):
        """
        Forward pass for the Node Level GCN.

        :A: Adjecency matrix
        :H: node embeddings of last layer
        :return: torch tensor of size (number of nodes x output_dim)
        """

        # apply GCN-layers
        y = self.input_layer(A, H)
        for i in range(self.num_layers - 1):
            y = self.hidden_layers[i](A, y)

        # linear output layer, with dropout
        y = self.dropout1(y)
        y = self.output_layer(y)
        return y