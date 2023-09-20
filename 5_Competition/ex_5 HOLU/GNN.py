import torch
import torch.nn as nn


from GNN_layer import GNN_Layer
from Virtual_Node import Virtual_Node
from Sparse_Sum_Pooling import Sparse_Sum_Pooling

class GNN(torch.nn.Module):

    def __init__(self, hidden_dim, aggr_type, num_layers, node_attrs, edge_attrs,drop_out, virtual_node = False):
        """
        Initializes the graph level GNN with several GNN layers, Virtual Node if requested, Sparse Sum Pooling and
        an MLP with one hidden layer and Relu activation.

        :hidden_dim: size of hidden layers
        :aggr_type: type of aggregation 
        :num_layers: number of GNN-layers
        :node_attrs: number of node attributes 
        :edge_attrs: number of edge attributes (here: 5)
        :drop_out: the dropout rate
        :virtual_node: if True, insert virtual node layers after each GNN layer but the last one
        """
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.virtual_node = virtual_node
        
        #add first GNN Layer without residual connection and with different dimension than the following layers, e.g. 
        #node_attrs+edge_attrs, but from second layer on node_attrs changes to hidden_dim (-> column dimension of H)
        self.input_layer = GNN_Layer(node_attrs+edge_attrs, hidden_dim, node_attrs+hidden_dim, aggr_type,res_conn=False)
        # Store multiple submodules in 'ModuleList', with residual connection
        self.hidden_layers = torch.nn.ModuleList(
            [GNN_Layer(hidden_dim+edge_attrs, hidden_dim, hidden_dim+hidden_dim, aggr_type,res_conn=True) for _ in range(num_layers-1)]
        )

        #virtual node after each but the last layer if virtual_node=True
        if virtual_node:
            self.virtual_nodes = torch.nn.ModuleList(
                [Virtual_Node(hidden_dim) for _ in range(num_layers-1)]
            )

        self.sum_pooling = Sparse_Sum_Pooling()

        # add linear modules for subsequent classification
        self.MLP_layer = nn.Linear( hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1) #one class at the end for regression output
        
        # add dropout against overfit
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        

        

    def forward(self, H, idx, X_e, batch_idx):
        """
        Forward pass for the GNN.

        :H: vertex embedding of last layer
        :idx: Index for edge nodes (2 x 2|E|)
        :X_e: Edge attributes (2|E| x d')
        :batch_idx: Index noting which graph a node belongs to 

        :return: torch.float giving regression estimation
        """
        # apply GNN-layers
        y = self.input_layer(H, idx, X_e) 

        for i in range(self.num_layers-1):
            # apply virtual node if requested
            if self.virtual_node:
                y = self.virtual_nodes[i](y, batch_idx)

            y = self.hidden_layers[i](y, idx, X_e)

        # Sum Pooling to get one row for each graph
        y = self.sum_pooling(y, batch_idx)
    
        
        # MLP with one hidden layer + relu, then a linear output layer, with dropout (dropout rate specified by user)
        y = self.dropout1(y)
        y = self.MLP_layer(y)
        y = self.dropout2(y)
        y = torch.relu(y)

        y = self.output_layer(y)

        return y

#with open("../datasets/HOLU/data.pkl", "rb") as f:
#    HOLU = pickle.load(f)
#HOLU_labeled = HOLU[1000:]
#dataset = CustomDataset(HOLU_labeled)

