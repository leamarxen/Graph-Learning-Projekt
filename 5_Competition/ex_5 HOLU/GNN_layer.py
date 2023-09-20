from re import X
import torch
from torch_scatter import scatter_max, scatter_sum, scatter_mean
from custom_dataset import CustomDataset
import pickle

class GNN_Layer(torch.nn.Module):

    def __init__(self, dim_in, hidden_dim, hidden_dim2, aggr_type, res_conn):
        """
        Initializes a GNN Layer.

        :dim_in: sum of d_h (column dimension of H_l-1, from 2nd layer on equals hidden_dim) 
                            + d' (column dim of X_e (edge_attr))
        :hidden_dim: can be chosen arbitrarily
        :hidden_dim2: sum of d_h (column dim of H_l-1 (2nd layer: hidden_dim)) + hidden_dim
        :aggr_type: type of scatter operation: choose between max, sum and mean
        """
        super(GNN_Layer, self).__init__()
        self.aggr_type = aggr_type
        self.res_conn = res_conn

        #use Kaiming Init when using ReLU
        #dim_in: d_h (dim of node_attr) + d' (dim of edge_attr), hidden_dim: can be chosen arbitrarily
        self.W1 = torch.nn.Parameter(torch.zeros(dim_in, hidden_dim)).double() 
        #hidden_dim2: d_h (dim of H_l-1) + hidden_dim (dim of Z_l)      
        self.W2 = torch.nn.Parameter(torch.zeros(hidden_dim2, hidden_dim)).double() 
        torch.nn.init.kaiming_normal_(self.W1)
        torch.nn.init.kaiming_normal_(self.W2)

        #layer which performs normalization
        self.norm1 = torch.nn.LayerNorm(hidden_dim, dtype=float)
        self.norm2 = torch.nn.LayerNorm(hidden_dim, dtype=float)


    def forward(self, H, idx, X_e):
        """
        Forward pass for a GCN Layer.

        :H: vertex embedding of last layer
        :idx: (directed) edge list
        :X_e: Edge feature matrix
        :return: vertex embedding of this layer
        """
        # concatenate input
        x = torch.cat((H[idx[0]], X_e), dim=1)
        y = torch.matmul(x, self.W1)
        #apply batch normalization
        y = self.norm1(y)
        # apply activation
        y = torch.relu(y)
       
        if self.aggr_type == "max":
            y = scatter_max(y,idx[1],dim=0)[0]
        elif self.aggr_type == "sum":
            y = scatter_sum(y,idx[1],dim=0)
        elif self.aggr_type == "mean":
            y = scatter_mean(y,idx[1],dim=0)
        else:
            raise Exception("Scatter operation not supported. Choose between max, sum and mean.")
   
        y = torch.cat((H,y), dim=1)         
        y = torch.matmul(y, self.W2)
        #apply batch normalization
        y = self.norm2(y)
        # apply activation
        y = torch.relu(y)

        # apply residual connection if res_conn = True
        if self.res_conn:
            y = y + H

        return y.float()

#test functionality
#with open("../datasets/HOLU/data.pkl", "rb") as f:
#    HOLU = pickle.load(f)
#HOLU_labeled = HOLU[1000:]
#dataset = CustomDataset(HOLU_labeled)

#H = dataset[0][1]
#idx = dataset[0][0]
#X_e = dataset[0][2]
#print("idx",idx[0].size())
#print("X_E", X_e.size(1))
#print("H", H.size(1))
#print("H idx", H[idx[0]].size())
#print(X_e.size())
#HX = torch.cat((H[idx[0]],X_e), dim=1)
#print("HX", HX.size())
#x = torch.cat((H[idx[0]], X_e), dim=1)
#print(idx.size(), H[idx[0]].size(), H.size())
#y = scatter_max(HX,idx[1],dim=0)[0]
#print("y",y.size())

#gn = GNN_Layer(H.size(1)+ X_e.size(1), 32,H.size(1)+32, "sum" , res_conn =False)
#print(gn(H, idx, X_e))
