import torch
from torch_scatter import scatter_sum


class Sparse_Sum_Pooling(torch.nn.Module):

    def __init__(self):
        """
        Initializes a Sparse Sum Pooling Layer.
        """
        super(Sparse_Sum_Pooling, self).__init__()


    def forward(self, H, batch_idx):
        """
        Forward pass for Sum Pooling.

        :H: vertex embedding of last layer
        :return: graph embeddings
        """
        y = scatter_sum(H, batch_idx, dim=0)
        
        return y


