import torch

#basic torch module

class GCN_Layer(torch.nn.Module):

    def __init__(self, dim_in, dim_out, is_linear=False):
        """
        Initializes a GCN Layer.

        :dim_in: input dimension
        :dim_out: output dimension
        :is_linear: GCN Layer applies Relu if is_linear=False
        """
        super(GCN_Layer, self).__init__()
        self.is_linear = is_linear

        #use Kaiming Init when using ReLU
        self.W = torch.nn.Parameter(torch.zeros(dim_in, dim_out))
        torch.nn.init.kaiming_normal_(self.W)

        #layer which performs batch normalization
        self.m = torch.nn.LayerNorm(dim_in)



    def forward(self, A, H):
        """
        Forward pass for a GCN Layer.

        :A: Adjecency matrix
        :H: node embeddings of last layer
        :return: node embeddings of this layer
        """
        # linear transformation on input
        x = torch.matmul(A, H)
        # batch normalization

        x = self.m(x)

        y = torch.matmul(x, self.W)
        # apply activation
        if not self.is_linear:
            y = torch.relu(y)
        return y
