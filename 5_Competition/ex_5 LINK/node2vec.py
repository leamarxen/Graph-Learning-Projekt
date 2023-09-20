import torch


class node2vec(torch.nn.Module):
    """
    Method that computes a Node2Vec embedding
    """
    def __init__(self,input_dim):
        """
        Initialize the Node2Vec 

        :input_dim: length of the graph
        """
        super(node2vec, self).__init__()
        #set column dimension of X to 128
        self.X=torch.nn.Parameter(torch.zeros(input_dim,128))
        torch.nn.init.kaiming_normal_(self.X)
        self.softmax=torch.nn.Softmax(dim=0)

    def forward(self,s,w,neg):
        """
        Forward pass
        
        :s: the starting node of the walk
        :w: the rest of the walk (length self.l-1)
        :neg: negative samples 

        :return:
            loss for each walk
        """
        #put walk and negative samples in one matrix
        temp = self.X[torch.cat((w,neg),1)]
        #scalar product between s and each node (embedding) in X
        temp = torch.matmul(temp, torch.unsqueeze(self.X[s],2))
        temp=torch.squeeze(temp)
        #compute softmax probabilities
        temp=self.softmax(temp)
        #compute loss - only probabilities of the walk (and not the negative samples) relevant
        temp=torch.log(temp[:,:5])
        return -torch.sum(temp,dim=1)

