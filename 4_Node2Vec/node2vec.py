import torch
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
import pickle



class node2vec(torch.nn.Module):
    """
    Method that computes a Node2Vec embedding
    """
    def __init__(self,input_dim):
        """
        Initialize the Node2Vec 

        input_dim: length of the graph
        """
        super(node2vec, self).__init__()
        self.X=torch.nn.Parameter(torch.zeros(input_dim,128))
        torch.nn.init.kaiming_normal_(self.X)
        self.softmax=torch.nn.Softmax(dim=0)

    def forward(self,s,w,neg):
        """
        Forward pass
        
        s =  the starting node of the walk
        w = the rest of the walk (length self.l-1)
        neg: negative samples 
        """
        temp = self.X[torch.cat((w,neg),1)-1]
        temp = torch.matmul(temp, torch.unsqueeze(self.X[s-1],2)) #node starts from 1 but index starts from 0
        # temp=([torch.matmul(self.X[s],torch.transpose(self.X[i],0,1)) for i in torch.cat((w,neg),1)])
        temp=torch.squeeze(temp)
        temp=self.softmax(temp)
        temp=torch.log(temp[:,:5])
        return -torch.sum(temp,dim=0)


#test functionality
#model=node2vec(len(data[0]))


#model.train()
#train_loader = DataLoader(pq_walks(data[0], 1,2,5,5), batch_size=64)
#for s, walk, neg_samples in train_loader:
#    model(s,walk,neg_samples)
