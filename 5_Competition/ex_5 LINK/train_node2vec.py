from node2vec import node2vec
import torch
from torch.utils.data import DataLoader
from random_walks import pq_walks

def train_node2vec(data,p,q,l,ln,epoch=300,batch_size=64,lr=0.002):

    """
    Trains the node2vec

    :param
        :data: graph on which we want to train node2vec
        :p: 1/p is probability for the second last node to be sampled again
        :q: 1/q is probability for "unknown" nodes to be sampled
        :l: walk length
        :ln: negative sample size
        :epoch: number of epochs for training
        :batch_size: batch size
        :lr: learning rate
    
    :return:
        node embedding matrix (for data=(V,E), size: |V|x128)
    """

    device='cpu'

    #initialize dataset and model
    train_dataset=pq_walks(data, p, q, l, ln)
    model=node2vec(len(data))
    model.train()

    model.to(device)

    opt=torch.optim.Adam(model.parameters(),lr=lr)

    #Training Loop

    for i in range(epoch):

        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        for s, walk, neg_samples in train_loader:

            opt.zero_grad()

            #put data to device
            s=s.to(device)
            walk=walk.to(device)
            neg_samples=neg_samples.to(device)

            #compute loss and backpropagate
            loss_batch=model(s,walk,neg_samples)
            loss = torch.mean(loss_batch)

            loss.backward()
            opt.step()

        #if wanted, can print loss for each epoch
        #print(f'loss for the {i+1}th epoch:',loss)

    return [x.data for x in model.parameters()][0]



