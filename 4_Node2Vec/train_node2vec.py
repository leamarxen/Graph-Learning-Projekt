from node2vec import node2vec
import pickle
import torch
from torch.utils.data import DataLoader
from random_walks import pq_walks


def train_node2vec(datapath,p,q,l,ln,epoch=300,batch_size=64,lr=0.002, edges_to_delete = None):

    """
    Trains the node2vec

    :param
        :datapath: Data for training data
        :p: paramter p (of pq walk) given in the exercise
        :q: paramter q (of pq walk) given in the exercise
        :l: walk length
        :ln: negative sample size
        :epoch: number of epochs
        :batch_size: batch size
        :lr: learning rate
        edges_to_dete: edges to delete
    """
    with open(datapath, 'rb') as f:
        data = pickle.load(f)

    #only one graph but stored in the list
    data = data[0]

    if edges_to_delete:
        data.remove_edges_from(edges_to_delete)

    device='cpu'

    train_dataset=pq_walks(data, p, q, l, ln)

    model=node2vec(len(data))
    model.train()

    model.to(device)

    opt=torch.optim.Adam(model.parameters(),lr=lr)

    lossl1=torch.nn.L1Loss()

    #Training Loop

    for i in range(epoch):

        train_loader = DataLoader(train_dataset, batch_size=batch_size)

        for s, walk, neg_samples in train_loader:

            opt.zero_grad()

            s=s.to(device)
            walk=walk.to(device)
            neg_samples=neg_samples.to(device)

            y_pred=model(s,walk,neg_samples)

            loss=lossl1(y_pred,torch.zeros(len(y_pred)))

            loss.backward()

            opt.step()


        #print(f'loss for the {i+1}th epoch:',loss)

    return [x.data for x in model.parameters()][0]


#train_node2vec('/Users/haron/Desktop/GraphLearning/Ex04/datasets/Citeseer/data.pkl',1,2,5,5)
