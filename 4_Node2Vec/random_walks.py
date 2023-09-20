import torch
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
import networkx as nx
import random as rd
#from node2vec import Node2Vec
import numpy as np
import pickle

class pq_walks(IterableDataset):
    """
    create a subclass of an iterable dataset
    """

    def __init__(self, G, p, q, l, ln):
        """
        initialize the iterable dataset

        G: one graph
        p: 1/p is probability for the second last node to be sampled again
        q: 1/q is probability for "unknown" nodes to be sampled
        l: length of a sampled walk
        ln: length of a negative sample
        """
        super(pq_walks, self).__init__()
        # set all the necessary variables
        self.l = l
        self.ln = ln
        self.G=G
        self.p = p
        self.q = q


    def one_random_walk(self):
        """
        samples one random walk of length self.l from the graph self.G, as well as random nodes
        of length self.ln

        :return:
            s = walk[0]: the starting node of the walk
            walk[1:]: the rest of the walk (length self.l-1)
            neg: the negative samples of length self.ln
        """
        no_neighbor = True
        # sample a starting node s
        while no_neighbor:
            s = rd.sample(self.G.nodes, 1)[0]
            # sometimes a node only has no neighbors or only itself as neighbor, 
            # then algorithm doesn't work. Filter those cases out here
            if len(self.G[s])==0:
                no_neighbor = True
            elif len(self.G[s])==1 and list(self.G[s].keys())[0]==s:
                no_neighbor = True
            else:
                no_neighbor = False

        # sample the second node with equal probabilities
        v = rd.sample(list(self.G[s]),1)[0]

        walk=[s,v]
        neg=[]
        for _ in range(self.l-1):
            nodes=[s]

            # get neighbors of the last two nodes
            neighbors_s=set(self.G[s]) 
            neighbors_v=set(self.G[v]) 

            # get common neighbors of s and v
            node_1 = list(neighbors_s.intersection(neighbors_v))
            nodes+=node_1

            # get the other neighbors of v (excluding s)
            neighbors_s.add(s)
            node_q = list(neighbors_v.difference(neighbors_s))
            nodes+=node_q

            v = s
            # set the probabilities to be sampled for s, then the common neighbors, then all others
            pro = [1/self.p] +  [1]*len(node_1) + [1/self.q]*len(node_q)
            # choose a neighbor and append
            s=rd.choices(nodes,pro,k=1)[0]
            walk.append(s)

        # sample the negative samples from all the nodes excluding the ones in the walk
        nodes_to_sample_from = list(set(self.G).difference(set(walk)))
        neg = rd.sample(nodes_to_sample_from, self.ln)

        return walk[0], walk[1:], neg

    def __iter__(self):
        """
        implements the iter-function of the iterable dataset
        
        :return: through yield returns iterable of torch tensors 
            s: starting node of random walk
            walk: rest of random walk
            neg_samples: negative samples
        """
        # check if there is more than one worker working on the data
        worker_info = get_worker_info()
        # if worker info was specified, divide the work by the number of workers 
        if worker_info:
            loopsize = int(np.ceil(1000/worker_info.num_workers))
        else:
            loopsize = 1000

        # sample a walk
        for _ in range(loopsize):
            s, walk, neg_samples = self.one_random_walk()
            yield torch.tensor(s), torch.tensor(walk), torch.tensor(neg_samples)
                    

#test functionality

#path='Citeseer/data.pkl'
#with open(path,'rb') as f:
#    data=pickle.load(f)
# train_loader = DataLoader(pq_walks(data[0], 1,2,5,5), batch_size=64)
# for s, walk, neg_samples in train_loader:
#     print("s:", s, "\nwalk:", walk, "\nneg_samples:", neg_samples)

