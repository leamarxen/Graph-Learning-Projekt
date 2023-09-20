import networkx as nx
import numpy as np
import random



#Exercise 2


def graphlet_kernel(data):
    '''
    Count the number of randomly sampled graphlets of a graph for all graphs in the dataset.

    Key idea: Sample graphlets with five nodes a thousand times from a given graph and store the number of isomorphic types in a histogram

    input:dataset
    output:list of counter for all the graphs in the dataset
    '''

#create all the non-isoorphic graphs with 5 nodes and store it in a list called 'dic'
    g0=nx.empty_graph(5)
    dic=[0]*34
    dic[0]=nx.create_empty_copy(g0)
    dic[0].add_edges_from([(0,1)])
    dic[1]=nx.create_empty_copy(g0)
    dic[1].add_edges_from([(0,1),(0,2)])
    dic[2]=nx.create_empty_copy(g0)
    dic[2].add_edges_from([(0,1),(2,3)])
    dic[3]=nx.create_empty_copy(g0)
    dic[3].add_edges_from([(0,1),(0,2),(0,3)])
    dic[4]=nx.create_empty_copy(g0)
    dic[4].add_edges_from([(0,1),(0,2),(3,4)])
    dic[5]=nx.create_empty_copy(g0)
    dic[5].add_edges_from([(0,1),(1,2),(2,3)])
    dic[6]=nx.create_empty_copy(g0)
    dic[6].add_edges_from([(0,1),(0,2),(1,2)])
    dic[7]=nx.create_empty_copy(g0)
    dic[7].add_edges_from([(0,1),(0,2),(0,3),(0,4)])
    dic[8]=nx.create_empty_copy(g0)
    dic[8].add_edges_from([(0,1),(0,2),(1,3),(2,3)])
    dic[9]=nx.create_empty_copy(g0)
    dic[9].add_edges_from([(0,1),(0,2),(0,3),(3,4)])
    dic[10]=nx.create_empty_copy(g0)
    dic[10].add_edges_from([(0,1),(0,2),(0,3),(2,3)])
    dic[11]=nx.create_empty_copy(g0)
    dic[11].add_edges_from([(0,1),(1,2),(2,3),(3,4)])
    dic[12]=nx.create_empty_copy(g0)
    dic[12].add_edges_from([(0,1),(0,2),(1,2),(3,4)])
    dic[13]=nx.create_empty_copy(g0)
    dic[13].add_edges_from([(0,1),(0,2),(0,3),(2,4),(3,4)])
    dic[14]=nx.create_empty_copy(g0)
    dic[14].add_edges_from([(0,1),(0,2),(1,2),(1,3),(2,4)])
    dic[15]=nx.create_empty_copy(g0)
    dic[15].add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,2)])
    dic[16]=nx.create_empty_copy(g0)
    dic[16].add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,0)])
    dic[17]=nx.create_empty_copy(g0)
    dic[17].add_edges_from([(0,1),(0,2),(1,2),(1,3),(2,3)])
    dic[18]=nx.create_empty_copy(g0)
    dic[18].add_edges_from([(0,1),(0,2),(2,3),(2,4),(3,4)])
    dic[19]=nx.create_empty_copy(g0)
    dic[19].add_edges_from([(0,1),(0,2),(3,1),(3,2),(4,1),(4,2)])
    dic[20]=nx.create_empty_copy(g0)
    dic[20].add_edges_from([(0,1),(0,2),(1,2),(1,3),(2,4),(3,4)])
    dic[21]=nx.create_empty_copy(g0)
    dic[21].add_edges_from([(0,1),(0,2),(0,3),(1,2),(2,3),(2,4)])
    dic[22]=nx.create_empty_copy(g0)
    dic[22].add_edges_from([(0,1),(0,2),(0,3),(2,3),(2,4),(3,4)])
    dic[23]=nx.create_empty_copy(g0)
    dic[23].add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,2),(3,4)])
    dic[24]=nx.create_empty_copy(g0)
    dic[24].add_edges_from([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
    dic[25]=nx.create_empty_copy(g0)
    dic[25].add_edges_from([(0,1),(0,2),(1,2),(1,3),(2,3),(1,4),(2,4)])
    dic[26]=nx.create_empty_copy(g0)
    dic[26].add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4)])
    dic[27]=nx.create_empty_copy(g0)
    dic[27].add_edges_from([(0,1),(0,2),(0,3),(1,2),(2,3),(1,4),(3,4)])
    dic[28]=nx.create_empty_copy(g0)
    dic[28].add_edges_from([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),(1,4)])
    dic[29]=nx.create_empty_copy(g0)
    dic[29].add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)])
    dic[30]=nx.create_empty_copy(g0)
    dic[30].add_edges_from([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3),(1,4),(2,4)])
    dic[31]=nx.create_empty_copy(g0)
    dic[31].add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4)])
    dic[32]=nx.create_empty_copy(g0)
    dic[32].add_edges_from([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)])
    dic.insert(0,nx.create_empty_copy(g0))

#make a initial dict called 'vektor' to count the non-isoorphic graphs
    vektor=dict(zip(dic,[0]*34))
    output=[]
    
#to check which graphlet the induced subgraph is isomorphic to,and plus 1.
#input:the induced subgraph

    def count_graphlet(g):
        for k,v in temp.items():
            if nx.is_isomorphic(k,g):
                temp[k]+=1
                break
    
#iterate over all graphs in the dataset
    for graph in data:
        temp=vektor.copy()
#if the number of nodes of the gragh is less than 5,then output a vektor with zeros,because can't be isomorphic graph.
        if len(graph.nodes())<5:
            output.append(list(temp.values()))
        else:
#if the number of nodes is more than 5,randomly sample 1000 times.
            for j in range(1000):
                temp_subgraph=graph.subgraph(random.sample(graph.nodes(),5))
                count_graphlet(temp_subgraph)
            output.append(list(temp.values()))
    return output
