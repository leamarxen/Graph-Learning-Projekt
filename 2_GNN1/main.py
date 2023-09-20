from train_graph_GCN import train_graph_GCN
from load_data import load_data
import data_utils
from GCN_modul import GCN_Layer
import graph_level_GCN
import normalized_adj
import adj_matrix
from graph_level_GCN import graph_level_GCN
from load_data_node_level import load_data_node
from node_level_GCN import GCN_node
from train_node_level import train_node_GCN
import argparse
import pickle

""" 
The following code is the main code where the defined kernels and functions are imported and called.
"""
 
#Specified parameters
parser=argparse.ArgumentParser()
parser.add_argument('-p1','--path',required=True,help='Choose the path of the dataset')
parser.add_argument('-p2','--path2',help='Choose the path of the evaluation dataset')
parser.add_argument('-l','--level',required=True,help='Choose the level of Classification')
args=parser.parse_args()


#select the level of classification
#node classification needs two paths
if __name__=='__main__':
    print('start')
    if args.level=='graph':
        A, H, y = load_data(args.path)
        train_graph_GCN(graph_level_GCN, A, H, y)
    if args.level=='node':
        if not args.path2:
            raise Exception('Please choose the evaluation dataset')
        A, B, H, G, y, yt = load_data_node(args.path, args.path2,
                                      node_attrs=True)
        train_node_GCN(GCN_node, A, H, y, B, G, yt)

    else:
        raise Exception('Chosen level does not exist')


