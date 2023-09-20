from train_GCN import train_node_GCN
from GCN import GCN_node
from load_data import load_data_node
import argparse

# use argparse to transfer parameters
parser = argparse.ArgumentParser()
parser.add_argument('-path', '--path', type=str,required=True ,help='Choose the path of the dataset')
parser.add_argument('-test_size', '--test_size', type=float,default=0.2,help='Choose the test_size(percentage)')
parser.add_argument('-num_layers', '--layer', type=int ,default=3,help='Choose the number of layers in GCN')
parser.add_argument('-lr', '--lr', type=float,default=0.001,help='Choose the learning rate')
parser.add_argument('-hidden_dim', '--hidden_dim', type=int,default=64,help='Choose the number of hidden_dimensions of GCN')
parser.add_argument('-epochs', '--epochs', type=int,default=100,help='Choose the epochs of training')


args = parser.parse_args()


if __name__ == '__main__':
    A,H,y = load_data_node(train_path=args.path)
    train_node_GCN(GCN_node,A,H,y,lr=args.lr,epochs=args.epochs,hid_dimen=args.hidden_dim,num_layers=args.layer,test_size=args.test_size)


