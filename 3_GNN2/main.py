import argparse
from Train_GNN import train_GNN


""" 
The following code is the main code where the defined kernels and functions are imported and called.
"""

# Specified parameters
parser = argparse.ArgumentParser()
parser.add_argument('-ptr', '--p_train', required=True, help='Choose the path of the train dataset')
parser.add_argument('-pv', '--p_val', required=True,help='Choose the path of the validation dataset')
parser.add_argument('-pte', '--p_test', required=True, help='Choose the path of test dataset')
parser.add_argument('-dim', '--dim', type=int,default=250,help='Choose the dimension of hidden layers(from 50 to 500)')
parser.add_argument('-type', '--type',default="sum", help='Choose the type of aggregation(max,sum,mean)')
parser.add_argument('-layers', '--layers',type=int,default=5, help='Choose the number of layers(from 3 to 10)')
parser.add_argument('-drop_out', '--drop_out',type=float,default=0.0, help='Choose the drop_out(from 0.0 to 1.0)')
parser.add_argument('-virtual', '--virtual',default=False, help='Set to True if virtual nodes are required')
parser.add_argument('-epochs', '--epochs',type=int,default=200, help='Choose the epochs for traning')
parser.add_argument('-size', '--size', type=int,default=100,help='Choose the batch_size')
parser.add_argument('-lr', '--lr', type=float,default=0.004 ,help='Choose the learning rate')
args = parser.parse_args()



if __name__ == '__main__':
    print('start')
    if args.dim<50 or args.dim>500:
        raise Exception('please choose the right dimension of hidden layers')
    if args.layers<3 or args.layers>10:
        raise Exception('please choose the right number of layers')
    if args.type not in ['sum','mean','max']:
        raise Exception('please choose the right aggregation type')
    train_GNN(args.p_train,args.p_val,args.p_test,args.dim,args.type,args.layers,args.drop_out,args.virtual,args.epochs,args.size,args.lr)


