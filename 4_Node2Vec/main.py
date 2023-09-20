import argparse
import warnings
from logistic_regression_link import logistic_regression_link
from logistic_regression import logistic_regression

""" 
The following code is the main code where the defined type of classification and paramaters of training
"""

# Specified parameters
parser = argparse.ArgumentParser()
parser.add_argument('-path', '--path', type=str,required=True ,help='Choose the path of the dataset')
parser.add_argument('-type', '--type', type=str,required=True,help='Choose the type of classification')
parser.add_argument('-p', '--p', type=float,default=1 ,help='Choose the p of pq_walks')
parser.add_argument('-q', '--q', type=float,default=1 ,help='Choose the q of pq_walks')
parser.add_argument('-lr', '--lr', type=float,default=0.004 ,help='Choose the learning rate')
parser.add_argument('-batch_size', '--batch_size', type=int,default=200 ,help='Choose the batch_size')
parser.add_argument('-epoch', '--epoch', type=int,default=200,help='Choose the number of epochs')
parser.add_argument('-C', '--C', type=float,default=2 ,help='Choose the C of logistic_regreesion')

args = parser.parse_args()



if __name__ == '__main__':
    print('start')
    if args.type not in ['link','node']:
        raise Exception('please choose the right classification type')
    if args.type == 'link':
        if ('Citeseer' or 'Cora') in args.path:
            warnings.warn('This dataset is used for node classification')
        logistic_regression_link(path=args.path,lr=args.lr,batch_size=args.batch_size,C=args.C,epoch=args.epoch)
    if args.type == 'node':
        if ('Facebook' or 'PPI') in args.path:
            warnings.warn('This dataset is used for link classification')
        logistic_regression(path=args.path,lr=args.lr,batch_size=args.batch_size,C=args.C,epoch=args.epoch,p=args.p,q=args.q)



