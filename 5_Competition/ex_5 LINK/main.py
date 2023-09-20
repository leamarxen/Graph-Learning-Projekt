import argparse
import warnings
import pickle
from classification import classification

""" 
The following code is the main code where the defined type of classification and paramaters of training
"""

# Specified parameters
parser = argparse.ArgumentParser()
parser.add_argument('-path', '--path', type=str, required=True, help='Choose the path of the dataset')
parser.add_argument('-input_perc', '--input_perc', type=float, default=0.5, 
                        help='Choose percentage of edges that should be used as input for the classification')
parser.add_argument('-validation_perc', '--validation_perc', type=float, default=0.1, 
                        help='Choose percentage of edges that should be used as validation for the classification')
parser.add_argument('-hidden_layer_sizes', '--hidden_layer_sizes', type=tuple, default=(128,64,32), 
                                    help='Choose size of three hidden layers of the classification MLP')
parser.add_argument('-lr', '--lr', type=float, default=0.004, help='Choose the learning rate for MLP')
parser.add_argument('-batch_size', '--batch_size', type=int, default=64, help='Choose the batch_size for MLP')
parser.add_argument('-epoch', '--epoch', type=int, default=100, help='Choose the number of epochs for MLP')
parser.add_argument('-save_matrix', '--save_matrix', type=bool, default=False, 
                            help='Want to use a precomputed node embedding matrix instead of Node2Vec embedding (has to be of hidden size 128)')


args = parser.parse_args()



if __name__ == '__main__':
    print('start')
    if 'LINK' not in args.path:
        warnings.warn('This dataset is used for link classification of the LINK dataset')
    _, _, pred = classification(path=args.path, input_perc=args.input_perc, validation_perc=args.validation_perc,
        hidden_layer_sizes=args.hidden_layer_sizes, epochs_mlp=args.epoch, batch_size_mlp=args.batch_size,
        lr_mlp=args.lr, save_matrix=args.save_matrix)

    with open("LINK-Predictions.pkl", "wb") as f:
        pickle.dump(pred, f)



