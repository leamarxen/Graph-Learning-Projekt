import numpy as np
import pickle
import torch

from train_node2vec import train_node2vec
from modify_data import concatenation, edge_split
from MLP import train_MLP


def classification(path, input_perc, validation_perc, hidden_layer_sizes, epochs_mlp=100, 
                                                batch_size_mlp=64, lr_mlp=0.004, save_matrix=False):
    """
    runs classification model (node2vec + MLP) and performs classification of the test data

    :path: path of dataset
    :input_perc: percentage of input edges (i.e. how many edges should be used for input)
    :validation_perc: percentage of validation edges (i.e. how many edges should be used for 
                            validation during the classification) -> rest of edges used for training
    :hidden_layer_sizes: size of hidden layers of MLP, of form (x,y,z) -> 3 hidden layers
    :epochs_mlp: number of epochs during training of MLP
    :batch_size_mlp: batchsize during training of MLP
    :lr_mlp: learning rate during training of MLP
    :save_matrix: can be used if we don't want to recompute node2vec node embedding but use precomputed
                    matrix instead, default=False

    :return:
        :np.mean(accuracy_nn): mean accuracy of validation data
        :np.std(accuracy_nn): standard deviation of validation data
        :list(test_predict): list of predicted test labels
    """
    #load graph
    with open(path, 'rb') as f:
        data = pickle.load(f)

    # X is node embedding for graph nodes
    #if save_matrix=True: work with precomputed matrix
    if save_matrix:
        try:
            with open("saved_matrix.pkl", "rb") as f:
                X = pickle.load(f)
        #if no matrix precomputed yet: compute and store it so that it can be reused the next time
        except:
            X = train_node2vec(data, p=1, q=1, l=5, ln=5, epoch=100, batch_size=100, lr=0.004)
            with open("saved_matrix.pkl", "wb") as f:
                pickle.dump(X, f)
    else:
        X = train_node2vec(data, p=1, q=1, l=5, ln=5, epoch=100, batch_size=100, lr=0.004)
    
    #store accuracy 
    accuracy_train = []
    accuracy_val = []

    #train MLP 5 times and compute mean accuracy in order to get more reliable values
    for fold in range(5):
        #print(f"################ fold {fold+1}/5 #################")
        
        #compute split of edge data
        e_input, e_train, e_train_l, e_val, e_val_l, unlabeled = edge_split(data, 
                            input_percentage=input_perc, validation_percentage=validation_perc)
        e_train_l, e_val_l = torch.tensor(e_train_l), torch.tensor(e_val_l)

        #compute second node embeddings, which are based on the outgoing and incoming edges and 
        # their labels of each node
        N_out = torch.zeros((X.size(0),30))
        N_in = torch.zeros((X.size(0),30))
        for (e_out, e_in), label in e_input:
            N_out[e_out, label] += 1
            N_in[e_in, label] +=1

        # get edge embeddings
        edge_train = concatenation(X, e_train, N_out, N_in)
        edge_val = concatenation(X, e_val, N_out, N_in)
        edge_test = concatenation(X, unlabeled, N_out, N_in)

        #train MLP and get the respective accuracies
        model, acc_train, acc_val = train_MLP(edge_train, e_train_l, edge_val, e_val_l, 
                            hidden_layer_sizes=hidden_layer_sizes, epochs=epochs_mlp, batch_size=batch_size_mlp, lr=lr_mlp)
        accuracy_train.append(acc_train)
        accuracy_val.append(acc_val)

        #if wanted, print accuracy of training and validation data
        #print("Accuracy of training data (nn):", accuracy_train[-1])
        #print("Accuracy of validation data (nn):", accuracy_val[-1])

    #put model in evaluation mode and get labels for test data
    model.eval()
    test_predict = model(edge_test)

    print(f"Accuracy and std of training data: {np.mean(accuracy_train), np.std(accuracy_train)}\n" 
        + f"Accuracy and std of validation data: {np.mean(accuracy_val), np.std(accuracy_val)}")
    return np.mean(accuracy_val), np.std(accuracy_val), list(test_predict)

#test functionality
# import itertools

# hidden = [(100, 100, 60), (128, 64, 32)]
# input_perc = [0.5]

# for hidden_layers, in_perc in itertools.product(hidden,input_perc):
#     model_name = "hidden_layers: " + str(hidden_layers) + " input percentage: " + str(in_perc)
#     mean, std, _ = classification("LINK/data.pkl", input_perc=in_perc, validation_perc=0.1, 
#                             hidden_layer_sizes=hidden_layers, save_matrix=True)
#     print(model_name, mean, std)
#     more_lines = ['',model_name,
#                             "average validation error:", str(mean),
#                             "average test error:", str(std)]
#     with open('models.txt', 'a+') as f:
#         f.writelines('\n'.join(more_lines))
