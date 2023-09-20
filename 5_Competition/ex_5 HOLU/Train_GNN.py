import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import mean_absolute_error
import pickle
import random as rd

from custom_dataset import CustomDataset
from collate_graphs import collate_graphs
from GNN import GNN


def train_GNN(data_path, hidden_dim, aggr_type, num_layers,drop_out,
              virtual_node=False,epochs=400, batch_size=100, lr=0.004):
    """
    Trains the GNN.
    :param
        :data_path: Data path for data 
        :hidden:dim: size of hidden layers
        :aggr_type: aggregation type of interest ("max", "mean" or "sum")
        :num_layers: Number of layers
        :drop_out: Dropout percentage
        :virtual_node: set True for implementing Virtual node, else False 
        :epoch: number of epochs of training
        :batch_size: batch size during each epoch
        :lr: learning rate
    """

    # load the given data 
    data = pickle.load(open(data_path, "rb"))
    test_data = data[:1000] #no_label graphs
    other_data = data[1000:]

    train_ints = list(range(len(other_data)))
    

    for fold in range(5):
        print(f"############## FOLD {fold+1}/5 #############")

        index_train = rd.sample(train_ints,int(len(train_ints)*0.8))

        train_data = [other_data[i] for i in index_train]
        validation_data = [other_data[i] for i in train_ints if i not in index_train]

        # load all data
        train_dataset = CustomDataset(train_data)
        validation_dataset = CustomDataset(validation_data)
        

        # set to 'cuda' if gpu is available
        device = 'cpu'

        training_error = list()
        validation_error = list()
        
        val_loader = DataLoader(validation_dataset, collate_fn = collate_graphs, batch_size=100)
        
        # construct neural network and move it to device
        model = GNN(hidden_dim, aggr_type, num_layers,virtual_node=virtual_node, node_attrs = 36, edge_attrs = 5,drop_out=drop_out)
        model.train() # put model in training mode 
        model.to(device)
        # construct optimizer and loss function
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.1e-4)
        lossL1 = nn.L1Loss()

        val_score = 1
        # Training Loop
        for i in range(epochs):
            acc_score = list()
            #reshuffle at each epoch
            train_loader = DataLoader(train_dataset, collate_fn = collate_graphs, batch_size=batch_size, shuffle = True)
            

            #training
            for edge_list, node_features, edge_features, graph_label, batch_idx in train_loader:
                # set gradients to zero
                opt.zero_grad()

                # move data to device
                edge_list = edge_list.to(device)
                node_features = node_features.to(device)
                edge_features = edge_features.to(device)
                graph_label = graph_label.to(device)
                batch_idx = batch_idx.to(device)

                # forward pass and loss
                y_pred = model(node_features, edge_list, edge_features, batch_idx)
                y_pred = torch.squeeze(y_pred)
                loss = lossL1(y_pred, graph_label)

                # backward pass and sgd step
                loss.backward()
                opt.step()

                # computation of current error
                acc_score.append(mean_absolute_error(graph_label.detach().numpy(), y_pred.detach().numpy()))

            # compute val_score for every 5th epoch
            if (i % 5) == 4:
                val_score = validation(model, val_loader, device)

            #print validation every 5th epoch
            if (i % 5) == 4:

                print(f"epoch {i+1}: Training MAE ", np.mean(acc_score),  
                            "\tValidation MAE:", val_score)
    
            #for each epoch, save the training and validation error
            training_error.append(np.mean(acc_score))
            validation_error.append(val_score)

    predicted_labels = test_labeling(test_data, model, device)
    return predicted_labels

    #get test error
    #test_error = validation(model, test_loader, device)
    # torch.save(model.state_dict(), model_name)

    #print("average training error:", training_error[-1],
    #     "\naverage validation error:", validation_error[-1])
    

# code in parts from exercise of Text Mining lecture
def validation(model, val_loader, device):
    """
    Gets test/validation data and returns the MAE of the given model.
    :param 
        :model: the trained model
        :val_loader: DataLoader object with the given data 
        :device: cpu or gpu (where to test)
    :return: mean absolute error of the test data given the current model
    """
    true_labels = []
    pred_labels = []
    
    #put model to evaluation mode
    model.eval()
    with torch.no_grad(): #so that no gradients will be changed

        #loop over all data in val_loader and get the predicted labels
        for edge_list, node_features, edge_features, graph_label, batch_idx in val_loader:

            # move data to device
            edge_list = edge_list.to(device)
            node_features = node_features.to(device)
            edge_features = edge_features.to(device)
            graph_label = graph_label.to(device)
            batch_idx = batch_idx.to(device)


            #forward pass to classify validation data
            y_pred = model(node_features, edge_list, edge_features, batch_idx)
            #format label data and save them in pred_labels and true_labels respectively

            pred_labels.extend(y_pred.numpy().tolist())
            true_labels.extend(graph_label.numpy().tolist())

    # put model to train mode        
    model.train()
    return mean_absolute_error(true_labels, pred_labels)

def test_labeling(test_data, model, device):

    """
    Predict the graph label of test data
    :param 
        :test_data: Data with no labeling
        :model: the trained model
        :device: cpu or gpu (where to test)
    :return: Predicted graph labels of test data 
    """

    test_dataset = CustomDataset(test_data, test_data= True)

    test_loader = DataLoader(test_dataset, collate_fn = collate_graphs, batch_size=1000)

    pred_labels =[]
    model.eval()

    with torch.no_grad(): #so that no gradients will be changed

        #loop over all data in val_loader and get the predicted labels
        for edge_list, node_features, edge_features, graph_label, batch_idx in test_loader:

            # move data to device
            edge_list = edge_list.to(device)
            node_features = node_features.to(device)
            edge_features = edge_features.to(device)

            batch_idx = batch_idx.to(device)

            #forward pass to classify test data aka predicted graph labels
            y_pred = model(node_features, edge_list, edge_features, batch_idx)
            y_pred = torch.squeeze(y_pred)
            
            #format label data and save them in pred_labels and true_labels respectively
            pred_labels.extend(y_pred.numpy().tolist())
    

    # put model to train mode        
    model.train()
    return pred_labels


predicted_labels=train_GNN("../datasets/HOLU/data.pkl", hidden_dim= 64, aggr_type='sum',num_layers= 7,drop_out= 0.0,
              virtual_node=False,epochs=100, batch_size=100, lr=0.004)

#print(predicted_labels)

#potential hidden_dim values: 64, 128
# num_layers= 5, 7
# drop = 0.0, 0.2

# Goal:  MAE < 0.33

#64,5,0.0
#128,5,0.0
#64,7,0.0