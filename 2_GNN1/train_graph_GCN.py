import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from graph_level_GCN import graph_level_GCN
from load_data import load_data


def train_graph_GCN(clf, data_A, data_H, y_labels, epochs=400, batch_size=100, lr=0.004):
    """
    Trains the graph level GCN.
    :param
        clf: The classifier/model, in our case graph level GCN
        data_A: stacked and padded adjacency matrices of the given graphs
        data_H: stacked and padded vertex embeddings of the given graphs
        y_labels: stacked and one-hot-encoded class labels for the given graphs
        epochs: number of epochs to train
        batch_size: batch size during each epoch
        lr: learning rate
    """
    #load data
    dataset = TensorDataset(data_A, data_H, y_labels)
    #get number of classification classes
    num_labels = y_labels.size(1)

    # set to 'cuda' if gpu is available
    device = 'cpu'

    #apply *Stratified* k-fold Cross-validation to ensure stable learning
    strat_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    training_acc = list()
    validation_acc = list()
     
    
    for fold, (train_idx,val_idx) in enumerate(strat_kfold.split(data_A,  torch.argmax(y_labels, dim=1))):
        print(f"############## FOLD {fold+1}/10 #############")
        
        #sample validation/test data
        val_sampler = SubsetRandomSampler(val_idx)
        val_loader = DataLoader(dataset, batch_size=100, sampler=val_sampler)

        # construct neural network and move it to device
        model = clf(input_dim=data_H.size(2), output_dim=num_labels, num_vertices=data_A.size(1),
                    hidden_dim=64, num_layers=5)
        model.train()
        model.to(device)
        # construct optimizer
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.1e-4)


        # Training Loop
        for i in range(epochs):
            acc_score = list()
            #reshuffle at each epoch
            train_sampler = SubsetRandomSampler(train_idx)
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            
            #training
            for A, H, y_true in train_loader:
                # set gradients to zero
                opt.zero_grad()

                # move data to device
                A = A.to(device)
                H = H.to(device)
                y_true = y_true.to(device)

                # forward pass and loss
                y_pred = model(A, H)
                loss = cross_entropy(y_pred, y_true)

                # backward pass and sgd step
                loss.backward()
                opt.step()

                # computation of current accuracy
                y_pred_class = torch.argmax(
                    nn.functional.softmax(y_pred, dim=1), dim=1)
                y_true_class = torch.argmax(y_true, dim=1)
                acc_score.append(accuracy_score(
                    y_true_class.tolist(), y_pred_class.tolist()))

            
            #compute validation every 5th epoch
            if (i % 5) == 4:
                val_score = validation(model, val_loader, device)

                print(f"epoch {i+1}: AVG training accuracy", np.mean(acc_score), 
                            "\tvalidation accuracy:", val_score)
        
        #for each fold, save the training and validation accuracy
        training_acc.append(np.mean(acc_score))
        validation_acc.append(val_score)
    
    print("average training accuracy & standard deviation:", np.mean(training_acc), np.std(training_acc),
        f"\t(train. acc. per fold: {training_acc})"
        "\naverage validation accuracy & standard deviation:", np.mean(validation_acc), np.std(training_acc),
        f"\t(val. acc. per fold: {validation_acc})")

            
# code in parts from exercise of Text Mining lecture
def validation(model, val_loader, device):
    """
    Gets test/validation data and returns the accuracy score of the given model.
    :param 
        model: the trained model
        val_loader: DataLoader object with the given test data (A, H, labels)
        device: cpu or gpu (where to test)
    :return: accuracy score of the test data given the current model
    """
    true_labels = []
    pred_labels = []
    model.eval()
    with torch.no_grad(): #so that no gradients will be changed

        #loop over all data in val_loader and get the predicted labels
        for data_A, data_H, y_true in val_loader:
            #data to device
            data_A = data_A.to(device)
            data_H = data_H.to(device)
            y_true = y_true.to(device)

            #forward pass to classify validation data
            y_pred = model(data_A, data_H)
            #format label data and save them in pred_labels and true_labels respectively
            y_pred_class = torch.argmax(
                    nn.functional.softmax(y_pred, dim=1), dim=1)
            y_true_class = torch.argmax(y_true, dim=1)
            pred_labels.extend(y_pred_class.numpy().tolist())
            true_labels.extend(y_true_class.numpy().tolist())
    model.train()
    return accuracy_score(true_labels, pred_labels)

# ENZYMES
# A,H,y = load_data("datasets/ENZYMES/data.pkl")
# train_graph_GCN(graph_level_GCN, A, H, y, epochs=400, batch_size=100, lr=0.004)

# NCI1
#A,H,y = load_data("datasets/NCI1/data.pkl")
#train_GCN(graph_level_GCN, A, H, y, epochs=100, batch_size=200, lr=0.004) 
