import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np

from node_level_GCN import GCN_node
from load_data_node_level import load_data_node

def train_node_GCN(clf, data_A, data_H, y_labels, A_test, H_test,y_label_test, epochs=100, batch_size=1, lr=0.001):
    
    """
    Train the GCN and take the output of load_data_node_level as input. Further, we add epochs, batch_size and 
    learning rate for Adam algorithm.
    """
    
    # Dataset wrapping tensor
    train_dataset = TensorDataset(data_A, data_H, y_labels)
    test_dataset = TensorDataset(A_test, H_test, y_label_test)

    #get number of classification classes
    num_labels = y_labels.size(2)

    # set to 'cuda' if gpu is available
    device = 'cpu'

    training_acc = list()
    validation_acc = list()
    
    # load test data
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle =True)
        
    for k in range(10):
        print(f"########## round {k+1}/10 ###########")

        # construct neural network and move it to device
        model = clf(input_dim=data_H.size(2), output_dim=num_labels, num_vertices=data_A.size(1),
                    hidden_dim=64, num_layers=3)
        model.train()
        model.to(device)
        # construct optimizer
        opt = torch.optim.Adam(model.parameters(), lr)
    
        # Training Loop
        for i in range(epochs):
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle =True)
            acc_score = list()
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
                y_pred_class = torch.argmax(y_pred, dim=2)[0]
                y_true_class = torch.argmax(y_true, dim=2)[0]
                acc_score.append(accuracy_score(
                    y_true_class.tolist(), y_pred_class.tolist()))

          
            #compute validation every 5th epoch
            if (i % 5) == 4:
                val_score = validation(model, test_loader, device)
        
                print(f"epoch {i+1}: AVG training accuracy", np.mean(acc_score), 
                                    "\tvalidation accuracy:", val_score)
                
        #for each fold, save the training and validation accuracy
        training_acc.append(acc_score)
        validation_acc.append(val_score)
            
    print("average training accuracy & standard deviation:", np.mean(training_acc), np.std(training_acc),
        f"\t(train. acc. per fold: {training_acc})"
        "\naverage validation accuracy & standard deviation:", np.mean(validation_acc), np.std(training_acc),
        f"\t(val. acc. per fold: {validation_acc})")

    
    

def validation(model, val_loader, device):
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
                    torch.nn.functional.softmax(y_pred, dim=2), dim=2)[0]
            y_true_class = torch.argmax(y_true, dim=2)[0]
            pred_labels.extend(y_pred_class.numpy().tolist())
            true_labels.extend(y_true_class.numpy().tolist())
    model.train()
    return accuracy_score(true_labels, pred_labels)




# A,B,H,G, y, yt = load_data_node("datasets/Cora_Train/data.pkl", "datasets/Cora_Eval/data.pkl",
#                   node_attrs=True)
#
#
# train_node_GCN(GCN_node, A, H, y, B, G, yt)

