import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

'''
This file implements an MLP, including the MLP model, training and validation functions.
'''


class MLP(nn.Module):

    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        Initializes an MLP

        :input_size: size of (last dimension of) input  
        :hidden_layer_sizes: of form (x,y,z), because we have 3 hidden layers
        :output_size: number of classes for classification
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_1_size, self.hidden_2_size, self.hidden_3_size = hidden_layer_sizes
        self.num_classes = output_size

        #initialize linear layers and relu
        self.fc1 = nn.Linear(self.input_size, self.hidden_1_size)
        self.fc2 = nn.Linear(self.hidden_1_size, self.hidden_2_size)
        self.fc3 = nn.Linear(self.hidden_2_size, self.hidden_3_size)
        self.fc4 = nn.Linear(self.hidden_3_size, self.num_classes)
        self.relu = nn.ReLU()

        # use dropout
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        

    def forward(self, x):
        """
        forward pass for MLP

        :x: edge embedding for batched edges
       
        :return:
            :x: output of MLP, size: |batch| x output_size
        """

        x = self.relu(self.fc1(x))
        x = self.dropout1(self.fc2(x))
        x = self.relu(x)
        x = self.dropout2(self.fc3(x))
        x = self.relu(x)
        x = self.dropout3(self.fc4(x))
        
        return x


def train_MLP(train_data, train_labels, valid_data, valid_labels, 
                hidden_layer_sizes, epochs=100, batch_size=64, lr=0.004):
    """
    training loop for MLP

    :train_data: embedding for training edges
    :train_labels: labels of training edges
    :valid_data: embedding for validation data
    :valid_labels: labels of validation data
    :hidden_layer_sizes: size of hidden layers, of form (x,y,z) -> 3 hidden layers
    :epochs: number of epochs during training, default=100
    :batch_size: batch size, default=64
    :lr: learning rate, default=0.004
    
    :return:
        :model: MLP, to be used for classification of test data
        :np.mean(acc_score): accuracy of training data (last epoch, mean over all batches)
        :val_score: accuracy of validation data (last epoch)
    """
    #load data
    train_d = TensorDataset(train_data, train_labels)
    val_d = TensorDataset(valid_data, valid_labels)
    val_loader = DataLoader(val_d, batch_size=100)

    # set to 'cuda' if gpu is available
    device = 'cpu'  

    # construct neural network and move it to device
    model = MLP(input_size=train_data.size(1), hidden_layer_sizes=hidden_layer_sizes, output_size=30)
    model.train()
    model.to(device)
    # construct optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.1e-4)
    
    # Training Loop
    for i in range(epochs):
        #re-initialize training accuracy list for each epoch
        acc_score = list()

        #reshuffle at each epoch
        train_loader = DataLoader(train_d, batch_size=batch_size, shuffle=True)
            
        #training
        for X, y_true in train_loader:
            # set gradients to zero
            opt.zero_grad()

            # move data to device
            X = X.to(device)
            y_true = y_true.to(device)

            # forward pass and loss
            y_pred = model(X)
            loss = cross_entropy(y_pred, y_true)
               
            # backward pass and sgd step
            loss.backward()
            opt.step()

            # computation of current accuracy
            y_pred = torch.argmax(y_pred, dim=1)
            acc_score.append(accuracy_score(
                    y_true.tolist(), y_pred.tolist()))

        #compute validation every 5th epoch
        if (i % 5) == 4:
            val_score = validation(model, val_loader, device)

            #if wanted, print training and validation accuracy
            #print(f"epoch {i+1}: AVG training accuracy", np.mean(acc_score), 
            #               "\tvalidation accuracy:", val_score)
        
    
    return model, np.mean(acc_score), val_score

            
def validation(model, val_loader, device):
    """
    validation function for MLP

    :model: MLP model to evaluate
    :val_loader: validation data in DataLoader form
    :device: device to run computation on (CPU, GPU)
    
    :return:
        :accuracy_score: accuracy of validation data
    """
    true_labels = []
    pred_labels = []
    model.eval()
    with torch.no_grad(): #so that no gradients will be changed

        #loop over all data in val_loader and get the predicted labels
        for X, y_true in val_loader:
            #data to device
            X = X.to(device)
            y_true = y_true.to(device)

            #forward pass to classify validation data
            y_pred = model(X)
            #format label data and save them in pred_labels and true_labels respectively
            y_pred_class = torch.argmax(
                    nn.functional.softmax(y_pred, dim=1), dim=1)
            pred_labels.extend(y_pred_class.numpy().tolist())
            true_labels.extend(y_true.numpy().tolist())
    #set model back in training mode
    model.train()
    #compute and return accuracy
    return accuracy_score(true_labels, pred_labels)