
import torch
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import accuracy_score
import numpy as np
import random as rd
from sklearn.model_selection import ShuffleSplit
import pickle




def train_node_GCN(clf, A, H, y_true, epochs=100, test_size=0.2, hid_dimen = 128,num_layers = 3,lr=0.001):
    """

    :param clf: GCN
    :param A: adjacency matrix
    :param H: node embeddings
    :param y_true: all given node labels except None-class
    :param epochs:  number of epochs
    :param test_size: size of validation dataset (percentage from 0 to 1)
    :param hid_dimen: dimension of hidden_layers
    :param num_layers: number of layers of GCN
    :param lr: learning rate
    :return:
    accurency score of training data for each fold
    """


    # get number of classification classes
    num_labels = y_true.size(1)

    # set to 'cuda' if gpu is available
    device = 'cpu'

    # list of index of all the nodes
    index_all = [i + 1000 for i in range(len(y_true))]

    # 10-fold ShuffleSplit using the given test_size
    ss = ShuffleSplit(n_splits=10,test_size=test_size,random_state=0)
    k = 0

    result = []

    # Iterate over each training dataset and validation dataset
    for train_idx , eval_idx in ss.split(index_all):

        print(f"########## round {k + 1}/10 ###########")

        # construct neural network and move it to device
        model = clf(input_dim=H.size(1), output_dim=num_labels,hidden_dim=hid_dimen, num_layers=num_layers)
        model.train()
        model.to(device)
        # construct optimizer
        opt = torch.optim.Adam(model.parameters(), lr)

        # adj_matrix for train dataset
        A_train = A.detach().clone()
        A_train[eval_idx] = 0
        A_train[:1000] = 0

        # adj_matrix for validation dataset
        A_eval = A.detach().clone()
        A_eval[train_idx] = 0
        A_eval[:1000] = 0

        # Training Loop
        for i in range(epochs):
            acc_score = list()

            # set gradient to zero
            opt.zero_grad()

            # move data to device
            A_train = A_train.to(device)
            H = H.to(device)
            y_true = y_true.to(device)

            # forward pass and loss
            y_pred = model(A_train, H)

            # ignore the first 1000 predict labels
            y_pred = y_pred[1000:,:]

            # get all the predict labels and target labels
            y_pred_train = y_pred[np.array(train_idx)-1000]

            y_true_train = y_true[np.array(train_idx)-1000]

            y_true_eval = y_true[np.array(eval_idx)-1000]

            # compute the cross_entropy loss
            loss = cross_entropy(y_pred_train,y_true_train )
            print('training loss of {}-epoch'.format(i),loss)


            # backward pass and sgd step
            loss.backward()
            opt.step()

            # computation of current accuracy
            y_pred_class = torch.argmax(y_pred, dim=1)
            y_true_class = torch.argmax(y_true, dim=1)
            acc_score.append(accuracy_score(
                y_true_class.tolist(), y_pred_class.tolist()))
            print(accuracy_score(y_true_class.tolist(), y_pred_class.tolist()))

            #compute validation every 5th epoch
            if (i % 5) == 4:
                val_score = validation(model, A_eval,H,y_true_eval,eval_idx, device)

                print(f"epoch {i + 1}: AVG evaluation accuracy", np.mean(val_score))

        # using the trained model to predict
        predict_result = model(A,H)

        # ingore the first 1000 predict labels
        predict_result = predict_result[:1000]
        class_result = torch.argmax(predict_result,dim=1)

        result.append(list(class_result))

        # str1 = '\n{}-fold output:'.format(k)+str(class_result)
        # with open('result.txt', 'a') as f:
        #     f.writelines(str1)
        # k += 1
        # for each fold, save the training and validation accuracy
        # training_acc.append(acc_score)
        # validation_acc.append(val_score)
    result_true = []

    # select the frequentest label for each node in 10-fold
    for i in range(1000):
        count = []
        for output in result:
            count.append(output[i])
        value = max(set(count),key=count.count)
        result_true.append(int(value))

    str2 = str(result_true)
    with open('CITE-Predictions.pkl', 'wb') as f:
        pickle.dump(result_true,f)

    # print("average training accuracy & standard deviation:", np.mean(training_acc), np.std(training_acc),
    #       f"\t(train. acc. per fold: {training_acc})"
    #       "\naverage validation accuracy & standard deviation:", np.mean(validation_acc), np.std(training_acc),
    #       f"\t(val. acc. per fold: {validation_acc})")


def validation(model,A_eval, H_eval,y_true_eval,index_eval, device):

    """

    :param model: GCN
    :param A_eval: adjacency matrix
    :param H_eval: node_embeddings
    :param y_true_eval: all true labels used for validation
    :param index_eval: index of all node embeddings in validation dataset
    :param device: device
    :return:
    accurency score of validation data
    """
    true_labels = []
    pred_labels = []
    model.eval()
    with torch.no_grad():  # so that no gradients will be changed

        A_eval = A_eval.to(device)
        H_eval = H_eval.to(device)
        index_eval = torch.tensor(index_eval)
        index_eval = index_eval.to(device)
        y_true_eval = y_true_eval.to(device)


        # forward pass to classify validation data
        y_pred_eval = model(A_eval, H_eval)
        y_pred_eval = y_pred_eval[index_eval]
        # format label data and save them in pred_labels and true_labels respectively

        y_pred_class = torch.argmax(y_pred_eval,dim=1)
            # torch.nn.functional.softmax(y_pred_eval, dim=1), dim=1)

        y_true_class = torch.argmax(y_true_eval, dim=1)
        pred_labels.extend(y_pred_class.numpy().tolist())
        true_labels.extend(y_true_class.numpy().tolist())
    model.train()
    return accuracy_score(true_labels, pred_labels)



