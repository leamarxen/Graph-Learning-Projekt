from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from train_node2vec import train_node2vec
from link_prediction import sample_non_edges,element_wise_product,link_prediction
from sklearn.metrics import roc_auc_score


def logistic_regression_link(path,lr,batch_size,epoch,C):
    """

    :param path: path of dataset
    :param lr: learning rate
    :param batch_size: batch_size of training
    :param epoch: number of epochs of training
    :param C: inverse of regularization strength
    :return: accuracy of logistic_regression for link
    """

    #set of train_edges and eval_edges
    E_train, E_eval = link_prediction(path)

    #set of train_non_edges and eval_non_edges
    N_train, N_eval = sample_non_edges(path, len(E_train), len(E_eval) )

    # X is node embeddings for graph
    X = train_node2vec(path, p=1,q=1, l=5,ln=5,epoch=epoch, batch_size=batch_size,lr=lr, edges_to_delete =E_eval)

    # get edge embeddings using element wise product of the connected node embeddings
    edge_train = element_wise_product(X, E_train+N_train)

    # label the edges as 1 and non_edges as 0
    edge_train_true = [1]*len(E_train)+[0]*len(N_train)

    # get non-edge embeddings using element wise product of the connected node embeddings
    edge_eval = element_wise_product(X,E_eval+N_eval)
    edge_eval_true = [1]*len(E_eval)+[0]*len(N_eval)

    clf = LogisticRegression(random_state=0,C=C).fit(edge_train, edge_train_true)

    roc_score = roc_auc_score(edge_eval_true, clf.predict_proba(edge_eval)[:, 1])

    accuracy = accuracy_score(edge_eval_true, clf.predict(edge_eval))
    print("Accuracy of test data:",accuracy,'\nRoc_score:',roc_score)
    return accuracy,roc_score





