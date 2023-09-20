from sklearn.linear_model import LogisticRegression
import pickle
import torch 
from sklearn.model_selection import train_test_split, cross_val_score
from train_node2vec import train_node2vec
from node2vec import node2vec
from sklearn.metrics import accuracy_score



def logistic_regression(path,p,q,lr,batch_size,epoch,C):
    """

    :param path: path of dataset
    :param p: p of pq_walks
    :param q: q of pq_walks
    :param lr: learning rate
    :param batch_size: batch_size of training
    :param epoch: number of epochs of training
    :param C: inverse of regularization strength
    :return: accuracy of logistic_regression
    """
    #open file
    with open(path, 'rb') as f:
        Graph = pickle.load(f)

    #get class of node for every node(targets)
    y = [Graph[0].nodes[i]['node_label'] for i in Graph[0]]

    #get node embeddings for every node(features)
    X = train_node2vec(path, p=p, q=q, l=5, ln=5, lr=lr,batch_size=batch_size,epoch=epoch )

    X = X.numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    clf = LogisticRegression(random_state=0,C=C).fit(X_train, y_train)

    #10-fold cross validation
    accuracy= cross_val_score(clf, X_train, y_train, cv=10)

    print ("Mean accuracy and standard deviation of training data (10-fold cross validation):", 
        accuracy.mean(), accuracy.std())
    print ("Accuracy of test data:", accuracy_score(y_test,clf.predict(X_test)))


