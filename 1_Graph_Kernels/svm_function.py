import scipy.sparse as sp
import numpy as np


from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

# Exercise 4

'''
This file defines the Support Vector Machines which will train on the datasets.

Key Idea: 
Make use of 10-fold cross validation to measure the accuracy of each kernel on each dataset. Further, choose 80% of the dataset as trainin
'''

#SVC with 'linear' kernel
#input:features and targets calculated from previous kernel
#output:mean and deviation accuracy of validation data and accuracy of test data
def svm_tt(features,targets):
    X_train, X_test, y_train, y_test = train_test_split(np.array(features),np.array(targets), test_size=0.2)
    clf = SVC(kernel='linear', C=1)
    clf.fit(X_train,y_train)
    accuracy= cross_val_score(clf,X_train, y_train, cv=10 )
    print ("Mean accuracy and standard deviation of training data (10-fold cross validation):", 
        accuracy.mean(), accuracy.std())
    print ("Accuracy of test data:", accuracy_score(y_test,clf.predict(X_test)))

#LinearSVC
#input:features and targets calculated from previous kernel
#output:mean and deviation accuracy of validation data and accuracy of test data
def svm_linear_tt(features,targets):
    X_train, X_test, y_train, y_test = train_test_split(np.array(features),np.array(targets), test_size=0.2)
    clf = LinearSVC(C=1)
    clf.fit(X_train,y_train)
    accuracy= cross_val_score(clf,X_train, y_train, cv=10 )
    print ("Mean accuracy and standard deviation of training data (10-fold cross validation):", 
        accuracy.mean(), accuracy.std())
    print ("Accuracy of test data:", accuracy_score(y_test,clf.predict(X_test)))

#SVC with 'precomputed' kernel
#input:features and targets calculated from previous kernel
#output:mean and deviation accuracy of validation data and accuracy of test data
def svm_precomputed_tt(feat_vecs, target_vec):
    X_train, X_test, y_train, y_test = train_test_split(feat_vecs, target_vec, test_size=0.2, random_state=4)
#in order to use the 'precomputed' kernel,first calculate the gram_matrix
    train_feat = sp.vstack(X_train)
    test_feat = sp.vstack(X_test)
    gram_matrix = train_feat.dot(train_feat.transpose()).todense()
    gram_test = train_feat.dot(test_feat.transpose()).todense().T
    
    clf = SVC(kernel='precomputed')
    clf.fit(np.array(gram_matrix), np.array(y_train))
    accuracy = cross_val_score(clf, np.array(gram_matrix), np.array(y_train), cv=10)
    print ("Mean accuracy and standard deviation of training data (10-fold cross validation):", 
        accuracy.mean(), accuracy.std())
    print("Accuracy of test data:", accuracy_score(y_test, clf.predict(np.array(gram_test))))  
 
