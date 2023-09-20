import pickle
import argparse

""" 
The following code is the main code where the defined kernels and functions are imported and called.
"""

# import defined kernels and functions
from closed_walk_kernel import closed_walk_kernel
from graphlet_kernel import graphlet_kernel
from wl_kernel import wl_kernel
from svm_function import svm_precomputed_tt, svm_linear_tt,svm_tt

# Create arguments which are needed for the command line
parser = argparse.ArgumentParser()
parser.add_argument('-k', '--kernel', required = True, help='Choose the kernel of interest')
parser.add_argument('-P', '--path', required = True, help='Choose the path of the dataset of interest')
parser.add_argument('-eval', '--svm', help='Call if you want to make use of SVM')
args = parser.parse_args()

# load the data
with open(args.path, 'rb') as file:
    data = pickle.load(file)

# 'react' if this file is called 
# run the chosen kernel
# If SVM is called then run it
if __name__ == '__main__':
    print("Computing Kernel")
    if args.kernel == 'closed_walk_kernel':
        feature_vectors = closed_walk_kernel(data)

    elif args.kernel == 'graphlet_kernel':
        feature_vectors = graphlet_kernel(data)

    elif args.kernel == 'wl_kernel':
        feature_vectors = wl_kernel(data)

    else:
        raise Exception("Chosen kernel does not exist :S")

    if args.svm == 'svm':
        print("Computing SVM")
        target_label = [g.graph['label'] for g in data]
        if args.kernel == 'wl_kernel':
            svm_precomputed_tt(feature_vectors, target_label)
        elif args.kernel == 'closed_walk_kernel':
            svm_linear_tt(feature_vectors, target_label)
        else:
            svm_tt(feature_vectors, target_label)









