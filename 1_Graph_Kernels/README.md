# Graph Learning
This readme is used to describe how to run our code and the results we obtained in each exercise.

## Structure of the repository

The repository contains five different files: \
\
&nbsp; &nbsp; 1. closed_walk_kernel.py: This file contains the Closed Walk Kernel of exercise 1. \
&nbsp; &nbsp; 2. graphlet_kernel.py: This file contains the Graphlet Kernel of exercise 2. \
&nbsp; &nbsp; 3. wl_kernel.py: This file contains the Weisfeiler-Leman-Kernel of exercise 3.\
&nbsp; &nbsp; 4. svm_function.py: This file contains the Support Vector Machine of exercise 4.\
&nbsp; &nbsp; 5. arg_code_ex1.py: This is the main code where the defined kernels and functions (1.-5.) are imported and called.

## How to run the script

This script uses argparse. \
\
To run the script it is necessary to call the file 'arg_code_ex1.py'. It is required to choose the kernel and the dataset of interest. The arguments '-k' and '-P' are implemented to adress the kernels and paths respectively. The kernel of interest can be chosen with the filename (without filename extension), while the dataset can be chosen with the pathname. \
Further, there is an additional optional argument '-eval' which runs the Support Vector Machine. 


#### Example for running the script

The following command should be run in the terminal to call the Closed Walk Kernel with the dataset 'Enzymes' (if the folder "datasets" is in the same location as the python file): \
\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;   -- python arg_code_ex1.py -k closed_walk_kernel -P datasets/ENZYMES/data.pkl

The following command should be run in the terminal to call the Closed Walk Kernel with the dataset 'Enzymes' and then perform graph classification with a Support Vector Machine: \
\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;   -- python arg_code_ex1.py -k closed_walk_kernel -P datasets/ENZYMES/data.pkl -eval svm

## Notes on the Exercises

### Ex.1: Choice of maximal length l

Our goal was to find a variable which takes into account the size of the respective graphs. After having considered several graph characteristics, like the diameter, minimum or maximum number of nodes, we chose l to be the mean number of nodes of the respective graph collections, because the other attributes are either too high (maximum number of nodes), too low (minimum number of nodes) or gave an infinite path length because the datasets contain graphs which are not connected (diameter). Our choice ensures a suitable balance between information and complexity. This yields the results DD: 284, ENZYMES: 32 and NCI: 29 (rounded down).

### Ex.4: Train/Test split and Gram Matrix

In order to ensure reliable and independent results for the SVM, we performed a train-test-split of the graph data (train: 80%, test: 20%). We then trained a classifier on the training data and computed the 10-fold cross-validation on the training data. The testing data was evaluated separately.

As the feature vectors of the WL-kernel are very large and sparse, we used the option "kernel=precomputed" in the SVM and used the gram matrix of the feature vectors as input. For the other two kernels, we used the "raw" feature vectors as input to the SVM.

## Results

### DD

|                             | Closed Walk Kernel |  Graphlet Kernel | WL-Kernel |
|-----------------------------|--------------------|------------------|-----------|
|**train data mean accuracy** | 0.593              | 0.744            | 0.789     |
|**train data standard dev.** | 0.004              | 0.021            | 0.046     |
|**test data accuracy**       | 0.559              | 0.707            | **0.822** |


### ENZYMES

|                             | Closed Walk Kernel |  Graphlet Kernel | WL-Kernel |
|-----------------------------|--------------------|------------------|-----------|
|**train data mean accuracy** | 0.187              | 0.2625           | 0.517     |
|**train data standard dev.** | 0.034              | 0.053            | 0.084     |
|**test data accuracy**       | 0.142              | 0.175            | **0.492** |


### NCI1

|                             | Closed Walk Kernel |  Graphlet Kernel | WL-Kernel |
|-----------------------------|--------------------|------------------|-----------|
|**train data mean accuracy** | 0.510              | 0.610            | 0.815     |
|**train data standard dev.** | 0.061              | 0.021            | 0.016     |
|**test data accuracy**       | 0.533              | 0.658            | **0.813** |


Our results show that the Weisfeiler-Leman-Kernel performs best throughout all datasets. This is not surprising, as the WL-Kernel is the most sophisticated kernel we have used so far. Compared to the paper *Weisfeiler-Lehman Graph Kernels*, our WL-kernel achieved equal accuracy as the paper on all three datasets respectively (paper NCI1: 82.19 (&pm; 0.18), DD: 79.78 (&pm;0.36), ENZYMES 46.42 (&pm;1.35)).

It is also interesting to note that a (3-)Graphlet Kernel appeared in the paper as a reference Kernel, but both in our calculations and in the paper, the WL-Kernel always outperformed the Graphlet Kernel (albeit sometimes only slightly). The closed walk, on the other hand, does not have enough explanatory power to achieve competitive results, so it also does not appear in the paper.
