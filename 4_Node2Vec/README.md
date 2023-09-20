# Graph Learning
This readme is used to describe how to run our code and the results we obtained in each exercise.

## Structure of the repository of exercise 4

The repository contains several different files: \
\
&nbsp; &nbsp; 0. main.py: main file to run other codes. \
&nbsp; &nbsp; 1. random_walks.py: This file implements a dataset that samples random pq-walks as described in exercise 1. \
&nbsp; &nbsp; 2. node2vec.py: This file contains the Node2Vec embedding for exercise 2. \
&nbsp; &nbsp; 3. train_node2vec.py: This file contains a function with the training loop for the Node2Vec embedding.\
&nbsp; &nbsp; 4. logistic_regression.py: This file implements perform node classiﬁcation as described in exercise 3.\
&nbsp; &nbsp; 5. logistic_regression_link.py: This file uses the Node2Vec implementation to perform link prediction as described in exercise 4. \
&nbsp; &nbsp; 6. link_prediction.py: This file contains functions which are used in 'logistic_regression_link.py'. 

## How to run the script
This script uses argparse.

To run the script it is necessary to call the file 'main.py'. 
It is required to choose the following arguments: \
\
REQUIRED ARGUMENTS
- data path (-path): path of the dataset
- type of classiﬁcation (-type): choose between node or link classiﬁcation\
\
OPTIONAL ARGUMENTS
- p of pq_walks (-p), default=1
- q of pq_walks (-q), default=1
- inverse of regularization strength, smaller values specify stronger regularization(-C),default=2
- number of training epochs (-epoch), default=200
- batch size during training (-batch_size), default=200
- learning rate for training (-lr)', default=0.004

NOTES: p,q could be set when using node classiﬁcation.But for link classiﬁcation,are specified as 1.\
WARNING: There is a warning when using the inappropriate dataset, but does not interrupt the program.
## Results
### Node Classiﬁcation

|         Cora dataset     | version 1 | version 2 | version 3 | 
|------------------|-----------|-----------|----------|
| epochs           |  **200**  | 200       | 200      |
| batch size       |  **64**   |  64       | 64       |
| lr               |  **0.02** |  0.02     | 0.02     |
| p                |  **1**    |  1        | 0.1      |
| q                |  **1**    |  0.1      | 1        |
| C                |  **2**    |  2        | 2        |
|------------------|-----------|-----------|----------|
|mean accuracy     | **0.74972**|  0.72158  | 0.72943 | 
|std dev           | **0.02790**| 0.02969 | 0.02636   |  
|Accuracy for evaluation data |   **0.76383**      |  0.70664        | 0.74723       |  

|         Citeseer dataset     | version 1 | version 2 | version 3 | 
|------------------|-----------|-----------|----------|
| epochs           |  300  |  300       | 300      |
| batch size       |  100   |  64       | 64       |
| lr               |  0.004 |  0.002    | 0.004     |
| p                |  1    |  1        | 0.1      |
| q                |  1    |  0.1      | 1        |
| C                |  1    |  1        | 2        |
|------------------|-----------|-----------|----------|
|mean accuracy     | 0.58023   |  **0.59116**  | 0.56927 | 
|std dev           | 0.02483   | **0.04004** | 0.020246   |  
|Accuracy for evaluation data | 0.57918 |   **0.60181**      | 0.57466      |  



### Link Prediction

With the parameters \
&nbsp; &nbsp; epochs: 300,\
&nbsp; &nbsp; batch size: 300,\
&nbsp; &nbsp; lr: 0.007, \
&nbsp; &nbsp; C: 3\
we achieved the following result on the Facebook dataset:
|        Facebook    | Round 1 | Round 2 | Round  3 | Round 4 | Round 5 |-|
|----------------|-----------|-----------|----------|------------|-----------|------------|
|Accuracy test data |  0.90548  | 0.90406   | 0.90471  |  0.90613   |0.90650    |----------|
| Roc_score         |  0.94994  |  0.94747  | 0.95005  |  0.95204   | 0.95108   |---------|
|Mean Accuracy|---------|---------|---------|---------|--------|0.90538    |
|std dev|---------|---------|---------|---------|--------|0.00089     |

With the parameters \
&nbsp; &nbsp; epochs: 300,\
&nbsp; &nbsp; batch size: 300,\
&nbsp; &nbsp; lr: 0.006, \
&nbsp; &nbsp; C: 3\
we achieved the following result on the PPI dataset:

|        PPI    | Round 1 | Round 2 | Round  3 | Round 4 | Round 5 |-|
|----------------|-----------|-----------|----------|------------|-----------|------------|
|Accuracy test data |  0.69865  | 0.69885   | 0.70078  |  0.70253   |0.70253    |----------|
| Roc_score         |  0.75452  |  0.75622  | 0.76041  |  0.75866   | 0.75836   |---------|
|Mean Accuracy|---------|---------|---------|---------|--------|0.70067    |
|std dev|---------|---------|---------|---------|--------|0.00204     |



