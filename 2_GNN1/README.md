# Graph Learning
This readme is used to describe how to run our code and the results we obtained in each exercise.

## Structure of the repository of exercise 2 - TODO

The repository contains several different files: \
\
&nbsp; &nbsp; 0. main.py: main file to run other codes. \
&nbsp; &nbsp; 1. data_utils.py: This file was given and not changed by us. \
&nbsp; &nbsp; 2. normalized_adj.py: This file contains the adjacency normalization computation of exercise 1. \
&nbsp; &nbsp; 3. GCN_modul.py: This file implements a GCN layer described in exercise 2.\
&nbsp; &nbsp; 4. graph_level_gcn.py: This file implements a graph level GCN described in exercise 3.\
&nbsp; &nbsp; 5. load_data.py: This file's task is to load all the necessary data and put it in the right format to train the GCN. \
&nbsp; &nbsp; 6. train_graph_GCN: This file contains a function with the training loop and an evaluation function for the graph level GCN. \
&nbsp; &nbsp; 7. load_data_node_level: This file's task is to load all the necessary data for the node level classification and put it in the right format to train the GCN. \
&nbsp; &nbsp; 8. node_level_GCN: This file implements a node level GCN described in exercise 4.\
&nbsp; &nbsp; 9. train_node_level: This file contains a function with the training loop and an evaluation function for the node level GCN. \
&nbsp; &nbsp; 10. adj_matrix: This file contains the adjacency normalization computation of exercise 1. \
## How to run the script - TODO

This script uses argparse. \
\
To run the script it is necessary to call the file 'main.py'. 
It is required to choose the following arguments: 


#### Example for running the script - TODO

The following command should be run in the terminal to call ... (if the folder "datasets" is in the same location as the python file): \
\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;   -- python main.py -p1 datasets/ENZYMES/data.pkl -l graph

The following command should be run in the terminal to call ...: \
\
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;   -- python main.py -p1 datasets/Citeseer_Train/data.pkl -p2 datasets/Citeseer_Eval/data.pkl -l node

## Results



### Graph-Level GCN

|                             | ENZYMES            |  NCI1            |
|-----------------------------|--------------------|------------------|
|**train data mean accuracy** |  0.6376             |   0.7620         | 
|**train data standard dev.** |  0.028             |    0.020         | 
|**test data accuracy**       |  0.4251             |    0.7326         | 
|**test data standard dev.** |   0.028            |     0.020        |


### Node-Level GCN


|                             | Cora               |  Citeseer            |
|-----------------------------|--------------------|------------------|
|**train data mean accuracy** |     0.8014          |    0.8313        | 
|**train data standard dev.** |     0.072        |       0.047      | 
|**test data accuracy**       |     0.5168          |    0.4787         | 
|**test data standard dev.** |      0.072        |   0.047         |

