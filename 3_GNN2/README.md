# Graph Learning
This readme is used to describe how to run our code and the results we obtained in each exercise.

## Structure of the repository of exercise 3

The repository contains several different files: \
\
&nbsp; &nbsp; 0. main.py: main file to run other codes. \
&nbsp; &nbsp; 1. custom_Dataset.py: This file contains the custom dataset class of exercise 1. \
&nbsp; &nbsp; 2. collate_graphs.py: This file contains the collate function for exercise 2. \
&nbsp; &nbsp; 3. GNN_Layer.py: This file implements a GNN layer as described in exercise 3.\
&nbsp; &nbsp; 4. Sparse_Sum_Pooling.py: This file implements a sparse sum pooling layer as described in exercise 4.\
&nbsp; &nbsp; 5. Virtual_Node.py: This file implements a Virtual Node Modul as described in exercise 5. \
&nbsp; &nbsp; 6. GNN.py: This file implements a GNN with all its different layers. \
&nbsp; &nbsp; 7. Train_GNN.py: This file contains a function with the training loop and an evaluation function for the GNN. \


## How to run the script

This script uses argparse. \
\
To run the script it is necessary to call the file 'main.py'. 
It is required to choose the following arguments: \
\
REQUIRED ARGUMENTS
- training data path (-ptr): path to get the training data
- validation data path (-pv): path to get the validation data
- test data path (-pte): path to get the test data \
\
OPTIONAL ARGUMENTS
- dimension of hidden layers (-dim), default=250
- type of aggregation (-type), choose between sum, mean and max, default="sum"
- number of layers (-layers), default=5
- dropout rate (-drop_out), default=0.0
- if virtual nodes should be used (-virtual), default=False
- number of training epochs (-epochs), default=200
- batch size during training (-size), default=100
- learning rate for training (-lr)', default=0.004


## Results

notes:The marked result meets the requirements in exercise.

### Sum Aggregation

|                  | version 1 | version 2 | version 3 | version 4 | version 5 | version 6 | **version 7** |
|------------------|-----------|-----------|----------|------------|-----------|-----------|-----------|
| hidden dimension |  50       | 50        | 50       |  50        | 50        | 50        | **250**       |
| number of layers |  3        |  5        | 3        |  7         | 7         | 7         | **5**         |
| epochs           |  100      |  150      | 200      |  200       | 200       | 200       | **200**       |
| batch size       |  100      |  100      | 100      |  100       | 100       | 100       | **100**       |
| learning rate    |  0.001    |  0.004    | 0.01     |  0.004     | 0.004     | 0.004     | **0.004**     |
| dropout rate     |  0.0      |  0.0      | 0.3      |  0.0       | 0.0       | 0.3       | **0.0**       |
| virtual nodes    |  False    |  False    | False    |  False     | True      | False     | **False**     |
|------------------|-----------|-----------|----------|------------|-----------|-----------|-----------|  
|**train data MAE** | 0.3004   |  0.1848   | 0.5151   |  0.1754    | 0.5480    | 0.4109    | **0.1197**    |
|**validation data MAE** | 0.3968 | 0.2415 | 0.6295   |  0.2276    | 0.4771    | 0.6582    | **0.1925**    |
|**test data MAE** |   -       |  -        | -        |  0.2220    | 0.5072    | 0.7016    | **0.1896**    |


### Mean Aggregation

|                  | version 1 | version 2 | version 3 | version 4 | version 5 | version 6 | version 7 |
|------------------|-----------|-----------|----------|------------|-----------|-----------|-----------|
| hidden dimension |  50       | 50        | 50       |  50        | 50        | 50        | 150        |
| number of layers |  3        |  5        | 3        |  7         | 7         | 7         | 4         |
| epochs           |  100      |  150      | 200      |  200       | 200       | 200       | 200       |
| batch size       |  100      |  100      | 100      |  100       | 100       | 100       | 100       |
| learning rate    |  0.001    |  0.004    | 0.01     |  0.004     | 0.004     | 0.004     | 0.004     |
| dropout rate     |  0.0      |  0.0      | 0.3      |  0.0       | 0.0       | 0.3       | 0.3       |
| virtual nodes    |  False    |  False    | False    |  False     | True      | False     | True      |
|------------------|-----------|-----------|----------|------------|-----------|-----------|-----------|  
|**train data MAE** | 0.3537   |  0.2480   | 0.5669   |  0.2083    | 0.6560    | 0.4229    | 0.4146    |
|**validation data MAE** | 0.3642 | 0.2993 | 0.5920   |  0.2765    | 0.6286    | 0.5976    | 0.4905    |
|**test data MAE** |   -       |  -        | -        |  0.3033    | 0.6819    | 0.6718    | 0.5413    |


### Max Aggregation

|                  | version 1 | version 2 | version 3 | version 4 | version 5 | version 6 | version 7 |
|------------------|-----------|-----------|----------|------------|-----------|-----------|-----------|
| hidden dimension |  50       | 50        | 50       |  150       | 150        | 150       | 150       |
| number of layers |  3        |  3        | 3        |  4         | 4         | 4         | 4         |
| epochs           |  150      |  150      | 200      |  100       | 100       | 100       | 100       |
| batch size       |  100      |  100      | 100      |  100       | 100       | 100       | 100       |
| learning rate    |  0.001    |  0.004    | 0.01     |  0.004     | 0.004     | 0.004     | 0.004         |
| dropout rate     |  0.0      |  0.0      | 0.3      |  0.0       | 0.0       | 0.3       | 0.3       |
| virtual nodes    |  False    |  False    | False    |  True      | False     | True      | False     |
|------------------|-----------|-----------|----------|------------|-----------|-----------|-----------|  
|**train data MAE** | 0.3164   |  0.3002   | 0.3946   |  0.3455    | 0.2349    | 0.4535    | 0.4373    |
|**validation data MAE** | 0.3422 | 0.3384 | 0.4256   |  0.3879    | 0.3246    | 0.5918    | 0.4390    |
|**test data MAE** |   -       |  -        | -        |  0.4281    | 0.3632    | 0.6571    | 0.4986    |
