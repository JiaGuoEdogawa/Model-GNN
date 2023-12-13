"""
Code for "A Model-Based GNN for Learning Precoding" by Jia Guo, 2023-11-27
This file includes functions for processing data used for training the GNNs.
For any reproduce, further research or development, please kindly cite our TWC Journal paper:
`J. Guo and C. Yang, “A Model-Based GNN for Learning Precoding,” IEEE Trans. Wireless Commun., accepted, 2023.`
"""

import numpy as np
import pandas as pd

def data_prep(dataset_file, spl_indx, *var_name):
    output = dict()
    for i in range(len(var_name)):
        output[str(i)] = dataset_file[var_name[i]][spl_indx]
    output = tuple(output.values())
    return output


def get_random_block_from_data(batch_size, *var_name):
    output = dict()
    start_index = np.random.randint(0, len(var_name[0]) - batch_size)
    for i in range(len(var_name)):
        output[str(i)] = var_name[i][start_index: start_index + batch_size]
    output = tuple(output.values())
    return output


def standard_scale(X_train, X_test):
    import sklearn.preprocessing as prep
    data_shape_tr = X_train.shape; data_shape_te = X_test.shape
    X_train = np.reshape(X_train, [X_train.shape[0], -1])
    X_test = np.reshape(X_test, [X_test.shape[0], -1])

    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)

    X_train = np.reshape(X_train, data_shape_tr); X_test = np.reshape(X_test, data_shape_te)
    return X_train, X_test