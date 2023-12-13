"""
Code for "A Model-Based GNN for Learning Precoding" by Jia Guo, 2023-11-27
This file includes functions of Model-GNNs.
For any reproduce, further research or development, please kindly cite our TWC Journal paper:
`J. Guo and C. Yang, “A Model-Based GNN for Learning Precoding,” IEEE Trans. Wireless Commun., accepted, 2023.`
"""

import tensorflow
if tensorflow.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf
import numpy as np
from common_funcs.PENN_model import ini_weights_highdim, add_2d_pe_layer
from common_funcs.FCNN_model import ini_bias


def PENN_BF(H, layernum):
    """
    :param H: 信道矩阵，形状为（N_spl, K, Ntx, 2）
    :param layernum: 每一层的隐藏层节点数，是一个list
    :return:
    """
    K = int(H.shape[1]); Ntx = int(H.shape[2])
    Hh_Re = tf.transpose(H, [0, 2, 1, 3])[..., 0:1]; Hh_Im = -1 * tf.transpose(H, [0, 2, 1, 3])[..., 1:2]
    Hh = tf.concat((Hh_Re, Hh_Im), axis=3)
    V_temp = Hh
    H = tf.expand_dims(H, axis=1); V_temp = tf.expand_dims(V_temp, axis=1)
    for i in range(len(layernum)):
        Alpha = complex_matmul_tf(H * tf.ones([1, int(V_temp.shape[1]), 1, 1, 1]), V_temp)
        B = complex_matmul_tf(V_temp, Alpha)
        X = tf.concat((V_temp, B), axis=-1)
        X = tf.reshape(tf.transpose(X, [0, 2, 3, 1, 4]), [-1, Ntx, K, int(X.shape[1]*X.shape[-1])])
        V_new = add_2d_pe_layer_new(X, int(X.shape[-1]), 2 * layernum[i], 'penn_bf_layer_' + str(i),
                                    transfer_function=tf.nn.tanh, is_transfer=False)
        V_temp = tf.transpose(tf.reshape(V_new, [-1, Ntx, K, layernum[i], 2]), [0, 3, 1, 2, 4])
        V_temp = V_temp / (tf.sqrt(tf.reduce_sum(tf.pow(V_temp, 2), axis=[-1, -2, -3], keepdims=True)) + 1e-6)

    V_temp = tf.transpose(V_temp[:, 0, ...], [0, 2, 1, 3])

    return V_temp


def complex_matmul_tf(X, Y):
    X_Re = X[..., 0]; X_Im = X[..., 1]
    Y_Re = Y[..., 0]; Y_Im = Y[..., 1]

    XY_Re = tf.matmul(X_Re, Y_Re) - tf.matmul(X_Im, Y_Im)
    XY_Im = tf.matmul(X_Re, Y_Im) + tf.matmul(X_Im, Y_Re)

    XY = tf.concat((tf.expand_dims(XY_Re, axis=-1), tf.expand_dims(XY_Im, axis=-1)), axis=-1)

    return XY


def add_2d_pe_layer_new(input, inshape, outshape, name, stddev=0.1, transfer_function=tf.nn.leaky_relu, is_BN=False,
                    is_transfer=True, k_factor = 1.0, aggr_func=tf.reduce_mean, is_trainBN=False):

    W = ini_weights_highdim([inshape, outshape, 5], name + '_Wh', stddev)
    U = W[:, :, 0]; V = W[:, :, 1]; P = W[:, :, 2]; Q = W[:, :, 3]; M = W[:, :, 4]

    b = ini_bias(outshape, name + '_bias')

    input_sum_dim1 = aggr_func(input, axis=-2, keepdims=True)
    input_sum_dim2 = aggr_func(input, axis=-3, keepdims=True)
    input_sum_dim12 = aggr_func(input, axis=[-2, -3], keepdims=True)

    output = tf.matmul(input, U-V-P+Q) + \
             tf.matmul(input_sum_dim1, V-Q)  + \
             tf.matmul(input_sum_dim2, P-Q) +  \
             tf.matmul(input_sum_dim12, Q) + b

    if is_transfer is True:
        output = transfer_function(output)
    if is_BN is True:
        output = tf.layers.batch_normalization(output, training=is_trainBN, name=name + '_BN',
                                                       reuse=tf.AUTO_REUSE, axis=[-1, -4])
    return output


