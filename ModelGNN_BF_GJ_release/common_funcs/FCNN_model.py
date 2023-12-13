"""
Code for "A Model-Based GNN for Learning Precoding" by Jia Guo, 2023-11-27
This file includes functions of the fully-connected neural networks (FNNs).
For any reproduce, further research or development, please kindly cite our TWC Journal paper:
`J. Guo and C. Yang, “A Model-Based GNN for Learning Precoding,” IEEE Trans. Wireless Commun., accepted, 2023.`
"""

import tensorflow
if tensorflow.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf
import numpy as np

def ini_weights(in_shape, out_shape, name, stddev=0.1, is_train=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
            weight_mat = tf.get_variable(name='weight', shape=[in_shape, out_shape],
                                         initializer=tf.truncated_normal_initializer(mean=0.0, stddev=stddev),
                                         trainable=is_train)
    return weight_mat


def ini_bias(out_shape, name, is_train=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
            bias_vec = tf.get_variable(name='bias', shape=[out_shape],
                                       initializer=tf.constant_initializer(0.1), trainable=is_train)
    return bias_vec


def add_layer(input, inshape, outshape, name, stddev=0.1, keep_prob=1.0, transfer_function=tf.nn.relu, is_transfer=True,
              is_BN=False, is_trainBN=True, is_train=True):
    W = ini_weights(inshape, outshape, name, stddev, is_train=is_train)
    b = ini_bias(outshape, name, is_train=is_train)
    output = tf.matmul(input, W) + b
    output = tf.nn.dropout(output, keep_prob=keep_prob)
    if is_BN is True:
        output = tf.layers.batch_normalization(output, training=is_trainBN, name=name + '_BN', reuse=tf.AUTO_REUSE)
    if is_transfer is True:
        output = transfer_function(output)
    return output, W, b


def fcnn(input, layernum, output_activation=tf.nn.sigmoid, is_BN=False, is_trainBN=False, name='',
         is_lastlayer_transfer=True, is_train=True):
    hidden = dict()
    weight = dict()
    bias = dict()
    if len(layernum) >= 2:
        for i in range(len(layernum)):
            if i == 0:
                hidden[str(i)], weight[str(i)], bias[str(i)] = \
                    add_layer(input, int(input.shape[-1]), layernum[i], name + 'layer' + str(i + 1), is_BN=is_BN,
                              is_trainBN=is_trainBN, is_train=is_train)
            elif i != len(layernum)-1:
                hidden[str(i)], weight[str(i)], bias[str(i)] = \
                    add_layer(hidden[str(i - 1)], layernum[i-1], layernum[i], name + 'layer' + str(i + 1), is_BN=is_BN,
                              is_trainBN=is_trainBN, is_train=is_train)
            else:
                output, weight[str(i)], bias[str(i)] = \
                    add_layer(hidden[str(i - 1)], layernum[i-1], layernum[i], name + 'layer' + str(i + 1), is_BN=is_BN,
                              transfer_function=output_activation, is_trainBN=is_trainBN, is_transfer=is_lastlayer_transfer,
                              is_train = is_train)
    else:
        output, weight['0'], bias['0'] = \
            add_layer(input, int(input.shape[-1]), layernum[0], name + 'layer1', is_BN=is_BN,
                      transfer_function=output_activation, is_trainBN=is_trainBN, is_transfer=is_lastlayer_transfer,
                      is_train=is_train)
    return output, hidden, bias


def fcnn_joint(input, n_obj, layernum, output_activation=tf.nn.sigmoid):
    hidden = dict()
    weight = dict()
    bias = dict()
    n_obj = int(np.sqrt(layernum[-1]))
    if len(layernum) >= 2:
        for i in range(len(layernum)):
            if i == 0:
                hidden[str(i)], weight[str(i)], bias[str(i)] = \
                    add_layer(input, int(input.shape[1]), layernum[i], 'layer' + str(i + 1))
            elif i != len(layernum)-1:
                hidden[str(i)], weight[str(i)], bias[str(i)] = \
                    add_layer(hidden[str(i - 1)], layernum[i-1], layernum[i], 'layer' + str(i + 1))
            else:
                output, weight[str(i)], bias[str(i)] = \
                    add_layer(hidden[str(i - 1)], layernum[i-1], layernum[i], 'layer' + str(i + 1),
                              transfer_function=output_activation)
    else:
        output, weight[str(0)], bias[str(0)] = \
            add_layer(input, int(input.shape[1]), layernum[0], 'layer' + str(1),
                      transfer_function=output_activation)
    output = tf.reshape(tf.reduce_sum(tf.expand_dims(tf.expand_dims(tf.eye(n_obj), axis=0), axis=3)
                                      * tf.reshape(output, [-1, n_obj, n_obj, 1]), axis=1), [-1, n_obj])
    return output, weight, bias


def gen_1d_shade_mat(K, M, N):
    """
    函数功能：生成 K X K 的单位矩阵，每个块是一个 M X N 的子矩阵。生成的矩阵是numpy格式。
    """
    temp = np.reshape(np.expand_dims(np.eye(K), axis=2) * np.ones([1, 1, N]), [K, K * N])
    shade_mat = np.reshape(np.expand_dims(temp, axis=1) * np.ones([1, M, 1]), [K*M, K*N])
    return shade_mat


def gen_2d_shade_mat(K, L, M, N):
    """
    函数功能：生成两个矩阵。第一个矩阵：K X K 的单位矩阵，每个块是 (L X M) X (L X N)的子矩阵。
    第二个矩阵：K X K 的矩阵， 每个块是 L X L的单位子矩阵，每个子块是 M X N的小矩阵。
    """
    shade_mat_a = gen_1d_shade_mat(K, L*M, L*N)
    shade_mat_b = np.tile(gen_1d_shade_mat(L, M, N), [K, K])
    return shade_mat_a, shade_mat_b


def pe_weight_reshape(W, K, L):
    W_rshp = np.array(np.split(np.array(np.split(W, K*L, axis=1)), K*L, axis=1))
    return W_rshp


def weight_normalize_2d_pe(W, b, K, L, shade_mat_a, shade_mat_b, hid_old, hid_new):
    if hid_old == '':
        W_rshp = pe_weight_reshape(W, K, L)
    else:
        pesudo_inv = np.matmul(np.linalg.inv(np.matmul(np.transpose(hid_new), hid_new)+1e-3*np.eye(hid_new.shape[1])), np.transpose(hid_new))
        W_rshp = pe_weight_reshape(np.matmul(np.matmul(pesudo_inv, hid_old), W), K, L)
    shade_mat_a1 = pe_weight_reshape(shade_mat_a, K, L)
    shade_mat_b1 = pe_weight_reshape(shade_mat_b, K, L)
    U = np.sum(W_rshp * shade_mat_a1 * shade_mat_b1, axis=(0, 1)) / (K * L)
    V = np.sum(W_rshp * shade_mat_a1 * (1-shade_mat_b1), axis=(0, 1)) / (K * L * (L-1))
    P = np.sum(W_rshp * (1-shade_mat_a1) * shade_mat_b1, axis=(0, 1)) / (K * (K-1) * L)
    Q = np.sum(W_rshp * (1-shade_mat_a1) * (1-shade_mat_b1), axis=(0, 1)) / (K * (K-1) * L * (L-1))
    W_nml = np.tile(U, (K*L, K*L)) * shade_mat_a * shade_mat_b + \
            np.tile(V, (K*L, K*L)) * shade_mat_a * (1-shade_mat_b) + \
            np.tile(P, (K*L, K*L)) * (1-shade_mat_a) * shade_mat_b + \
            np.tile(Q, (K*L, K*L)) * (1-shade_mat_a) * (1-shade_mat_b)
    b_nml = np.tile(np.mean(np.reshape(b, [K*L, -1]), axis=0), K*L)
    return W_nml, b_nml


def re_assign(weights, biases, K, L, hid_old='', hid_new=''):
    W = weights.eval()
    b = biases.eval()
    shade_mat_a, shade_mat_b = gen_2d_shade_mat(K, L, int(W.shape[0]/K/L), int(W.shape[1]/K/L))
    W_nml, b_nml = weight_normalize_2d_pe(W, b, K, L, shade_mat_a, shade_mat_b, hid_old, hid_new)
    return tf.to_float(tf.convert_to_tensor(W_nml)), tf.to_float(tf.convert_to_tensor(b_nml))


def submat2mat_2d(U, V, P, Q, b, K, L):
    shade_mat_a, shade_mat_b = gen_2d_shade_mat(K, L, int(U.shape[0]), int(U.shape[1]))
    W_nml = np.tile(U, (K * L, K * L)) * shade_mat_a * shade_mat_b + \
            np.tile(V, (K * L, K * L)) * shade_mat_a * (1 - shade_mat_b) + \
            np.tile(P, (K * L, K * L)) * (1 - shade_mat_a) * shade_mat_b + \
            np.tile(Q, (K * L, K * L)) * (1 - shade_mat_a) * (1 - shade_mat_b)
    b_nml = np.tile(b, K*L)
    return W_nml, b_nml
