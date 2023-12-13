"""
Code for "A Model-Based GNN for Learning Precoding" by Jia Guo, 2023-11-27
This file includes functions of the permutation equivariant DNNs (PENNs), which are also Vanilla-GNNs.
For any reproduce, further research or development, please kindly cite our TWC Journal paper:
`J. Guo and C. Yang, “A Model-Based GNN for Learning Precoding,” IEEE Trans. Wireless Commun., accepted, 2023.`
"""

import tensorflow
if tensorflow.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf


def ini_weights(in_shape, out_shape, name, stddev=0.1, is_train=True, initialize='normal'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
            if initialize == 'normal':
                weight_mat = tf.get_variable(name='weight', shape=[in_shape, out_shape],
                                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=stddev),
                                             trainable=is_train)
            else:
                weight_mat = tf.get_variable(name='weight', shape=[in_shape, out_shape],
                                             initializer=tf.zeros_initializer(), trainable=is_train)
    return weight_mat


def ini_weights_highdim(shape, name, stddev=0.1, is_train=True, initlize='normal'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('weights', reuse=tf.AUTO_REUSE):
            if initlize == 'normal':
                weight_mat = tf.get_variable(name='weight', shape=shape,
                                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=stddev),
                                             trainable=is_train)
            else:
                weight_mat = tf.get_variable(name='weight', shape=shape,
                                             initializer=tf.zeros_initializer(), trainable=is_train)
    return weight_mat


def ini_bias(out_shape, name, is_train=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
            bias_vec = tf.get_variable(name='bias', shape=[out_shape],
                                       initializer=tf.zeros_initializer(), trainable=is_train)
    return bias_vec


def ini_bias_highdim(shape, name, is_train=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('biases', reuse=tf.AUTO_REUSE):
            bias_vec = tf.get_variable(name='bias', shape=shape,
                                       initializer=tf.zeros_initializer(), trainable=is_train)
    return bias_vec


def add_2d_pe_layer(input, inshape, outshape, name, stddev=0.1, transfer_function=tf.nn.leaky_relu, is_BN=False,
                    is_transfer=True, k_factor = 1.0, aggr_func=tf.reduce_mean, is_trainBN=False, initialize='normal'):
    """
    :param input: 隐藏层输入，形状：(None, object数1，object数2，各object特征数)
    :return: output: 隐藏层输出，形状：(None, object数1，object数2，各object表示数)
    """
    weight = dict()
    U = ini_weights(inshape, outshape, name + '_U', stddev, initialize=initialize)
    V = ini_weights(inshape, outshape, name + '_V', stddev, initialize=initialize)
    P = ini_weights(inshape, outshape, name + '_P', stddev, initialize=initialize)
    Q = ini_weights(inshape, outshape, name + '_Q', stddev, initialize=initialize) * 0.0
    b = ini_bias(outshape, name + '_bias')
    weight['U'] = U; weight['V'] = V; weight['P'] = P; weight['Q'] = Q; bias = b
    input_sum_dim1  = aggr_func(input, axis=1,      keepdims=True)
    input_sum_dim2  = aggr_func(input, axis=2,      keepdims=True)
    input_sum_dim12 = aggr_func(input, axis=[1, 2], keepdims=True)
    output = tf.matmul(input,      U-V-P+Q) + \
             tf.matmul(input_sum_dim1, V-Q) + \
             tf.matmul(input_sum_dim2, P-Q) + \
             tf.matmul(input_sum_dim12,  Q) + b
    if is_BN is True:
        output = tf.layers.batch_normalization(output, training=is_trainBN, name=name + '_BN', reuse=tf.AUTO_REUSE)
    if is_transfer is True:
        output = transfer_function(output)

    return output, weight, bias


def penn_2d(input, n_obj1, n_obj2, layernum, output_activation=tf.nn.sigmoid, is_BN=False, is_trainBN=False, name='',
            is_lst_transfer=True):
    hidden = dict()
    # input_exdim = tf.reshape(input, [-1, n_obj1, n_obj2, int(input.shape[1]//n_obj1//n_obj2)])
    input_exdim = input
    if len(layernum) >= 2:
        for i in range(len(layernum)):
            if i == 0:
                hidden[str(i)], _, _ = add_2d_pe_layer(input_exdim, int(input_exdim.shape[3]), layernum[i],
                                                       name + 'layer' + str(i + 1),
                                                       is_BN=is_BN, is_trainBN=is_trainBN, transfer_function=tf.nn.relu)
            elif i != len(layernum)-1:
                hidden[str(i)], _, _ = add_2d_pe_layer(hidden[str(i - 1)], layernum[i-1], layernum[i],
                                                       name + 'layer' + str(i + 1),
                                                       is_BN=is_BN, is_trainBN=is_trainBN, transfer_function=tf.nn.relu)
            else:
                output_exdim, _, _ = add_2d_pe_layer(hidden[str(i - 1)], layernum[i-1], layernum[i],
                                                     name + 'layer' + str(i + 1),
                                                     transfer_function=output_activation, is_BN=is_BN,
                                                     is_trainBN=is_trainBN, is_transfer=is_lst_transfer)
    else:
        output_exdim, _, _ = add_2d_pe_layer(input_exdim, int(input_exdim.shape[3]), layernum[0], name + 'layer' + str(1),
                                       transfer_function=output_activation, is_BN=is_BN, is_trainBN=is_trainBN,
                                       is_transfer = is_lst_transfer)
    # output = tf.reshape(output_exdim, [-1, layernum[-1] * n_obj1 * n_obj2])
    output = output_exdim
    return output
