"""
Code for "A Model-Based GNN for Learning Precoding" by Jia Guo, 2023-11-27
This file includes functions of optimizer, loss functions, data processing, etc.
For any reproduce, further research or development, please kindly cite our TWC Journal paper:
`J. Guo and C. Yang, “A Model-Based GNN for Learning Precoding,” IEEE Trans. Wireless Commun., accepted, 2023.`
"""

import tensorflow
if tensorflow.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
else:
    import tensorflow as tf
import os
import numpy as np
import scipy.io as sio


def tf_cal_loss_func_ee(x, y_pred, lamda, config):
    var_noise    = config['var_noise']
    rau          = config['rau']
    C            = config['C']
    r_min        = config['rmin']
    scale_factor = config['SF']

    sum_rate, data_rate = tf_cal_datarate(x, y_pred, var_noise / scale_factor)
    sum_power = rau * tf.reduce_sum(tf.pow(y_pred, 2), reduction_indices=[1, 2, 3]) + C / scale_factor
    cost = -1 * tf.reduce_sum(sum_rate / sum_power) + tf.reduce_sum(
        0.1 * tf.nn.sigmoid(lamda) * tf.nn.leaky_relu(r_min + 0.2 - data_rate / tf.math.log(2.0)))

    return cost


def tf_cal_loss_func_se(x, y_pred, config):
    var_noise    = config['var_noise']
    sum_rate, _  = tf_cal_datarate(x, y_pred, var_noise)
    cost         = -1 * tf.reduce_sum(sum_rate)

    return cost


def tf_cal_datarate(x, y_pred, var_noise):
    K = int(x.shape[1])
    V_ReIm = tf.transpose(y_pred, [0, 2, 1, 3])
    V_Re = V_ReIm[:, :, :, 0]; V_Im = V_ReIm[:, :, :, 1]
    H_Re = x[:, :, :, 0];      H_Im = x[:, :, :, 1]
    G_Re = tf.matmul(H_Re, V_Re) - tf.matmul(H_Im, V_Im)
    G_Im = tf.matmul(H_Re, V_Im) + tf.matmul(H_Im, V_Re)
    G2 = tf.pow(G_Re, 2) + tf.pow(G_Im, 2)
    A = tf.reshape(tf.eye(K), [1, K, K])
    data_rate = tf.math.log(1 + tf.reduce_sum(G2 * A, axis=2) / (tf.reduce_sum(G2 * (1 - A), axis=2) + var_noise))
    sum_rate = tf.reduce_sum(data_rate, axis=1)

    return sum_rate, data_rate


def tf_optimizer(cost, var_keyw, is_var, LR, optimizer=tf.train.AdamOptimizer):
    opt = optimizer(learning_rate=LR)
    if is_var is True:
        train_var_list = [var for var in tf.trainable_variables() if var_keyw in var.name]
    else:
        train_var_list = [var for var in tf.trainable_variables() if var_keyw not in var.name]
    grads, vars = zip(*opt.compute_gradients(cost, var_list=train_var_list))

    return opt, grads, vars


def tf_initializer(gpu_fraction=0.4):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # saver = 0
    config = tf.ConfigProto()
    # 配置GPU内存分配方式，按需增长
    config.gpu_options.allow_growth = True
    # 配置可使用的显存比例
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
    # 在创建session的时候把config作为参数传进去
    sess = tf.InteractiveSession(config=config)

    return init, saver, sess, config


def model_folder_ini(is_temp_model):
    if is_temp_model is True:
        model_folder = './DNNmodel/FCmodel' + '_' + '%0.3f' % np.random.rand()
    else:
        model_folder = './DNNmodel'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_location = model_folder + '/model_demo.ckpt'

    return model_location


def dataset_preprocess(filename, x_keyw, y_keyw, n_spl):
    dfile = sio.loadmat(filename)
    X_cplx = dfile[x_keyw]; Y_cplx = dfile[y_keyw]
    X = np.concatenate((np.expand_dims(np.real(X_cplx), axis=-1), np.expand_dims(np.imag(X_cplx), axis=-1)), axis=-1)
    Y = np.concatenate((np.expand_dims(np.real(Y_cplx), axis=-1), np.expand_dims(np.imag(Y_cplx), axis=-1)), axis=-1)
    Y = np.transpose(Y, [0, 2, 1, 3])
    X_data = X[0: n_spl]; Y_data = Y[0: n_spl]

    return X_data, Y_data


def dataset_prepare_se(config):
    from Precoding.Model.beamforming_wmmse import cal_sum_rate_BF_UE1TX
    from Precoding.Gen_dataset import gen_dataset

    N_TRAIN      = config['N_TRAIN']
    N_TEST       = config['N_TEST']
    n_marco      = config['n_marco']
    n_pico       = config['n_pico']
    nUE_marco    = config['nUE_marco']
    nUE_pico     = config['nUE_pico']
    nTX_marco    = config['nTX_marco']
    nTX_pico     = config['nTX_pico']
    var_noise    = config['var_noise']

    file_suffix = str(n_marco) + 'MBS_' + str(nTX_marco) + 'MTX_' + str(nUE_marco) + 'MUE_' + \
                  str(n_pico) + 'PBS_' + str(nTX_pico) + 'PTX_' + str(nUE_pico) + 'PUE' + str(var_noise) + 'N'

    filename = 'Dataset/Train_' + file_suffix + '.mat'
    if not os.path.exists(filename):
        gen_dataset(config, var_noise, dataset='train')
    X_train, Y_train = dataset_preprocess(filename, 'Xtrain', 'Ytrain', N_TRAIN)

    filename = 'Dataset/Test_' + file_suffix + '.mat'
    if not os.path.exists(filename):
        gen_dataset(config, var_noise, dataset='test')
    X_test, Y_test = dataset_preprocess(filename, 'Xtest', 'Ytest', N_TEST)

    H_TEST = X_test[:, :, :, 0] + 1j * X_test[:, :, :, 1]
    V_TEST = np.transpose(Y_test[:, :, :, 0] + 1j * Y_test[:, :, :, 1], [0, 2, 1])
    pyrate, _ = cal_sum_rate_BF_UE1TX(H_TEST, V_TEST, var_noise)
    pyrate = np.mean(pyrate)

    return X_train, Y_train, X_test, Y_test, H_TEST, pyrate


def perf_eval_se(Y_pred, H, config, pyrate):
    from Precoding.Model.beamforming_wmmse import cal_sum_rate_BF_UE1TX
    var_noise =    config['var_noise']
    V_pred_test = np.transpose(Y_pred[:, :, :, 0] + 1j * Y_pred[:, :, :, 1], [0, 2, 1])
    nnrate, _ = cal_sum_rate_BF_UE1TX(H, V_pred_test, var_noise)
    nnrate = np.mean(nnrate)
    ratio = nnrate / pyrate * 100

    return ratio, nnrate


def gen_prog_bar(epoch, max_epoch, max_block=20):
    n_black = int(epoch / max_epoch * max_block)
    n_white = max_block - n_black
    list = '|' + '■' * (n_black) + '>' + ' ' * n_white + '|'

    return list