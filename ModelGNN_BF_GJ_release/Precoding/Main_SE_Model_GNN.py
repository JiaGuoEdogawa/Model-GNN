"""
Code for "A Model-Based GNN for Learning Precoding" by Jia Guo, 2023-11-27
Thie is the main function for training the Model-GNN to maximize SE.
For any reproduce, further research or development, please kindly cite our TWC Journal paper:
`J. Guo and C. Yang, “A Model-Based GNN for Learning Precoding,” IEEE Trans. Wireless Commun., accepted, 2023.`
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICE'] = '-1'         # 使用CPU运行
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=Warning)
import tensorflow
if tensorflow.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.compat.v1.disable_eager_execution()
else:
    import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import sys
sys.path.append('../')
from common_funcs.data_process import get_random_block_from_data
from Precoding.Model.DRNN_model_BF import PENN_BF
from Precoding.Model.tf_train_beamforming import *
import time
import json

# ================================================ CONFIGS ========================================================== #

if len(sys.argv) > 1:
    configfile = './Configs/Config_' + sys.argv[1] + '.json'
    Configfile = open(configfile, 'r', encoding='utf8')
    config = json.load(Configfile)
    N_TRAIN = int(sys.argv[2]); config['N_TRAIN'] = N_TRAIN
    is_temp_model = True
    var_noise = float(sys.argv[3]); config['var_noise'] = var_noise
else:
    Configfile = open('./Configs/Config_BF.json', 'r', encoding='utf8')
    config = json.load(Configfile)
    N_TRAIN = config['N_TRAIN']
    is_temp_model = False
    var_noise = config['var_noise']

LEARNING_RATE = config['LEARNING_RATE']
if N_TRAIN <= 200:
    MAX_EPOCHS = 1000
else:
    MAX_EPOCHS = config['MAX_EPOCHS']
N_TEST = config['N_TEST']

n_marco   = config['n_marco']               # number of marco BSs
n_pico    = config['n_pico']                # number of pico BSs
nUE_marco = config['nUE_marco']             # number of UEs associate to marco BSs
nUE_pico  = config['nUE_pico']              # number of UEs associate to pico BSs
nTX_marco = config['nTX_marco']             # number of antennas of marco BSs
nTX_pico  = config['nTX_pico']              # number of antennas of pico BSs
nTX_UE    = config['nTX_UE']                # number of antennas of UEs
p_marco   = config['p_marco']               # power of marco BSs
p_pico    = config['p_pico']                # power of pico BSs

if nTX_marco <= 16:
    BATCH_SIZE = min(N_TRAIN-1, config['BATCH_SIZE'])
else:
    BATCH_SIZE = min(N_TRAIN - 1, 128)

p_BS = np.concatenate([np.ones(n_pico) * p_pico, np.ones(n_marco) * p_marco])
nUE_BS = np.int64(np.concatenate([np.ones(n_pico) * nUE_pico, np.ones(n_marco) * nUE_marco]))
nTX_BS = np.int64(np.concatenate([np.ones(n_pico) * nTX_pico, np.ones(n_marco) * nTX_marco]))
nTX_allUE = sum(nUE_BS); nTX_allBS = sum(nTX_BS)

model_location = model_folder_ini(is_temp_model)

X_train, Y_train, X_test, Y_test, H_TEST, pyrate = dataset_prepare_se(config)

# ==================================================== DNN =========================================================== #

x = tf.placeholder('float', [None, nTX_allUE, nTX_allBS, 2])
y = tf.placeholder('float', [None, nTX_allUE, nTX_allBS, 2])
is_trainBN = tf.placeholder(tf.bool)

n_hidden = [32, 32, 8, 1]
# n_hidden = [8, 8, 1]
y_pred = PENN_BF(x, n_hidden)

cost = tf_cal_loss_func_se(x, y_pred, config)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# ============================================== TRAIN INITIALIZE ==================================================== #
init, saver, sess, config_train = tf_initializer(gpu_fraction=0.4)
time_cost = 0; test_interval = 5; min_cost = 2020; break_flag = 1

# =============================================== TRAIN PROCESS ====================================================== #

with tf.Session(config=config_train) as sess:
# with tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})) as sess:  # 使用CPU运行
    sess.run(init)
    # saver.restore(sess, model_location)
    for epoch in range(MAX_EPOCHS):
        total_batch = int(N_TRAIN/BATCH_SIZE)
        for i in range(total_batch):
            start = time.perf_counter()
            batch_x, batch_y = get_random_block_from_data(BATCH_SIZE, X_train, Y_train)
            _, train_cost = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, is_trainBN: True})
            time_cost = time_cost + time.perf_counter() - start

        if epoch % test_interval == 0:
            Y_pred_test, test_cost = sess.run([y_pred, cost], feed_dict={x: X_test, y: Y_test, is_trainBN: False})
            ratio, nnrate = perf_eval_se(Y_pred_test, H_TEST, config, pyrate)
            if test_cost <= min_cost:
                min_cost = test_cost
                min_epoch = epoch
                saver.save(sess, model_location)

            print('Epoch:', '%04d' % (epoch + 1), 'Train cost', '%04f' % train_cost, 'Test cost', '%04f' % test_cost,
                  'Performance ratio', '%04f' % ratio)

        if len(sys.argv) > 1 and time_cost / 100 > break_flag:
            os.system('python Programbreak.py')
            break_flag = break_flag + 1

# =============================================== PERFORMANCE EVALUATE ============================================== #
    saver.restore(sess, model_location)
    start = time.perf_counter()
    Y_pred_test = sess.run(y_pred, feed_dict={x: X_test, y: Y_test, is_trainBN: False})
    test_time = time.perf_counter() - start
    ratio, nnrate = perf_eval_se(Y_pred_test, H_TEST, config, pyrate)