"""
Code for "A Model-Based GNN for Learning Precoding" by Jia Guo, 2023-11-27
Thie is the code for generating samples for training the GNNs to maximize SE.
For any reproduce, further research or development, please kindly cite our TWC Journal paper:
`J. Guo and C. Yang, “A Model-Based GNN for Learning Precoding,” IEEE Trans. Wireless Commun., accepted, 2023.`
"""

import numpy as np
from Precoding.Model.beamforming_wmmse import gen_sample
import json
import scipy.io as sio


def gen_dataset(config, var_noise, dataset='train'):
    N_TRAIN = config['MAX_N_TRAIN']
    N_TEST = config['MAX_N_TEST']

    n_marco = config['n_marco']              # number of marco BSs
    n_pico = config['n_pico']                # number of pico BSs
    nUE_marco = config['nUE_marco']          # number of UEs associate to marco BSs
    nUE_pico = config['nUE_pico']            # number of UEs associate to pico BSs
    nTX_marco = config['nTX_marco']          # number of antennas of marco BSs
    nTX_pico = config['nTX_pico']            # number of antennas of pico BSs
    nTX_UE = config['nTX_UE']                # number of antennas of UEs
    p_marco = config['p_marco']              # power of marco BSs
    p_pico = config['p_pico']                # power of pico BSs

    p_BS = np.concatenate([np.ones(n_pico) * p_pico, np.ones(n_marco) * p_marco])
    nUE_BS = np.int64(np.concatenate([np.ones(n_pico) * nUE_pico, np.ones(n_marco) * nUE_marco]))
    nTX_BS = np.int64(np.concatenate([np.ones(n_pico) * nTX_pico, np.ones(n_marco) * nTX_marco]))

    file_suffix = str(n_marco) + 'MBS_' + str(nTX_marco) + 'MTX_' + str(nUE_marco) + 'MUE_' + \
                  str(n_pico) + 'PBS_' + str(nTX_pico) + 'PTX_' + str(nUE_pico) + 'PUE' + str(var_noise) + 'N'

    if dataset == 'train':
        H = np.zeros([N_TRAIN, sum(nUE_BS)*nTX_UE, sum(nTX_BS)], dtype=complex)
        V = np.zeros([N_TRAIN, sum(nTX_BS), sum(nUE_BS)*nTX_UE], dtype=complex)
        SR = np.zeros(N_TRAIN)

        trainfilename = 'Dataset/Train_' + file_suffix + '.mat'
        for i in range(N_TRAIN):
            Hmat, Vmat, rate = gen_sample(nUE_BS, nTX_BS, nTX_UE, p_BS, var_noise, is_gen_label=False)
            H[i, :, :] = Hmat
            V[i, :, :] = Vmat
            SR[i] = rate[0]
            if (i + 1) % 100 == 0:
                print('The ' + str(i+1) + ' th training sample has been generated...')

        sio.savemat(trainfilename, {'Xtrain': H, 'Ytrain': V, 'SR': SR})

    else:
        H = np.zeros([N_TEST, sum(nUE_BS)*nTX_UE, sum(nTX_BS)], dtype=complex)
        V = np.zeros([N_TEST, sum(nTX_BS), sum(nUE_BS)*nTX_UE], dtype=complex)
        SR = np.zeros(N_TEST)

        testfilename = 'Dataset/Test_' + file_suffix + '.mat'
        for i in range(N_TEST):
            Hmat, Vmat, rate = gen_sample(nUE_BS, nTX_BS, nTX_UE, p_BS, var_noise)
            H[i, :, :] = Hmat
            V[i, :, :] = Vmat
            SR[i] = rate[0]
            if (i + 1) % 1 == 0:
                print('The ' + str(i+1) + ' th testing sample has been generated...')

        sio.savemat(testfilename, {'Xtest': H, 'Ytest': V, 'SR': SR})


if __name__ == '__main__':
    Configfile = open('./Configs/Config_MBF3.json', 'r', encoding='utf8')
    config = json.load(Configfile)
    gen_dataset(config, 0.01)