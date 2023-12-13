"""
Code for "A Model-Based GNN for Learning Precoding" by Jia Guo, 2023-11-27
This file includes functions of numerical algorithms for optimizing precoding.
For any reproduce, further research or development, please kindly cite our TWC Journal paper:
`J. Guo and C. Yang, “A Model-Based GNN for Learning Precoding,” IEEE Trans. Wireless Commun., accepted, 2023.`
"""

import numpy as np
import time

def ZF_precoding(H):
    """
    函数功能：给单个基站进行ZF预编码，得到预编码矩阵
    :param H: H是某一基站和用户之间的复信道，形状为用户数Ik×基站天线数Ntx（基站天线数大于等于用户数）
    :return: H_abs_ZF是经过迫零预编码后的信道向量，形状为（Ik,）。W是迫零预编码矩阵，形状为(Ntx, Ik)
    """

    HT = (H.conj()).T
    W = np.matmul(HT, np.linalg.inv(np.matmul(H, HT)))          # 求伪逆
    W = W / np.sqrt(np.sum(np.abs(W)**2))
    H_abs_ZF = np.diagonal(np.matmul(H, W))

    return W

def WMMSE_BF(H, nUE_BS, nTX_BS, nTX_UE, Pmax_BS, var_noise):
    """
    函数功能：利用 WMMSE 算法实现 IBC信道 下使多用户 和数据率最大 的 波束成形 算法
             所考虑的优化问题是：文献
             An Iteratively Weighted MMSE Approach to Distributed Sum-Utility
             Maximization for a MIMO Interfering Broadcast Channel
             中的问题(1)
             注意：本代码只能实现各用户单流传输的情况，即 N_k = 1，此外没有考虑和数据率加权的情况。

    :param H: 复信道矩阵，形状为 (Σ_k Σ_i (N_ik),  Σ_k(Mk))
    :param nUE_BS: 接入每个基站的用户数量，形状为(nBS), 其中第k个元素为接入第k个BS的用户数
    :param nTX_BS: 每个基站的天线数量，形状为(nBS), 其中第k个元素为第k个BS的天线数
    :param nTX_UE: 每个用户的天线数量，要求是定值，是一个标量
    :param Pmax_BS: 每个基站的最大发射功率，形状为(nBS), 其中第k个元素为第k个BS的最大发射功率
    :param var_noise: 用户端噪声方差，这是一个标量。

    :return: V：预编码策略向量，形状为 (1, Σ_k (I_k X M_k))
    """
    nBS = len(nUE_BS)
    nUE = sum(nUE_BS)

    V = np.zeros(sum(nUE_BS * nTX_BS), dtype=complex)

    iTX = 0
    for iBS in range(nBS):
        V[iTX: iTX + nUE_BS[iBS] * nTX_BS[iBS]] = np.sqrt(Pmax_BS[iBS] / (nTX_BS[iBS] * nUE_BS[iBS]))
        iTX = iTX + nUE_BS[iBS] * nTX_BS[iBS]

    U = np.zeros(nUE * nTX_UE, dtype=complex)
    W = np.zeros(nUE, dtype=complex)

    for iter in range(100):
        W_old = np.copy(W)
        for iBS in range(nBS):
            for iUE in range(nUE_BS[iBS]):
                iUE_idx = sum(nUE_BS[0:iBS]) + iUE
                interf = 0
                for jBS in range(nBS):
                    for jUE in range(nUE_BS[jBS]):
                        h = H[iUE_idx*nTX_UE: (iUE_idx+1)*nTX_UE, sum(nTX_BS[0:jBS]): sum(nTX_BS[0:jBS+1])]
                        v = V[sum(nUE_BS[0:jBS]*nTX_BS[0:jBS])+nTX_BS[jBS]*jUE :
                              sum(nUE_BS[0:jBS]*nTX_BS[0:jBS])+nTX_BS[jBS]*(jUE+1)]
                        v = np.reshape(v, [-1, 1])
                        interf = interf + np.matmul(np.matmul(np.matmul(h, v), (v.conj()).T), (h.conj()).T)
                interf = interf + var_noise * np.eye(nTX_UE)
                h = H[iUE_idx*nTX_UE: (iUE_idx+1)*nTX_UE, sum(nTX_BS[0:iBS]): sum(nTX_BS[0:iBS+1])]
                v = V[sum(nUE_BS[0:iBS] * nTX_BS[0:iBS]) + nTX_BS[iBS] * iUE:
                      sum(nUE_BS[0:iBS] * nTX_BS[0:iBS]) + nTX_BS[iBS] * (iUE + 1)]
                U[iUE_idx*nTX_UE: (iUE_idx+1)*nTX_UE] = np.matmul(np.matmul(np.linalg.inv(interf), h), v)

        for iBS in range(nBS):
            for iUE in range(nUE_BS[iBS]):
                iUE_idx = sum(nUE_BS[0:iBS]) + iUE
                h = H[iUE_idx * nTX_UE: (iUE_idx + 1) * nTX_UE, sum(nTX_BS[0:iBS]): sum(nTX_BS[0:iBS + 1])]
                v = V[sum(nUE_BS[0:iBS] * nTX_BS[0:iBS]) + nTX_BS[iBS] * iUE:
                      sum(nUE_BS[0:iBS] * nTX_BS[0:iBS]) + nTX_BS[iBS] * (iUE + 1)]
                u = U[iUE_idx*nTX_UE: (iUE_idx+1)*nTX_UE]
                W[iUE_idx] = 1 / (1 - np.matmul(np.matmul((u.conj()).T, h), v))

        for iBS in range(nBS):
            interf = 0
            interf_k = 0
            for jBS in range(nBS):
                for jUE in range(nUE_BS[jBS]):
                    jUE_idx = sum(nUE_BS[0:jBS]) + jUE
                    h = H[jUE_idx * nTX_UE: (jUE_idx + 1) * nTX_UE, sum(nTX_BS[0:iBS]): sum(nTX_BS[0:iBS + 1])]
                    h_H = (h.conj()).T
                    u = U[jUE_idx * nTX_UE: (jUE_idx + 1) * nTX_UE]
                    u = np.reshape(u, [-1, 1])
                    u_H = (u.conj()).T
                    w = W[jUE_idx]
                    interf = interf + np.matmul(np.matmul(np.matmul(h_H, u) * w, u_H), h)
                    if jBS == iBS:
                        interf_k = interf_k + np.matmul(np.matmul(np.matmul(h_H, u) * (w ** 2), u_H), h)
            interf = interf + get_mu(Pmax_BS[iBS], interf, interf_k) * np.eye(nTX_BS[iBS])
            for iUE in range(nUE_BS[iBS]):
                iUE_idx = sum(nUE_BS[0:iBS]) + iUE
                h = H[iUE_idx * nTX_UE: (iUE_idx + 1) * nTX_UE, sum(nTX_BS[0:iBS]): sum(nTX_BS[0:iBS + 1])]
                h_H = (h.conj()).T
                u = U[iUE_idx*nTX_UE: (iUE_idx+1)*nTX_UE]
                w = W[iUE_idx]
                V[sum(nUE_BS[0:iBS] * nTX_BS[0:iBS]) + nTX_BS[iBS] * iUE:
                  sum(nUE_BS[0:iBS] * nTX_BS[0:iBS]) + nTX_BS[iBS] * (iUE + 1)] = \
                    np.matmul(np.matmul(np.linalg.inv(interf), h_H), u) * w
        if abs(np.sum(np.log(W_old)) - np.sum(np.log(W))) < 1e-4:
            break

    return V


def WMMSE_1BS1UETX(H, Pmax, var_noise):
    nUE = H.shape[0]
    nTX = H.shape[1]

    V = np.ones([nTX, nUE], dtype=complex) * np.sqrt(Pmax / (nTX * nUE))
    W = np.zeros(nUE, dtype=complex)

    for iter in range(100):
        W_old = np.copy(W)

        H_tmp = np.expand_dims(H, axis=1)
        V_tmp = np.expand_dims(V, axis=0)

        I = np.sum(np.power(np.abs(np.matmul(H_tmp, V_tmp)), 2), axis=2)
        V_tmp = np.expand_dims(np.transpose(V), axis=2)
        U = np.matmul(H_tmp, V_tmp)[:, 0, 0] / (I[:, 0] + var_noise)

        U_tmp = np.expand_dims(np.expand_dims(U, axis=1), axis=1)
        W = 1 / (1 - np.matmul(np.matmul(U_tmp.conj(), H_tmp), V_tmp))

        UHH = np.matmul(U_tmp.conj(), H_tmp)
        W_tmp = np.reshape(W, [-1, 1, 1])
        I = np.sum(np.matmul(np.matmul(np.transpose(UHH.conj(), [0, 2, 1]), W_tmp), UHH), axis=0)
        Ik = np.sum(np.matmul(np.matmul(np.transpose(UHH.conj(), [0, 2, 1]), W_tmp**2), UHH), axis=0)
        I = I + get_mu(Pmax, I, Ik) * np.eye(nTX)
        V = np.matmul(np.matmul(np.expand_dims(np.linalg.inv(I),axis=0), np.transpose(UHH.conj(), [0, 2, 1])), W_tmp)
        V = np.transpose(V[:, :, 0])

        if iter>=1 and abs(np.sum(np.log(W_old)) - np.sum(np.log(W))) < 1e-4:
            break

    return V


def cal_sum_rate_BF_UE1TX(H, V, var_noise):
    """
    函数功能：计算给定波束成形V和信道矩阵H下的数据率。注意：此函数只能用于计算各用户单天线场景下的数据率。

    :param H: 复数 信道矩阵，形状为（样本数、用户数、基站天线数）
    :param V: 复数 预编码矩阵，形状为（样本数、基站天线数、用户数）
    :param var_noise: 噪声功率

    :return: rate: 各样本和数据率，形状为（样本数）
    """
    if len(np.shape(H)) < 3:
        H = np.expand_dims(H, axis=0)
        V = np.expand_dims(V, axis=0)

    nUE = np.shape(H)[1]
    G = np.power(np.abs(np.matmul(H, V)), 2)
    A = np.reshape(np.eye(nUE), [1, nUE, nUE])
    rate = np.log2(1 + np.sum(G*A, axis=2) / (np.sum(G*(1-A), axis=2) + var_noise))
    sumrate = np.sum(rate, axis=1)

    return sumrate, rate


def gen_sample(nUE_BS, nTX_BS, nTX_UE, Pmax_BS, var_noise, chl_model='Ray', KdB=0, is_gen_label=True):
    """
    函数功能：生成一个样本

    :param nUE_BS: 接入每个基站的用户数量，形状为(nBS), 其中第k个元素为接入第k个BS的用户数
    :param nTX_BS: 每个基站的天线数量，形状为(nBS), 其中第k个元素为第k个BS的天线数
    :param nTX_UE: 每个用户的天线数量，要求是定值，是一个标量
    :param Pmax_BS: 每个基站的最大发射功率，形状为(nBS), 其中第k个元素为第k个BS的最大发射功率
    :param var_noise: 用户端噪声方差，这是一个标量。
    :param chl_model: 信道模型，‘Ray’为瑞利信道，‘Ric’为莱斯（Rician）信道。
    :param KdB: K因子(dB)，在《MIMO-OFDM Matlab实现》18页式(1.43)中有定义。如果chl_model='Ray'，该参数无用。
    :return:
    """

    nTX_allBS = sum(nTX_BS)
    nTX_allUE = sum(nUE_BS) * nTX_UE
    Ncell = len(nUE_BS)

    # scalability
    # if nTX_allUE % 2 == 0:
    #     Hmat = 1 / np.sqrt(2) * (np.random.randn(1, 2, nTX_allBS) + 1j * np.random.randn(1, 2, nTX_allBS))
    #     Hmat = Hmat * np.ones([nTX_allUE//2, 1, 1])
    #     Hmat = np.reshape(Hmat, [nTX_allUE, nTX_allBS])
    # else:
    #     Hmat = 1 / np.sqrt(2) * (np.random.randn(nTX_allUE, nTX_allBS) + 1j * np.random.randn(nTX_allUE, nTX_allBS))
    Hmat = 1 / np.sqrt(2) * (np.random.randn(nTX_allUE, nTX_allBS) + 1j * np.random.randn(nTX_allUE, nTX_allBS))
    # Hmat = Hmat * gen_adj_mat(Ncell, nTX_allBS//Ncell, nTX_allUE//Ncell)

    if chl_model == 'Ric':
        K = 10.0**(KdB / 10.0)
        Hmat = np.sqrt(K / (K + 1)) + np.sqrt(1 / (K + 1)) * Hmat

    if (nTX_UE == 1) and (len(nUE_BS) == 1): # 单基站、用户单天线的场景
        if is_gen_label is True:
            start = time.perf_counter()
            Vmat = WMMSE_1BS1UETX(Hmat, Pmax_BS[0], var_noise)
            cost_time = time.perf_counter() - start
            rate = cal_sum_rate_BF_UE1TX(Hmat, Vmat, var_noise)
        else:
            Vmat = np.zeros(Hmat.transpose().shape)
            rate = np.zeros(1)

    elif (nTX_UE == 1) and (len(nUE_BS) > 1):   # 多基站、用户单天线的场景
        Vmat = np.zeros([nTX_allBS, nTX_allUE], dtype=complex)
        if is_gen_label is True:
            Vopt = WMMSE_BF(Hmat, nUE_BS, nTX_BS, nTX_UE, Pmax_BS, var_noise)
            for iBS in range(len(nUE_BS)):
                Vtmp = np.transpose(np.reshape(Vopt[sum(nUE_BS[0:iBS]*nTX_BS[0:iBS]):sum(nUE_BS[0:iBS+1]*nTX_BS[0:iBS+1])],
                                               [nUE_BS[iBS], nTX_BS[iBS]]))
                Vmat[sum(nTX_BS[0:iBS]):sum(nTX_BS[0:iBS+1]), sum(nUE_BS[0:iBS]):sum(nUE_BS[0:iBS+1])] = Vtmp
        rate = cal_sum_rate_BF_UE1TX(Hmat, Vmat, var_noise)

    else:                                       # 用户多天线场景
        Vopt = WMMSE_BF(Hmat, nUE_BS, nTX_BS, nTX_UE, Pmax_BS, var_noise)
        Vmat = Vopt
        rate = np.nan

    return Hmat, Vmat, rate


def get_mu(Pmax, interf, interf_k):
    Lambda, D = np.linalg.eig(interf)
    D_H = (D.conj()).T
    Phi = np.matmul(np.matmul(D_H, interf_k), D)
    Phi = np.real(np.diag(Phi))
    Lambda = np.real(Lambda)
    mu_upp = 1.0
    mu_low = 0.0
    mu = (mu_upp + mu_low) / 2
    P = np.sum(Phi / (Lambda + mu) ** 2)
    while P > Pmax:
        mu_low = mu_upp
        mu_upp = mu_upp * 2
        P = np.sum(Phi / (Lambda + mu_upp) ** 2)
    while abs(mu_upp - mu_low) > 1e-4:
        mu = (mu_upp + mu_low) / 2
        P = np.sum(Phi / (Lambda + mu) ** 2)
        if P < Pmax:
            mu_upp = mu
        else:
            mu_low = mu
    return mu


def cal_ratio(pyrate_mat, nnrate_mat, n_spl_oneuser, method='div'):
    pyrate_mat = np.reshape(pyrate_mat, [-1, n_spl_oneuser])
    nnrate_mat = np.reshape(nnrate_mat, [-1, n_spl_oneuser])
    pyrate = np.mean(pyrate_mat, axis=1)
    nnrate = np.mean(nnrate_mat, axis=1)
    if method == 'div':
        ratio = nnrate / pyrate * 100
    elif method == 'subtract':
        ratio = (pyrate - nnrate) * 100
    else:
        print('wrong method!')
    return ratio


def cal_corr(H, V):
    '''
    Function: Calculate the correlation between channel matrix H and precoding matrix V
    :param H: Complex channel matrix, with shape (N_spl, K, Ntx, 2)
    :param V: Complex precoding matrix, with shape (N_spl, K, Ntx, 2)
    :return: A correlation factor, which is a scalar
    '''
    X = H; Y = V
    X_Re = X[..., 0]; X_Im = X[..., 1]
    Y_Re = Y[..., 0]; Y_Im = Y[..., 1]
    XY_Re = X_Re * Y_Re - X_Im * Y_Im
    XY_Im = X_Re * Y_Im + X_Im * Y_Re
    XY = np.concatenate((np.expand_dims(XY_Re, axis=-1), np.expand_dims(XY_Im, axis=-1)), axis=-1)
    temp = np.sum(XY, axis=2, keepdims=True) / (np.sqrt(np.sum(np.power(X, 2), axis=(2, 3), keepdims=True)) *
                                                np.sqrt(np.sum(np.power(Y, 2), axis=(2, 3), keepdims=True)))
    temp1 = np.sqrt(np.power(temp[..., 0], 2) + np.power(temp[..., 1], 2))
    corr = np.mean(temp1)

    return temp1[..., 0], corr


def cal_approx_pseodo_inv(H, var_noise=1):
    # H 用户数 X 天线数
    K = H.shape[1]
    H = H[:, :, :, 0] + 1j * H[:, :, :, 1]
    hHh = np.matmul(H, np.transpose(H, [0, 2, 1]).conj()) + var_noise * np.reshape(np.eye(K), [1, K, K])
    # approx_inv = np.reshape(np.eye(K), [1, K, K]) * 1 / np.real(hHh) + 1j * 0.0
    approx_inv = np.linalg.inv(hHh)
    approx_pinv = np.matmul(np.transpose(H, [0, 2, 1]).conj(), approx_inv)
    pinv_real = np.real(approx_pinv); pinv_imag = np.imag(approx_pinv)
    pinv = np.concatenate((np.expand_dims(pinv_real, axis=3), np.expand_dims(pinv_imag, axis=3)), axis=3)
    pinv = pinv / np.sqrt(np.sum(np.power(pinv, 2), axis=(1, 2, 3), keepdims=True))
    pinv = np.transpose(pinv, [0, 2, 1, 3])

    return pinv


def power_opt(K, Pmax, Pc, R0, H, rho, var_noise):
    """
    函数功能：实现K个收发机对、能效最大化的功率控制，不考虑干扰。优化问题如下：

                log2(1 + |h_k|^2 * p_k / σ^2)
           max  ——————————————————————————————
                    Σ_(k=1)^K ρ * p_k + Pc

           s.t. log2(1 + |h_k|^2 * p_k / σ^2) ≥ r0

                0 ≤ p_k ≤ Pmax, k = 1, ..., K

    :param K: 基站-用户对的数量
    :param Pmax: 各基站的最大发射功率，形状为（K）
    :param Pc: 各基站的电路功耗，形状为（K）
    :param R0: 各用户的最小数据率要求（QoS约束）
    :param H: 各基站-用户对之间的信道，形状为（K）的复数向量
    :param rho: 基站功放的效率
    :param var_noise: 噪声功率
    :return:
    """

    P = Pmax
    H = np.abs(H)
    ee_old, _ = cal_ee(H, P, Pc, rho, var_noise)
    for iter in range(100):
        R = np.log2(1.0 + H * H * P / var_noise)
        P = (np.maximum(np.power(2, R0), np.sum(rho * P + Pc) * H * H / var_noise / (rho * np.sum(R) * np.log(2))) - 1) *\
            var_noise / ((H * H) + 1e-6)
        P = np.maximum(np.minimum(P, Pmax), 0)
        ee_new, _ = cal_ee(H, P, Pc, rho, var_noise)
        if abs(ee_new - ee_old) <= 1e-3:
            # print('converge at the ' + str(iter + 1) + ' th iteration!')
            break
        ee_old = ee_new

    return P



def gen_adj_mat(Ncell, N, K):
    """
    函数功能：生成多小区波束成形的接入关系矩阵。这只能支持每个小区天线数、用户数相同的场景。
    :param Ncell: 小区数
    :param N: 各小区天线数：注意：不是总天线数！！！
    :param K: 各小区用户数：注意：不是总用户数！！！
    :return:
    """
    temp = np.reshape(np.eye(Ncell), [Ncell, Ncell, 1]) * np.ones([1, 1, N])
    Adj1 = np.reshape(temp, [Ncell, Ncell * N])
    temp = np.reshape(Adj1, [Ncell, 1, Ncell * N]) * np.ones([1, K, 1])
    Adj = np.reshape(temp, [Ncell * K, Ncell * N])

    return Adj




if __name__ == '__main__':
    # nUE_BS = np.array([30])
    # nTX_BS = np.array([64])
    # nTX_UE = 1
    # Pmax_BS = np.array([10])
    # var_noise = 1
    # H = np.random.randn(sum(nUE_BS)*nTX_UE, sum(nTX_BS)) + 1j * np.random.randn(sum(nUE_BS)*nTX_UE, sum(nTX_BS))
    # V = WMMSE_BF(H, nUE_BS, nTX_BS, nTX_UE, Pmax_BS, var_noise)

    K = 25; Ntx = 32; Pmax = 100000; var_noise = 0.1; rau = 1/0.311; rmin = 0.0; P0 = 43.3; Pc = 17.6
    H = 1 / np.sqrt(2) * (np.random.randn(K, Ntx) + 1j * np.random.randn(K, Ntx))
    V = EE_max_Precoding(H, Pmax, var_noise, rau, rmin, P0, Pc)
    # H_UEBS, _, _, Popt_BS, sum_rate = gen_channel(nUE_BS, nTX_BS, Pmax_BS, var_noise)