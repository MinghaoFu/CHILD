import os
import glob
import tqdm
import torch
import scipy
import random
import ipdb as pdb
import numpy as np
from torch import nn
from torch.nn import init
from collections import deque
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import ortho_group
from sklearn.preprocessing import scale
from IDOL.tools.utils import create_sparse_transitions, controlable_sparse_transitions
from itertools import product, permutations
from scipy.linalg import block_diag
import argparse

VALIDATION_RATIO = 0.2
root_dir = '../datasets/dataset3'
standard_scaler = preprocessing.StandardScaler()

def random_permutation_matrix(n):
    P = np.eye(n)
    np.random.shuffle(P)
    return P

def block_diagonal_permutation_matrix(z_dim_list=[1, 2, 4]):
    'generate transition, simple version'
    blocks = [random_permutation_matrix(block_size) for block_size in z_dim_list]
    return block_diag(*blocks)

def leaky_ReLU_1d(d, negSlope):
    if d > 0:
        return d
    else:
        return d * negSlope

leaky1d = np.vectorize(leaky_ReLU_1d)

# def leaky_ReLU(D, negSlope):
#     assert negSlope > 0
#     return leaky1d(D, negSlope)

def leaky_ReLU(D, negSlope):
    return D

def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

def sigmoidAct(x):
    return 1. / (1 + np.exp(-1 * x))

def generateUniformMat(Ncomp, condT):
    """
    generate a random matrix by sampling each element uniformly at random
    check condition number versus a condition threshold
    """
    A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
    for i in range(Ncomp):
        A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    while np.linalg.cond(A) > condT:
        # generate a new A matrix!
        A = np.random.uniform(0, 2, (Ncomp, Ncomp)) - 1
        for i in range(Ncomp):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())

    return A


def stationary_z_link(obs_dim = 4, hist = 0, inst = False, name = 'data', matrix=None, seed=None, z_dim_list=[1, 2, 4]):
    lags = 1
    Nlayer = 2
    length = 6
    condList = []
    negSlope = 0.2
    latent_size = sum(z_dim_list)
    transitions = []
    noise_scale = 0.1
    batch_size = 100000
    Niter4condThresh = 1e4
    
    n_layer = len(z_dim_list)   
    
    if seed is not None:
        np.random.seed(seed)

    path = os.path.join(root_dir, name)
    os.makedirs(path, exist_ok=True)

    for i in range(int(Niter4condThresh)):
        # A = np.random.uniform(0,1, (Ncomp, Ncomp))
        A = np.random.uniform(1, 2, (latent_size, latent_size))  # - 1
        for i in range(latent_size):
            A[:, i] /= np.sqrt((A[:, i] ** 2).sum())
        condList.append(np.linalg.cond(A))

    condThresh = np.percentile(condList, 25)  # only accept those below 25% percentile
    for l in range(lags):

        if hist == 0:
            B = generateUniformMat(latent_size, condThresh)
            transitions.append(B)
        elif hist == 1:
            assert(latent_size == 3)
            transitions.append(np.array([[0.4, 0.6, 0],
                                         [0, 1, 0],
                                         [0, 0, 1]], dtype=np.float32))
        elif hist == 2:
            # transition_matrix = np.eye(latent_size, dtype=np.float32)
            # # transition_matrix = transition_matrix[0] * 0.1
            # transitions.append(transition_matrix)
            if matrix is None:
                transitions.append(block_diagonal_permutation_matrix(z_dim_list))
            else:
                transitions.append(matrix)
        elif hist == 3:
            assert(latent_size == 5)
            trans_mat = np.eye(5, dtype=np.float32)
            trans_mat[1][3]+=1.0
            trans_mat[2][4]+=1.0
            transitions.append(trans_mat)
        
    transitions.reverse()

    all_mixingList = []
    for _ in range(n_layer):
        mixingList = []
        for l in range(Nlayer - 1):
            # generate causal matrix first:
            # A = ortho_group.rvs(latent_size)  # generateUniformMat(Ncomp, condThresh)
            A = ortho_group.rvs(obs_dim)  # generateUniformMat(Ncomp, condThresh)
            mixingList.append(A)
        all_mixingList.append(mixingList)

    y_l = np.random.normal(0, 1, (batch_size, lags, latent_size))
    y_l = (y_l - np.mean(y_l, axis=0, keepdims=True)) / np.std(y_l, axis=0 ,keepdims=True)

    yt = []; xt = []
    for i in range(lags):
        yt.append(y_l[:,i,:])
    # mixedDat = np.copy(y_l)
    mixedDat = np.copy(y_l[:,:,-z_dim_list[-1]:])
    # print(mixedDat.shape)
    # exit()
        
    # also generate with all mixingList, then concat them
    all_mixedDat = []
    for i in range(1):        
        mixedDat = np.copy(y_l[:,:,-z_dim_list[-1]:])
        for l in range(Nlayer - 1):
            mixedDat = leaky_ReLU(mixedDat, negSlope)
            mixedDat = np.dot(mixedDat, all_mixingList[i][l])
        
        all_mixedDat.append(mixedDat)
    all_mixedDat = np.concatenate(all_mixedDat, axis=-1)
    x_l = np.copy(all_mixedDat)
    for i in range(lags):
        xt.append(x_l[:,i,:])
    
    # randomly generate a permutation matrix
    # 1->2->4
    layer_mixing_list = []
    for idx, z_dim in enumerate(z_dim_list):
        if idx == len(z_dim_list) - 1:
            # make it is ones but not gaussian
            layer_mixing_list.append(np.ones((z_dim, obs_dim)))
            #layer_mixing_list[-1][np.random.rand(*layer_mixing_list[-1].shape) < 0.7] = 0
        else:
            # make it is ones but not gaussian
            layer_mixing_list.append(np.ones((z_dim, z_dim_list[idx+1])))
            # layer_mixing_list[-1][np.random.rand(*layer_mixing_list[-1].shape) < 0.7] = 0
        
    # Mixing function
    for i in range(length):
        # Transition function
        y_t_noise = np.random.normal(0, noise_scale, (batch_size, latent_size)) * 2
        x_t_noise = np.random.normal(0, noise_scale, (batch_size, latent_size))
        # Modulate the noise scale with averaged history
        # y_t_noise = y_t_noise * np.mean(y_l, axis=1)
        y_t = 0
        # transition
        for l in range(lags):
            y_t += leaky_ReLU(np.dot(y_l[:,l,:], transitions[l]), negSlope)
        y_t = leaky_ReLU(y_t, negSlope)


        p_hist = 0.2 # the weight of history data
        y_t[:,:z_dim_list[0]] = y_t[:,:z_dim_list[0]] + 1 * y_t_noise[:,:z_dim_list[0]]  # first layer variable is related to the 
        # y_t[:,0] = y_t[:,0] + 1 * y_t_noise[:,0]  # first layer variable is related to the 
        
        for layer in range(1, n_layer):
            for i in range(z_dim_list[layer]):
                if inst:
                    y_t[:,i] = y_t[:,i] * y_t_noise[:,i] + y_t_noise[:,i]
                else:
                    # y_t[:,i] = (p_hist * y_t[:,i] + (1-p_hist) * y_t[:,i-1]) * y_t_noise[:,i] + y_t_noise[:,i]
                    y_t[:, sum(z_dim_list[:layer]) + i] = \
                    p_hist * y_t[:, sum(z_dim_list[:layer]) + i]+ \
                        (1-p_hist) * i * y_t[:, sum(z_dim_list[:layer - 1]) : sum(z_dim_list[:layer])] @ layer_mixing_list[layer-1][:, i] + 1 * y_t_noise[:,i]

        yt.append(y_t)
        # Mixing function
        all_mixedDat = []   
        for k in range(1):
            mixedDat = np.copy(y_t[:, -z_dim_list[-1]:])
            for l in range(Nlayer - 1):
                mixedDat = leaky_ReLU(mixedDat, negSlope) # + x_t_noise * 0.05
                mixedDat = np.dot(mixedDat, all_mixingList[k][l])
            all_mixedDat.append(mixedDat)
        all_mixedDat = np.concatenate(all_mixedDat, axis=-1)    
        x_t = np.copy(all_mixedDat)
        xt.append(x_t)
        y_l = np.concatenate((y_l, y_t[:,np.newaxis,:]),axis=1)[:,1:,:]

    yt = np.array(yt).transpose(1,0,2); xt = np.array(xt).transpose(1,0,2)
    
    np.savez(os.path.join(path, "data"), 
            yt = yt, 
            xt = xt)
    print(yt.shape)
    print(xt.shape)

    for l in range(lags):
        B = transitions[l]
        np.save(os.path.join(path, "W%d"%(lags-l)), B)    

def generate_5x5_matrices():
    """
    生成所有满足以下条件的 5x5 二进制矩阵（numpy.ndarray）：
      1. 第 0 行全为 1
      2. 第 0 列全为 1
      3. 对于其余行和列，每行/列只包含 2 个 1
         （其中一个在第 0 列/行，另一个在子矩阵 4x4 内）
    返回：含有所有有效矩阵的列表
    """
    results = []
    # 遍历所有长度为 4 的排列
    for perm in permutations(range(4)):
        # 创建一个 5x5 的零矩阵
        mat = np.zeros((5, 5), dtype=int)
        
        # 第 0 行全部置为 1
        mat[0, :] = 0
        # 第 0 列全部置为 1
        mat[:, 0] = 0
        mat[0,0] = 1
        
        # 在 mat[1..4, 1..4] 区域内放置置换矩阵：
        # perm[i] 表示第 i 行 (在子矩阵中的第 i 行) 的 1 放在第 perm[i] 列
        for i in range(4):
            mat[i + 1, perm[i] + 1] = 1
        
        results.append(mat)
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=773)
    args = parser.parse_args()

    # n = m = 5
    # count = 0 
    # for idx, mat in enumerate(all_binary_matrices_np(n, m), 1):
    #     print(f"Matrix {idx}:")
    #     print(mat)
        # for row in mat:
        #     print(row)
        # print("-" * 20)
    # matrices = generate_5x5_matrices()
    # print(f"共生成了 {len(matrices)} 个符合要求的 5x5 矩阵。")

    # 打印其中前几个看看
    # for idx, m in enumerate(matrices, start=1):
    #     print(f"=== Matrix {idx} ===")
    #     print(m)
    #     stationary_z_link(5, 2, False, 'A_'+str(idx), matrix=m)
    # stationary_z_link(8, 2, False, 'G', seed=772, z_dim_list=[8, 8, 8])
    # stationary_z_link(4, 2, False, 'H', seed=772, z_dim_list=[4, 4, 4])
    # stationary_z_link(8, 2, False, f'G_{args.seed}', seed=args.seed, z_dim_list=[8, 8, 8])
    stationary_z_link(10, 2, False, f'I_{args.seed}', seed=args.seed, z_dim_list=[10, 10, 10])
    # stationary_z_link(5, 2, False, 'B')
    # stationary_z_link(8, 2, False, 'C')
    # stationary_z_link(8, 0, False, 'D')
    # stationary_z_link(8, 0, True, 'E')
    # stationary_z_link(16, 2, True, 'F')
