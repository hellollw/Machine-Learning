# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-19

"""
MDS算法（多维缩放）
"""

import numpy as np


# MDS算法
# 输入：距离矩阵：dist， 降维维度：d
# 输出：降维后的样本坐标：lowdimensionloc
def mds(D,q):
    D = np.asarray(D)
    DSquare = D**2
    totalMean = np.mean(DSquare)
    columnMean = np.mean(DSquare, axis = 0)
    rowMean = np.mean(DSquare, axis = 1)
    B = np.zeros(DSquare.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = -0.5*(DSquare[i][j] - rowMean[i] - columnMean[j]+totalMean)
    eigVal,eigVec = np.linalg.eig(B)
    X = np.dot(eigVec[:,:q],np.sqrt(np.diag(eigVal[:q])))

    return X