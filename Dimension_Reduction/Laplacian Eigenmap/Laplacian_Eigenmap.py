# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-21

"""
Laplacian Eigenmap(特征映射降维）

步骤：
    knn建图
    rbf核函数计算边权重
    构造拉普拉斯矩阵
    广义特征值计算

"""

from numpy import *
from lib import GraphCalculate as Graph


# knn建图
# 输入：样本数据矩阵:X, 选用的边值权重确定方式:kTup
# 输出：稀疏矩阵sparse_matrix
def constructKnn(X, kTup):
    sparse_matrix = Graph.constructDist(X, kTup)
    return sparse_matrix


# 构造拉普拉斯矩阵
# 输入：稀疏矩阵sparse_matrix
# 返回：拉普拉斯矩阵:laplacian_mat,度矩阵:D,邻接矩阵：A
def constructLaplacian(sparse_matrix):
    A = sparse_matrix
    D = diag(sum(A, axis=1).T.A[0])  # 按列缩减
    laplacian_mat = D - A
    return laplacian_mat, D, A


# 特征映射（求取维度转换矩阵W广义特征值求解） 求解方式：将其转化为特征值标准形式C=D.I*L，求解C的特征根
# 输入：训练样本数据:datamat，边值的权重确定方式:kTup 降维维数:d
# 输出：选取d个最小的特征值对应的特征向量构成维度转换矩阵:W
def Laplacian_Eigenmap(datamat, kTup, d):
    m = shape(datamat)[0]
    sparse_matrix = constructKnn(datamat, kTup)
    laplacian_mat, D, A = constructLaplacian(sparse_matrix)
    eigenval, eigenvector = linalg.eig(linalg.inv(D) * laplacian_mat)  # eigenval为一维数组，vector为矩阵
    eigen_index = eigenval.argsort()  # 按照升序排列，取大于0的前d个特征向量作为维度转换矩阵
    # 找到第一个大于0的特征根位置
    for i in range(len(eigen_index)):
        if eigenval[eigen_index[i]].real > 0:  # 可能会产生虚部
            first = i
            break
    W = mat(zeros((m, d)))
    for j in range(d):
        W[:, j] = eigenvector[:, eigen_index[first + j]]  # 取前d个最小的特征向量
        print(eigenval[eigen_index[first + j]])
    return W


if __name__ == '__main__':
    mat1 = mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    kTup = ('knn', 'rbf', 3, 3)
    print(Laplacian_Eigenmap(mat1, kTup, 2))
