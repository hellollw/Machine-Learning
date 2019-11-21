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

修改：
1. 先使用knn计算k近邻，再用rbf处理边的权重
2. line57 不同数据集的特征根选择不同！

"""

from numpy import *
from lib import GraphCalculate as Graph
from lib import BasicFunction as Basic


# knn建图
# 输入：样本数据矩阵:X, 选用的边值权重确定方式:kTup,近邻个数k
# 输出：稀疏矩阵sparse_matrix
def constructKnn(X, kTup, k):
    cur_kTup = ('knn', 'dist', 0, k)
    kmat = Graph.constructKmat(X, kTup=cur_kTup)
    if kTup[0] == 'rbf':
        sparse_matrix = exp(-kmat / kTup[1])
    else:
        raise NameError('还没设定另外的核方式计算权重')
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
# 输入：训练样本数据:datamat，边值的权重确定方式:kTup， 降维维数:d， 近邻个数：k
# 输出：选取d个最小的特征值对应的特征向量构成维度转换矩阵:W
def Laplacian_Eigenmap(datamat, kTup, d, k):
    m = shape(datamat)[0]
    sparse_matrix = constructKnn(datamat, kTup, k)
    laplacian_mat, D, A = constructLaplacian(sparse_matrix)
    eigenval, eigenvector = linalg.eig(linalg.inv(D) * laplacian_mat)  # eigenval为一维数组，vector为矩阵
    eigen_index = eigenval.argsort()  # 按照升序排列，取大于0的前d个特征向量作为维度转换矩阵

    # 找到第一个大于0的特征根位置（！特征根选择范围也很有讲究！）
    for i in range(len(eigen_index)):
        if eigenval[eigen_index[i]].real > 0:  # 可能会产生虚部
            first = i+2             # 将样本分离的情况需要更改特征根取值位置
            break
    W = mat(zeros((m, d)))
    for j in range(d):
        W[:, j] = eigenvector[:, eigen_index[first + j]]  # 取前d个最小的特征向量
        print(eigenval[eigen_index[first + j]])
    return W


if __name__ == '__main__':
    path = './test/iris.txt'
    training_mat, training_label, test_mat, test_label = Basic.readData(path)
    # 检验降维效果
    kTup = ('rbf', 2)
    k = 40
    W = Laplacian_Eigenmap(training_mat, kTup, 2, 40)
    labellist = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    Basic.plotdata(W, training_label, labellist)
