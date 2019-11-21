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
def knnConstruct(X,kTup):
    sparse_matrix = Graph.constructDist(X,kTup)
    return sparse_matrix

# 构造拉普拉斯矩阵
# 输入：稀疏矩阵sparse_matrix
# 返回：拉普拉斯矩阵:laplacian_mat,度矩阵:D,邻接矩阵：A
