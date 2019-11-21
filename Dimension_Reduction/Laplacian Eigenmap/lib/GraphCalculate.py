# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-21

"""
图论计算公式：
    不同核方式计算图的边界权值
"""

from numpy import *

# 定义核函数计算公式
# 输入：m个样本数据：datam, 一个想要进入核函数矩阵的样本数据i:datai, 描述核函数的元组:kTup
# 输出：对于i样本在m个样本数据中的核矩阵:dataK
def kernelCal(datam, datai, kTup):
    """
    # 定义核函数计算公式
    # 输入：m个样本数据：datam, 一个想要进入核函数矩阵的样本数据i:datai, 描述核函数的元组:kTup
    # 输出：对于i样本在m个样本数据中的核矩阵:dataK
    """
    m, n = shape(datam)
    datai = mat(datai).transpose()  # 转换为列向量,transpose()默认与.T相同
    dataK = mat(zeros((m, 1)))
    if kTup[0] == 'lin':  # 该线性内核指的是向量内积运算
        dataK = datam * datai
    elif kTup[0] == 'rbf':
        sigma = kTup[1]
        for j in range(m):
            dataj = datam[j, :].transpose()
            dif = dataj - datai
            dataK[j, :] = dif.T * dif
        dataK = exp(dataK / (-2 * sigma ** 2))  # 元素除法
    elif kTup[0] == 'dist':  # 距离核
        for j in range(m):
            dataj = datam[j, :].T
            dif = dataj - datai
            dif = dif.T * dif
            dataK[j, 0] = sum(dif)
    else:
        raise NameError('That kernel is not defined')
    return dataK

# 选取k近邻构建稀疏矩阵
# 输入：输入样本数据：datamat, 选择近邻构建方式:kTup
# 输出：稀疏矩阵:kmat
def constructKmat(datamat, kTup):
    curkTup = kTup[1:]
    m = shape(datamat)[0]
    kmat = mat(zeros((m, m))) + inf
    for i in range(m):
        samplei = datamat[i, :]  # 取出样本
        dataKernal = kernelCal(datam=datamat, datai=samplei, kTup=curkTup)
        sampleindex = dataKernal.T.A[0].argsort()  # 先转换为一维array，再返回从小到大的索引
        for j in range(curkTup[2]):  # 取前k个近邻,构建稀疏矩阵（为对称矩阵）
            kmat[sampleindex[j], i] = dataKernal[sampleindex[j], 0]
            kmat[i, sampleindex[j]] = dataKernal[sampleindex[j], 0]  # 满足图的对称性
    return kmat

# 构造距离矩阵dist(使用不同的内核取构造距离矩阵），经典MDS输入的距离矩阵应该为欧式矩阵
# 输入：输入样本数据：datamat, 核方式:kTup
# 输出：距离矩阵dist
def constructDist(datamat, kTup):
    """

    :param kTup:可选形式：
    1.rbf内核（'rbf',超参数:sigma=)
    2.向量内积内核('lin',)
    3.距离内核('dist',)
    4.k近邻选取('knn',边界计算方式: , 计算方式参数: , k近邻居数量: )
    :return: 图的距离表示矩阵
    """
    m = shape(datamat)[0]
    dist = mat(zeros((m, m)))
    if kTup[0] == 'knn':  # knn近邻选取
        dist = constructKmat(datamat, kTup)
    else:   #其余的都在核函数函数中
        for i in range(m):
            dist[:, i] = kernelCal(datamat, datamat[i, :], kTup)
    return dist