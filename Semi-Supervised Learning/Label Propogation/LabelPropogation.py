# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-22

"""
LabelPropogation(标签传播算法）
可用少量的标注数据得到大量无标注数据的标签值

使用iris数据集，用测试样本作为标签值，用训练样本作为无标签值

1.半监督学习的数据存放方式(有标签样本放在上面）：
    标签数据————标签label
        +
    无标签数据
        =
    样本数据集和datamat
"""

from numpy import *
from lib import BasicFunction as Basic
from lib import GraphCalculate as Graph


# 根据样本点构造k近邻图
# 输入：样本数据:datamat, 近邻选取数量:k
# 输出: k近邻关系
def constructKmat(datamat, k):
    kTup = ('knn', 'dist', 0, k)
    kmat = Graph.constructKmat(datamat, kTup)
    return kmat


# 构建权值矩阵W
# 输入：近邻图:kmat, 权重计算方式:kTup
# 输出：权值矩阵W
def constructWmat(kmat, kTup):
    if kTup[0] == 'rbf':  # 选用rbf内核
        W = exp(-kmat / kTup[1] ** 2)
    else:
        raise NameError('还没定义' + kTup[0] + '这种方法')
    return W


# 构建传播概率矩阵
# 输入:权值矩阵：W
# 输出：传播概率矩阵：T
def constructT(W):
    # T = mat(zeros(shape(W)))
    col_sum = sum(W, 0)  # 按行缩减
    T = W / col_sum
    return T


# 进入标签传播循环
# 输入：传播概率矩阵T, 有标签数据标签列表:label, 最大循环次数:maxiter
# 输出：无标签数据的标签矩阵:Yuu_label
def propogation(T, label, maxiter):
    m_label = len(label)
    m_unlabel = shape(T)[0] - m_label
    Tul = T[m_label:m_label + m_unlabel, 0:m_label].copy()
    Tuu = T[m_label:m_label + m_unlabel, m_label:m_label + m_unlabel].copy()
    # 找到标签总个数
    Whole_labellist = []
    for label_index in range(m_label):
        if label[label_index] not in Whole_labellist:
            Whole_labellist.append(label[label_index])
        else:
            continue
    label_num = len(Whole_labellist)
    # 构建标签矩阵
    Yll = mat(zeros((m_label, label_num)))
    for i in range(m_label):
        collabel = Whole_labellist.index(label[i])  # 找出标签所在的列记号
        Yll[i, collabel] = 1
    Yuu = mat(ones((m_unlabel, label_num)))
    # 进入循环
    iter = 0
    while iter < maxiter:
        Yuu = Tul * Yll + Tuu * Yuu
        print('现在为第%d轮循环'%iter)
        iter += 1
    Yuu_label_list = argmax(Yuu, 1).T.A[0]  # 以每行的最大值索引作为标签值
    Yuu_label = []
    for j in Yuu_label_list:
        Yuu_label.append(Whole_labellist[j])
    return Yuu_label


# 标签传播算法
# 输入：有标签样本数据:label_mat,无标签样本数据:unlabel_mat,标签集和:label,权值计算方式：kTup,近邻选取数量:k,最大迭代次数:maxiter
# 输出：无标签样本数据：unlabel_list
def labelPropogation(label_mat, unlabel_mat, label, kTup, k, maxiter):
    # 构造数据集
    ml, nl = shape(label_mat)
    mu, nu = shape(unlabel_mat)
    if nl != nu:
        raise SystemError('样本维度不一致')
    else:
        datamat = mat(zeros((ml + mu, nu)))
        datamat[0:ml, :] = label_mat.copy()
        datamat[ml:ml + mu, :] = unlabel_mat.copy()
        kmat = constructKmat(datamat, k)
        Wmat = constructWmat(kmat, kTup)
        T = constructT(Wmat)
        unlabel_list = propogation(T, label, maxiter)
        return unlabel_list


if __name__ == '__main__':
    path = './test/iris.txt'
    training_data, training_label, test_data, test_label = Basic.readData(path)
    kTup = ('rbf',3)
    k = 50
    maxiter = 80
    unlabel_list = labelPropogation(test_data,training_data,test_label,kTup,k,maxiter)
    wrong = 0
    for w in range(len(unlabel_list)):
        if unlabel_list[w]!=training_label[w]:
            wrong+=1
        else:
            continue
    print(wrong)