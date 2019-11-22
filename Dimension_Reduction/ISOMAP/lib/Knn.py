# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-19

"""
knn（k近邻算法）

应用数据集：iris数据集
"""

from numpy import *
import operator


# 读取数据
# 输入变量： 样本集路径：path
# 返回变量： 训练集样本数据矩阵:trainingData, 训练集样本标签列表：traininglabel
#           测试集数据矩阵：testData, 测试集标签列表：testLabel
def readData(path):
    """
  读取数据
# 输入变量： 样本集路径：path
# 返回变量： 训练集样本数据矩阵:trainingData, 训练集样本标签列表：traininglabel
#           测试集数据矩阵：testData, 测试集标签列表：testLabel
    """
    trainingSet = []
    traininglabel = []
    testSet = []
    testLabel = []
    fr = open(path)
    i = 1
    for line in fr.readlines():
        lineArray = line.strip().split(',')
        if len(lineArray) == 5:
            if i % 5 == 0:  # 每5个数据作为一个测试样本
                testSet.append(list(map(float, [lineArray[0], lineArray[1], lineArray[2], lineArray[3]])))
                # python3的map返回一个迭代器
                testLabel.append(lineArray[4])
            else:
                trainingSet.append(list(map(float, [lineArray[0], lineArray[1], lineArray[2], lineArray[3]])))
                traininglabel.append(lineArray[4])
            i += 1
        else:
            print('第:%d个数据错误' % (i - 1))
    return mat(trainingSet), traininglabel, mat(testSet), testLabel


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
            dif = datam.T - datai
            dif_square = sum(multiply(dif, dif), 0)
            dataK = dif_square.T
    else:
        raise NameError('That kernel is not defined')
    return dataK


# knn近邻算法
# 输入：训练样本集和：datamat,训练样本标签：label， 测试样本(一维向量）:testvector，选取的近邻数量:k
# 输出：测试样本的分类标签：label_test
def knn(datamat, label_list, testvector, k, kTup=('dist',)):
    """
    # knn近邻算法
    # 输入：训练样本集和：datamat,训练样本标签：label， 测试样本(一维向量）:testvector，选取的近邻数量:k
    # 输出：测试样本的分类标签：label_test
    """
    label_test_dic = {}
    dist = kernelCal(datamat, testvector, kTup).T.A[0].argsort()  # 转换成一维数组，调用argsort()返回从小到大索引
    for i in range(k):
        label = label_list[dist[i]]
        label_test_dic[label] = label_test_dic.get(label, 0) + 1  # get函数若找到则返回索引，若没有则返回默认值
    # 选取字典中的最大值对应的键(sorted将输入的可迭代对象排序，返回排序后的可迭代对象）
    sortedlabel = sorted(label_test_dic.items(), key=operator.itemgetter(1), reverse=True)
    return sortedlabel[0][0]


if __name__ == '__main__':
    path = './test/iris.txt'
    trainingSet, traininglabel, testset, testlabel = readData(path)
    wrong = 0
    for i in range(shape(testset)[0]):
        label = knn(trainingSet, traininglabel, testset[i, :], 20)
        if label != testlabel[i]:
            wrong += 1
    print(wrong / shape(testset)[0])
