# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-21

"""
一些基本的常用函数
数据读取readData(path)
数据描绘plotdata(dataMat, clusterassement, labellist)
"""
from numpy import *
import matplotlib.pyplot as plt


# 读取数据,数据以,分割
# 输入变量： 样本集路径：path
# 返回变量： 训练集样本数据矩阵:trainingData, 训练集样本标签列表：traininglabel
#           测试集数据矩阵：testData, 测试集标签列表：testLabel
def readData(path):
    """
  读取数据,
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


# 将训练集图示出来（适用于二维空间）——每个样本对应的标签值不同绘制的颜色不同
# 输入：训练集：dataMat, 各个样本分类：clusterassement, 样本总标签序列：labellist
# 输出：显示图像
def plotdata(dataMat, clusterassement, labellist):
    fig = plt.figure()
    ax0 = fig.add_subplot(111)
    ax0.set(title='Datashow', ylabel='Y-Axis', xlabel='X-Axis')

    markers = ['+', 'o', '*', 'x', 'd', '.', 'd', '^']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

    m = shape(dataMat)[0]
    for i in range(m):
        if clusterassement[i] == labellist[0]:
            ax0.scatter(dataMat[i, 0], dataMat[i, 1], marker=markers[0], color=colors[0])
        elif clusterassement[i] == labellist[1]:
            ax0.scatter(dataMat[i, 0], dataMat[i, 1], marker=markers[1], color=colors[1])
        elif clusterassement[i] == labellist[2]:
            ax0.scatter(dataMat[i, 0], dataMat[i, 1], marker=markers[2], color=colors[2])
        else:
            print(clusterassement[i] + '该标签不存在')
    plt.show()
