# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-20

"""
ISOMAP(Isometric Mapping)等度量映射
ISOMAP=Dijkstra+MDS
采用iris数据集进行降维后的数据观测

修改：
    1. 构建k近邻矩阵。这里展示的应该是一个表示样本与样本关系的图，所以图应该是对称的，所以这里的每个样本的k近邻并非严格的k近邻
"""
from lib import Knn as Knn
from lib import dijkstra as dij
from lib import MDS as MDS
from numpy import *
import matplotlib.pyplot as plt

# 将训练集图示出来（适用于二维空间）——每个样本对应的标签值不同绘制的颜色不同
# 输入：训练集：dataMat, 各个样本分类：clusterassement, 样本总标签序列：labellist
# 输出：显示图像
def plotdata(dataMat, clusterassement,labellist):
    fig = plt.figure()
    ax0 = fig.add_subplot(111)
    ax0.set(title='Datashow', ylabel='Y-Axis', xlabel='X-Axis')

    markers = ['+', 'o', '*', 'x', 'd', '.', 'd', '^']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

    m = shape(dataMat)[0]
    for i in range(m):
        if clusterassement[i]==labellist[0]:
            ax0.scatter(dataMat[i,0], dataMat[i,1], marker=markers[0], color=colors[0])
        elif clusterassement[i]==labellist[1]:
            ax0.scatter(dataMat[i, 0], dataMat[i, 1], marker=markers[1], color=colors[1])
        elif clusterassement[i]==labellist[2]:
            ax0.scatter(dataMat[i, 0], dataMat[i, 1], marker=markers[2], color=colors[2])
        else:
            print(clusterassement[i]+'该标签不存在')
    plt.show()

# 选取k近邻构建稀疏矩阵
# 输入：输入样本数据：datamat, k近邻数量:k
# 输出：稀疏矩阵:kmat
def constructKmat(datamat, k):
    m = shape(datamat)[0]
    kmat = mat(zeros((m, m))) + inf
    for i in range(m):
        samplei = datamat[i, :]  # 取出样本
        dataKernal = Knn.kernelCal(datam=datamat, datai=samplei, kTup=('dist',))
        sampleindex = dataKernal.T.A[0].argsort()  # 先转换为一维array，再返回从小到大的索引
        for j in range(k):  # 取前k个近邻,构建稀疏矩阵（为对称矩阵）
            kmat[sampleindex[j], i] = dataKernal[sampleindex[j], 0]
            kmat[i, sampleindex[j]] = dataKernal[sampleindex[j], 0]  # 满足图的对称性
    return kmat

# 构造距离矩阵dist(使用不同的内核取构造距离矩阵），经典MDS输入的距离矩阵应该为欧式矩阵
# 输入：输入样本数据：datamat, 核方式:kTup
# 输出：距离矩阵dist
def constructDist(datamat,kTup):


# 对样本之间的连接图进行Dijkstra算法处理，以最短路径作为样本间距离
# 输入： 样本间的连接图：kmat
# 输出：dijkstra算法输出的全连接图dijmat
def dijkstra(kmat):
    m = shape(kmat)[0]
    dijmat = []
    for i in range(m):
        founded,passpath = dij.dijkstra(kmat,i)
        dijmat.append(founded)
    return mat(dijmat)

# 使用ISOMAP来对样本进行降维
# 输入： 样本数据：datamat,降至维度：d,选取的k近邻数量:k
# 输出： 降维后的样本数据矩阵：lowdimensiondata
def isomap(datamat,d,k):
    kmat = constructKmat(datamat,k)
    dijmat = dijkstra(kmat)
    lowdimensiondata = MDS.mds(dijmat,d)
    return lowdimensiondata


if __name__ == '__main__':
    path = './test/iris.txt'
    trainingSet, traininglabel, testset, testlabel = Knn.readData(path)
    labellist = ['Iris-setosa','Iris-versicolor','Iris-virginica']
    lowdimensiondata = isomap(trainingSet,2,40)
    plotdata(lowdimensiondata,traininglabel,labellist)

