# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-13
"""
这里包含两种Kmeans算法
第一种是最基础的Kmeans，其迭代终止条件为样本聚类不再发生改变
第二种是基于基础Kmeans的二分Kmeans算法，其循环终止条件为码本数量达到所设定的数量，其基本原理是不断使用基础的Kmeans进行二分类，而后选取误差值改变最大的种群进行第二次分类，直至码本数量达到预设数量

修改：
1. 出现传入数组为空时报错
RuntimeWarning: Mean of empty slice.
解决：Kmeans函数的更新簇中心环节：
        先判断数组是否为空，再传入。即如果分类后属于该标签的样本数量为0，则保持该簇中心不变

"""

from numpy import *
import matplotlib
import matplotlib.pyplot as plt


# 将训练集图示出来（适用于二维空间）
# 输入：训练集：dataMat, 簇中心：centroids, 各个样本分类：clusterassement
# 输出：显示图像
def plotdata(dataMat, centroids, clusterassement):
    fig = plt.figure()
    ax0 = fig.add_subplot(111)
    ax0.set(title='Datashow', ylabel='Y-Axis', xlabel='X-Axis')

    markers = ['+', 'o', '*', 'x', 'd', '.', 'd', '^']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

    k = len(centroids[:, 0])

    for i in range(k):
        centroid_x = centroids[i, 0]
        centroid_y = centroids[i, 1]
        color = colors[i]
        marker = markers[i]
        ax0.scatter(centroid_x, centroid_y, marker=marker, color='k')  # 中心点使用特殊颜色标明

        currentdata = dataMat[nonzero(clusterassement[:, 0] == i)[0], :]  # 数组过滤
        currentdata_x = currentdata[:, 0].flatten().A[0]
        currentdata_y = currentdata[:, 1].flatten().A[0]  # 降维
        ax0.scatter(currentdata_x, currentdata_y, marker=marker, color=color)

    plt.show()


# 导入文本文件数据
# 输入：文本文件的文件名
# 输出：数据集(列表类型）
def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')  # strip()为去除首尾字符，split为分割出字符串('\t'是键盘上的tab键）
        if len(curline) != 1:
            filLine = list(map(float, curline))  # 含有空的字符串
            dataMat.append(filLine)

    # print(mat(dataMat))
    return dataMat


# 计算两样本之间的距离
# 输入：两个样本
# 输出：两个样本点之间的距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))


# 随机初始化初始k码本
# 输入：样本集合，码本数量
# 输出：初始化的k个码本矩阵
def randCent(dataMat, k):
    n = shape(dataMat)[1]  # 取特征数量
    centroids = mat(zeros((k, n)))

    for j in range(n):
        minJ = min(dataMat[:, j])
        maxJ = max(dataMat[:, j])  # 这里返回的依然是矩阵，即1x1的矩阵，需要转换为数值型才可可以相乘
        rangeJ = float(maxJ - minJ)  # 这里将矩阵转换为数值型数据
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)  # 取k个0~1之间的数作为乘积
    return centroids


# K均值聚类算法
# 输入：训练集dataMat，码本数量k
# 输出：簇中心centroids, 各个样本分类clusterassment
def kMeans(dataMat, k, distMeasure=distEclud, createinicent=randCent):
    m = shape(dataMat)[0]  # 样本数量
    centroids = createinicent(dataMat, k)
    clusterassment = mat(zeros((m, 2)))
    clusterassment[:, 0] = inf

    # 循环终止条件：当样本中簇分类不再发生改变
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        minIndex = -1
        # 更新样本所属
        for i in range(m):
            cluster_mindistance = inf
            for j in range(k):
                distance = distMeasure(dataMat[i, :], centroids[j, :])
                if distance < cluster_mindistance:
                    cluster_mindistance = distance
                    minIndex = j

            # 更新样本
            if clusterassment[i, 0] != minIndex:
                clusterassment[i, :] = minIndex, cluster_mindistance ** 2
                cluster_changed = True
        # 更新簇
        for cent in range(k):
            if len(nonzero(clusterassment[:, 0] == cent)[0]) != 0:  # 如果分类后属于该标签的样本数量为0，则保持该簇中心不变
                centroids[cent, :] = mean(dataMat[nonzero(clusterassment[:, 0] == cent)[0], :], axis=0)  # 数组过滤
            else:
                continue
    return centroids, clusterassment


# 二分K均值聚类算法
# 输入： 样本集dataMat,码本数量k
# 输出： 簇中心centroids, 各个样本分类clusterassment
def biKmeans(dataMat, k, distMeasure=distEclud):
    m = shape(dataMat)[0]
    centroid = mean(dataMat, axis=0).tolist()[0]  # centroid的扩容需要使用list容器。只有list容器才能够使用append方法添加簇中心
    centroids = [centroid]
    clusterassment = mat(zeros((m, 2)))
    iter = 1
    while len(centroids) < k:
        lessSSE = inf
        for cent in range(len(centroids)):
            if len(nonzero(clusterassment[:, 0] == cent)[0]):
                Curdata = dataMat[nonzero(clusterassment[:, 0] == cent)[0], :]

                Currcentroids, Currclusterassment = kMeans(Curdata, 2,
                                                           distMeasure=distMeasure)
                # 计算分类前的误差平方和
                SSEnotsplit = sum(clusterassment[nonzero(clusterassment[:, 0] != cent)[0], 1])
                # 计算分类后的误差平方和
                SSEsplit = sum(Currclusterassment[:, 1])
                CurrSSE = SSEnotsplit + SSEsplit
                if CurrSSE < lessSSE:
                    Bestcent = cent
                    Bestcentroids = Currcentroids
                    Bestclusterassment = Currclusterassment
                    lessSSE = CurrSSE
            else:
                continue
        print('the bestcent is:', Bestcent)
        print('number:', len(Currclusterassment[:, 1]))
        print('iter:', iter)
        iter += 1
        # 更新簇分类
        Bestclusterassment[nonzero(Bestclusterassment[:, 0] == 1)[0], 0] = len(centroids)  # 更新标签值
        Bestclusterassment[nonzero(Bestclusterassment[:, 0] == 0)[
                               0], 0] = Bestcent  # 当Bestcent为1时，下面识别过程也会出错，长度不可能存在为0的情况，故应该先让为1的赋值为长度为0
        centroids[Bestcent] = Bestcentroids[0, :].A[0]
        centroids.append(Bestcentroids[1, :].A[0])
        clusterassment[nonzero(clusterassment[:, 0].A == Bestcent)[0], :] = Bestclusterassment

    return mat(centroids), clusterassment


# 主函数
if __name__ == '__main__':
    dataSet = loadDataSet('testSet.txt')
    dataMat = mat(dataSet)
    centroids, clusterassement = biKmeans(dataMat, 4)
    plotdata(dataMat, centroids, clusterassement)
    SSE_bi = sum(clusterassement[:, 1])
    print('bikeams的误差为：', SSE_bi)
    Kcentroids, Kclusterassement = kMeans(dataMat, 4)
    plotdata(dataMat, Kcentroids, Kclusterassement)
    SSE_K = sum(Kclusterassement[:, 1])
    print('Kmeans的误差为：', SSE_K)
