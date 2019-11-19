# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-19

"""
Dijkstra算法
"""

from numpy import *

# 求取输入一维数组的最小值和索引
# 输入：一维行向量：row_array
# 返回：向量最小值:minvalue, 最小值索引：minindex
def getMin(row_array):
    row_list = row_array.tolist()
    minvalue = min(row_list)
    minindex = row_list.index(minvalue)
    return minvalue,minindex

# dijkstra算法寻找起点到其余点的最短路径
# 输入：权重图：distmat(对称矩阵）, 起点：start(从0开始计数）
# 输出：起点至其余点的最短距离：founded, 起点路径：二维向量
def dijkstra(distmat, start):
    m = shape(distmat)[0]
    founded = zeros((1, m))[0] + inf
    unfounded = distmat[start, :].A[0]
    passpath = [] #经过路径
    for i in range(m):
        passpath.append([start])
    for iter_num in range(m):
        iter_num+=1
        cur_mindist,cur_minindex = getMin(unfounded)    #找出U中距离最小点
        founded[cur_minindex] = cur_mindist   #更新S集和
        unfounded[cur_minindex] = inf
        passpath[cur_minindex].append(cur_minindex)
        #更新U集和,遍历除已知最短路径之外的节点(未知最短路径那么founded中为inf）
        for node_num in nonzero(founded==inf)[0]:
            updatedist = cur_mindist+distmat[cur_minindex,node_num]
            if updatedist<unfounded[node_num]:
                unfounded[node_num]=updatedist
                passpath[node_num].append(cur_minindex) #路径中添加当前索引
    return founded,passpath




if __name__ == '__main__':
    distmat = mat([[0, 4, inf, 2, inf],[4, 0, 4, 1, inf],[inf, 4, 0, 1, 3],[2, 1, 1, 0, 7],[inf, inf, 3, 7, 0]])
    start = 0
    print(dijkstra(distmat,start))