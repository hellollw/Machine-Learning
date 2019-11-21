# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-20

"""
MDS算法（多维缩放）

经典MDS算法：
输入dist距离矩阵应为欧式矩阵，这样计算出的内积矩阵B的特征根才能全部为正

修改：
1. 矩阵乘方**为矩阵运算，array乘方**为每个元素相乘。故要实现矩阵乘方需要用multiply点乘
2. all()运算对矩阵内进行了一次与运算，any()对矩阵内进行一次或运算
3. mds输入的距离矩阵必须为对称矩阵
4. 将特征值降序排列（对应特征向量也需要重新组合）
5. 计算得到的B不是对称矩阵，导致特征值出现虚数(原因：python数据精度不一样）
6. 输入的是实对称矩阵计算出来的特征值含有虚部？

"""

from numpy import *
import matplotlib.pyplot as plt


# 判断矩阵是否对称
# 输入：矩阵sym_mat
# 输出：判断结果：True or False
def symmetric(sym_mat):
    m, n = shape(sym_mat)
    if m != n:
        return False  # 矩阵维度不一致，不可能对称
    else:
        for i in range(m):
            for j in range(0, i + 1):
                if sym_mat[i,j] == sym_mat[j,i]:
                    continue
                else:
                    print(sym_mat[i,j],sym_mat[j,i])
                    print(i,j)
                    return False
        return True


# MDS算法
# 输入：距离矩阵：dist(为正定矩阵（正定一定为对称阵））， 降维维度：d
# 输出：降维后的样本坐标：lowdimensionloc
def mds(dist, d):
    m = shape(dist)[0]
    dist_square = multiply(mat(dist), mat(dist))
    row_mean = mean(dist_square, 1)  # 行均值按列缩减
    col_mean = mean(dist_square, 0)
    total_mean = mean(dist_square)
    B = mat(zeros((m,m)))
    # 获得降维后的内积矩阵(强行对称）
    for i in range(m):
        for j in range(0,i+1):
            B[i,j] = -0.5*(dist_square[i,j]-row_mean[i,0]-col_mean[0,j]+total_mean)
            B[j,i] = B[i,j]
    # B = -0.5 * (dist_square - row_mean - col_mean + total_mean)
    # print(symmetric(B))
    # 矩阵的特征分解
    eigen_val, eigen_vec = linalg.eig(B)
    print(eigen_val)
    val_index = argsort(-eigen_val)  # 将eigenvalue按照降序排列
    # 选取k个维度（特征值由大到小）
    eigen_low_value = zeros((1, d))[0]
    eigen_low_vector = mat(zeros((m, d)))
    for di in range(d):
        low_index = val_index[di]
        eigen_low_value[di] = eigen_val[low_index]
        eigen_low_vector[:, di] = eigen_vec[:, low_index]
    # 判断矩阵是否为正定
    if (eigen_low_value >= 0).all():
        # 取前d个特征值构成低维空间
        lowdimensionloc = eigen_vec[:, :d] * sqrt(diag(eigen_val[:d]))
        return lowdimensionloc
    else:
        raise SystemError('特征根为负,无法计算')


if __name__ == '__main__':
    D = [[0, 587, 1212, 701, 1936, 604, 748, 2139, 2182, 543],
         [587, 0, 920, 940, 1745, 1188, 713, 1858, 1737, 597],
         [1212, 920, 0, 879, 831, 1726, 1631, 949, 1021, 1494],
         [701, 940, 879, 0, 1374, 968, 1420, 1645, 1891, 1220],
         [1936, 1745, 831, 1374, 0, 2339, 2451, 347, 959, 2300],
         [604, 1188, 1726, 968, 2339, 0, 1092, 2594, 2734, 923],
         [748, 713, 1631, 1420, 2451, 1092, 0, 2571, 2408, 205],
         [2139, 1858, 949, 1645, 347, 2594, 2571, 0, 678, 2442],
         [2182, 1737, 1021, 1891, 959, 2734, 2408, 678, 0, 2329],
         [543, 597, 1494, 1220, 2300, 923, 205, 2442, 2329, 0]]
    label = ['Atlanta', 'Chicago', 'Denver', 'Houston', 'Los Angeles', 'Miami', 'New York', 'San Francisco', 'Seattle',
             'Washington, DC']
    X = mds(D, 2)
    plt.plot(X[:, 0], X[:, 1], 'o')
    for i in range(X.shape[0]):
        plt.text(X[i, 0] + 25, X[i, 1] - 15, label[i])
    plt.show()
