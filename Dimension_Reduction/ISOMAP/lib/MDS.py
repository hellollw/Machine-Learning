# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-20

"""
MDS算法（多维缩放）

修改：
1. 矩阵乘方**为矩阵运算，array乘方**为每个元素相乘。故要实现矩阵乘方需要用multiply点乘
2. all()运算对矩阵内进行了一次与运算，any()对矩阵内进行一次或运算
3. mds输入的距离矩阵必须为正定矩阵（正定矩阵一定为对称矩阵） ??
4. 将特征值降序排列（对应特征向量也需要重新组合）

"""

from numpy import *


# MDS算法
# 输入：距离矩阵：dist(为正定矩阵（正定一定为对称阵））， 降维维度：d
# 输出：降维后的样本坐标：lowdimensionloc
def mds(dist, d):
    dist_square = multiply(mat(dist), mat(dist))
    row_mean = mean(dist_square, 1)  # 行均值按列缩减
    col_mean = mean(dist_square, 0)
    total_mean = mean(dist_square)
    B = mat(zeros(shape(dist)))
    # 获得降维后的内积矩阵
    B = -0.5 * (dist_square - row_mean - col_mean + total_mean)
    # 矩阵的特征分解
    eigen_val, eigen_vec = linalg.eig(B)
    print(eigen_val)
    # 判断矩阵是否为正定
    if (eigen_val >= 0).all():
        # 取前d个特征值构成低维空间
        lowdimensionloc = eigen_vec[:, :d] * sqrt(diag(eigen_val[:d]))
        return lowdimensionloc
    else:
        raise SystemError('距离矩阵应为正定性质')



if __name__ == '__main__':
    A = mat([[1, 2, 3], [2, 1, 4], [3, 4, 1]])
    print(mds(A, 2))
