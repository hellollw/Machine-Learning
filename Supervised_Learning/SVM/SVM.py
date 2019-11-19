# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-13
"""
SMO实现SVM算法
"""
from numpy import *
import matplotlib.pyplot as plt


# 绘制样本数据和分类结果
# 输入：存储数据的结构体:os
# 输出：样本数据图像与超平面：fig1
def plotData(os):
    fig = plt.figure()
    ax0 = fig.add_subplot(111)
    ax0.set(title='Datashow', ylabel='Y-Axis', xlabel='X-Axis')

    markers = ['+', 'o', '*', 'x', 'd', '.', 'd', '^']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

    k = len(os.datamat)
    for i in range(k):
        if os.datalabel[i] == 1:
            marker = markers[1]
            color = colors[1]
        elif os.datalabel[i] == -1:
            marker = markers[2]
            color = colors[2]
        ax0.scatter(os.datamat[i, 0], os.datamat[i, 1], color=color, marker=marker)  # 绘制出样本点

    Support_vector = nonzero(os.alpha[:,0]>0)[0]    #取出支持向量的序号
    for s in Support_vector:
        ax0.scatter(os.datamat[s,0],os.datamat[s,1],s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red') #绘制支持向量点

    #   在二维平面上绘制出超平面
    if(os.kTup[0]=='lin'):
        w = calW(os)
        X_max = max(os.datamat[:,0]).A[0]
        X_min = min(os.datamat[:,0]).A[0]
        A = w[0,0]  #w为行向量，并非list
        B = w[0,1]
        C = float(os.b)    #Ax+By+C=0
        Y_max = -A*X_max/B-C/B
        Y_min = -A*X_min/B-C/B
        ax0.plot([X_max,X_min],[Y_max,Y_min],color='black')
    plt.show()


# 读取标签文本文档数据
# 输入：文本文档路径：path
# 输出：样本矩阵：datamat, 标签矩阵:datalabel
def readData(path):
    datamat = []
    datalabel = []
    fr = open(path)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        if len(lineArr) == 3:
            datamat.append([float(lineArr[0]), float(lineArr[1])])
            datalabel.append([float(lineArr[2])])  # 将读取的数据转换为字符串
        else:
            print('数据格式错误')
    return mat(datamat), mat(datalabel)


# 定义存储数据的类当作数据结构
# 输入： 样本数据:dataset， 样本标签:datalabel, 约束常数：C（控制“最大化间隔”和“完全切合样本分类”的权重， 松弛变量：toler, 核函数元组:kTup）
class optStruct:
    def __init__(self, dataset, datalabel, C, toler, kTup):
        self.datamat = mat(dataset)
        self.datalabel = mat(datalabel)  # 转换为列向量，对应样本矩阵
        self.m = shape(dataset)[0]  # 样本数量
        self.C = C
        self.toler = toler
        self.alpha = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # 存放已经优化过的样本（有效的）
        self.kTup = kTup
        self.kernel = mat((zeros((self.m, self.m))))
        for i in range(self.m):
            self.kernel[:, i] = kernelCal(self.datamat, self.datamat[i, :], self.kTup)


# 定义核函数计算公式
# 输入：m个样本数据：datam, 一个想要进入核函数矩阵的样本数据i:datai, 描述核函数的元组:kTup
# 输出：对于i样本在m个样本数据中的核矩阵:dataK
def kernelCal(datam, datai, kTup):
    m, n = shape(datam)
    datai = mat(datai).transpose()  # 转换为列向量,transpose()默认与.T相同
    dataK = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        dataK = datam * datai
    elif kTup[0] == 'rbf':
        sigma = kTup[1]
        for j in range(m):
            dataj = datam[j, :].transpose()
            dif = dataj - datai
            dataK[j, :] = dif.T * dif
        dataK = exp(dataK / (-2 * sigma ** 2))  # 元素除法
    else:
        raise NameError('That kernel is not defined')
    return dataK


# 限定输出参数的输出范围
# 输入：输入数据本身：x， 输出上界：H， 输出下界：L
# 输出：在输出范围内的数据：x_clip
def Clipper(x, H, L):
    if x > H:
        x_clip = H
    elif x < L:
        x_clip = L
    else:
        x_clip = x
    return x_clip


# 随机选取alpha2
# 输入： 已选取的样本的编号：alpha1， 样本矩阵：datam
# 输出： 随机选取的样本编号：alpha2
def randSelectj(alpha1, datam):
    m, n = shape(datam)
    alpha2 = random.randint(0, m)
    while alpha2 == alpha1:  # 唯一条件：不能让返回的样本编号与输入相同
        alpha2 = random.randint(0, m)
    return alpha2  # 返回样本数据


# 计算误差值E=f(x)-y
# 输入：存储数据的结构体：os, 需要计算误差的样本编号：k
# 返回：误差值Ek
def calE(os, k):
    Ek = multiply(os.datalabel, os.alpha).T * os.kernel[:,k] + os.b - \
         os.datalabel[k]
    return float(Ek)


# 按照步长最大的选取alpha2
# 输入：存储数据的结构体：os, 已经选取的样本编号：alpha1, 已选取样本的误差：Ei
# 输出：选取的第二个样本编号：alpha2, 第二个样本的误差：Ej
def SelectJ(os, alpha1, Ei):
    j = -1
    Ej = 0
    maxE = 0
    os.eCache[alpha1, :] = [1, Ei]  # 更新该样本状态（表示其可以被优化）
    ValueNum = nonzero(os.eCache[:, 0] != 0)[0]
    if len(ValueNum) > 1:
        for k in ValueNum:
            if k == alpha1: continue
            Ek = calE(os, k)
            difE = abs(Ei - Ek)
            if difE > maxE:
                j = k
                Ej = Ek
                maxE = difE
    else:
        j = randSelectj(alpha1, os.datamat)
        Ej = calE(os, j)
    return j, Ej


# 更新误差值E
# 输入：存储数据的结构体：os, 需要更新的样本编号：alpha
def Update(os, alpha):
    Ealpha = calE(os, alpha)
    os.eCache[alpha, :] = [1, Ealpha]


# 内循环
# 输入：存储数据的结构体：os, 需要更新的alpha编号alpha1
# 输出：alpha1是否发生更新：1（发生更新），0（没有发生更新）
def innerLoop(os, alpha1):
    Ei = calE(os, alpha1)
    yi = os.datalabel[alpha1, :]
    if ((yi * Ei < -os.toler) and (os.alpha[alpha1] < os.C)) or (
            (yi * Ei > os.toler) and (os.alpha[alpha1] > os.toler)):
        alpha2, Ej = SelectJ(os, alpha1, Ei)
        # 保存原先的alpha的值
        alpha1_old = os.alpha[alpha1].copy()
        alpha2_old = os.alpha[alpha2].copy()
        if os.datalabel[alpha1] * os.datalabel[alpha2] == 1:  # 取出的两个样本标签值同号
            L = max(0, alpha1_old + alpha2_old - os.C);
            H = min(os.C, alpha1_old + alpha2_old)
        elif os.datalabel[alpha1] * os.datalabel[alpha2] == -1:
            L = max(0, alpha2_old - alpha1_old);
            H = min(os.C, os.C + alpha2_old - alpha1_old)
        else:
            raise NameError('样本标签值不正确（即含有除-1，1以外的其他值')
        if L == H:
            print('alphai: %d L==H' % alpha1)
            return 0  # 两值相等，说明两个alpha均已经收敛到边界上，不需要优化
        else:
            # 计算带入关于alpha2的二阶导数
            eta = os.kernel[alpha1, alpha1] + os.kernel[alpha2, alpha2] - 2.0 * os.kernel[alpha1, alpha2]
            if eta <= 0:
                print('eta<=0')
                return 0  # 二阶导数小于0，函数极小值收敛在边缘
            else:
                # 更新alpha2
                os.alpha[alpha2] = alpha2_old + os.datalabel[alpha2] * (Ei - Ej) / eta
                os.alpha[alpha2] = Clipper(os.alpha[alpha2],H,L)    #限定alpha2的输出范围
                Update(os, alpha2)
                if abs(os.alpha[alpha2] - alpha2_old) < 0.00001:
                    print('alphai: %d  alphaj:%d is not moving enough' % (alpha1, alpha2))
                    return 0  # 小范围移动并不算做改变
                else:
                    # 更新alpha1
                    os.alpha[alpha1] = alpha1_old + os.datalabel[alpha1] * os.datalabel[alpha2] * (
                            alpha2_old - os.alpha[alpha2])
                    Update(os, alpha1)
                    b1 = os.b - Ei - os.datalabel[alpha1] * (os.alpha[alpha1] - alpha1_old) \
                         * os.kernel[alpha1, alpha1] - os.datalabel[alpha2] * (os.alpha[alpha2] - alpha2_old) * \
                         os.kernel[
                             alpha1, alpha2]
                    b2 = os.b - Ej - os.datalabel[alpha1] * (os.alpha[alpha1] - alpha1_old) \
                         * os.kernel[alpha1, alpha2] - os.datalabel[alpha2] * (os.alpha[alpha2] - alpha2_old) * \
                         os.kernel[
                             alpha2, alpha2]
                    if os.C > os.alpha[alpha1] > 0:
                        os.b = b1
                    elif os.C > os.alpha[alpha2] > 0:
                        os.b = b2
                    else:
                        os.b = (b1 + b2) / 2.0
                    return 1  # 更新完alpha1,alpha2,b，代表一次内循环完成，系数得到优化
    else:
        return 0  # 该alpha值满足KKT条件，不需要优化


# 外循环
# 输入：训练集样本：datamat, 训练集样本标签：datalabel. 约束常数：C，松弛变量：toler, 选择核函数类型：kTup, 最大迭代次数：maxIter
# 输出：训练后的存储数据结构：os
def outterLoop(datamat, datalabel, C, toler, kTup, maxIter):
    os = optStruct(dataset=datamat, datalabel=datalabel, C=C, toler=toler, kTup=kTup)  # C越大越容易产生过拟合
    Iter = 0
    WholeSet = True
    Alphachanged = 0
    while (Alphachanged != 0 and Iter < maxIter) or WholeSet:
        Alphachanged = 0
        if WholeSet:
            for i in range(os.m):
                Alphachanged += innerLoop(os, i)
            print('WholeSet  Iter: %d, Alphachanged: %d' % (Iter, Alphachanged))  # 打印输出
            Iter += 1
        else:
            nonBounds = nonzero((os.alpha[:, 0].A > 0) * (os.alpha[:, 0].A < os.C))[0]
            for k in nonBounds:
                Alphachanged += innerLoop(os, k)
            print('nonBounds Iter: %d, Alphachanged: %d' % (Iter, Alphachanged))
            Iter += 1
        if WholeSet:
            WholeSet = False
        elif Alphachanged == 0:
            WholeSet = True
    return os


# 根据Alpha值计算超参数w
# 输入：存储数据的结构：os
# 输出：超参数：w
def calW(os):
    validAlpha = nonzero(os.alpha[:, 0].A > 0)[0]
    w = multiply(os.datalabel[validAlpha, :], os.alpha[validAlpha, :]).T * os.datamat[validAlpha, :]
    return w

# 计算测试集数据
# 输入：存储数据的结构:os,测试集数据:datatest， 测试集标签：true_label, 是否计算多分类：mul
# 输出：测试集样本标签:testlabel, 测试集错误率：wrongrate
def Test(os,datatest,true_label,mul):
    m,n = shape(datatest)
    numError = 0
    testlabel = mat(zeros((m,1)))
    Support_Vector = nonzero(os.alpha[:,0]>0)[0]
    for t in range(m):
        testlabel[t] = multiply(os.alpha[Support_Vector,:],os.datalabel[Support_Vector,:]).T*kernelCal(os.datamat[
                                                                                                           Support_Vector,:],datatest[t,:],os.kTup)+os.b
        if testlabel[t]>=0:
            testlabel[t]=1
        else:
            testlabel[t]=-1
        if not mul:
            if testlabel[t]!=true_label[t]:
                numError+=1
    wrongrate = numError/m*100
    # print('wrongrate: %d percent' %wrongrate)
    return testlabel,wrongrate

if __name__ == '__main__':
    datamat, datalabel = readData('./test/trainingSet_Label_nonlinear.txt') #读取非标签值数据
    os = outterLoop(datamat=datamat, datalabel=datalabel, C=200, toler=0.0001, kTup=('rbf', 1.5), maxIter=100)
    plotData(os)
    trainingSet, trainingLabel = readData('./test/testSet_Label_nonlinear.txt')
    testlabel,wrongrate = Test(os,trainingSet,trainingLabel)    #查看测试集
    print('wrongrate:%d'%wrongrate)


