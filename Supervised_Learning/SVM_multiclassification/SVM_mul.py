# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-13

"""
使用SVM实现多分类
采用one vs one思想，针对n个目标类别，训练n(n-1)/2个SVM分类器，也就是每两个类别之间都训练一个SVM分类器，最后通过投票来抉择该样本属于哪一类（有点集成学习ensemble learning的思想）

数据训练：
本次采用著名的iris数据集（鸢尾花），其中包含了150个样本，一共3个类别，每5个样本取出一个作为测试集

构造多分类训练数据集：
因为自己写了一个ConstructTrainingSet函数，故：
    输入训练集样本数据需要为float
    输入训练集样本标签可为任意格式
    具体格式可参考iris数据集

"""
from numpy import *
import SVM as SVM  # 导入SVM分类器（SMO算法实现）


# 读取数据
# 输入变量： 样本集路径：path
# 返回变量： 训练集样本数据矩阵:trainingData, 训练集样本标签列表：traininglabel
#           测试集数据矩阵：testData, 测试集标签列表：testLabel
def readData(path):
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


# 构造多分类器训练数据集
# 输入：训练集数据矩阵：trainingData, 训练集样本标签矩阵:trainingLabel
# 输出：种类索引：Numlist, 标签列表：WholeLabelList， 训练数据列表:trainingList, 训练标签列表:trainingLabelList
def ConstructTrainingSet(trainingData, trainingLabel):
    m, n = shape(mat(trainingData))
    trainingList = []
    trainingLabelList = []
    WholeLabelList = []
    # 获得标签索引
    for label in trainingLabel:
        if label not in WholeLabelList:
            WholeLabelList.append(label)
    # 获得训练数据索引
    Num = len(WholeLabelList)
    Numlist = []
    for i in range(Num):
        for j in range(i + 1, Num):
            Numlist.append([i, j])
    # 获得训练数据列表
    for com in Numlist:
        Pos = WholeLabelList[com[0]]
        Neg = WholeLabelList[com[1]]
        Valid_Pos = [i for i, x in enumerate(trainingLabel) if x == Pos]  # 使用列表获得多个元素的索引值
        Valid_Neg = [i for i, x in enumerate(trainingLabel) if x == Neg]
        Valid = Valid_Pos + Valid_Neg
        Data_valid = mat(trainingData[Valid, :])
        Label_valid = mat(zeros((m, 1)))
        Label_valid[Valid_Pos, :] = [1]
        Label_valid[Valid_Neg, :] = [-1]
        Label_valid = mat(delete(Label_valid, nonzero(Label_valid[:, 0] == 0)[0], axis=0))  # 按行索引删除
        trainingList.append(Data_valid)
        trainingLabelList.append(Label_valid)
    return Numlist, WholeLabelList, trainingList, trainingLabelList


# 使用one vs one方法训练多分类SVM，对于k个种类需要训练k(k-1)/2个SVM分类器
# 输入：训练样本数据：trainingList, 训练样本标签：trainingLabelList, 约束常数：C，松弛变量:toler, 选择核函数类型：kTup, 最大迭代次数：maxIter
# 输出：k(k-1)/2个SVM分类器参数:SVMList, 种类索引：NumList, 标签列表：WholeLabelList
def TrainMulSVM(trainingMat, trainingLabel, C, toler, kTup, maxIter):
    NumList, WholeLabelList, trainingList, trainingLabelList = ConstructTrainingSet(trainingMat, trainingLabel)
    SVMList = []
    for i in range(len(trainingLabelList)):
        CurData = mat(trainingList[i])
        CurLabel = mat(trainingLabelList[i])
        CurSVM = SVM.outterLoop(CurData, CurLabel, C, toler, kTup, maxIter)
        SVMList.append(CurSVM)
        print('第%d个SVM训练完毕' % (i + 1))
    return SVMList, NumList, WholeLabelList


# 检测测试集
# 输入：k(k-1)/2个SVM分类器参数：SVMList, 测试集样本数据:testMat, 测试集样本标签：testLabel， 种类索引:NumList, 标签列表：WholeLabelList
# 输出：测试集测试标签：CalLabel, 测试集误差：WrongRate
def Test(SVMList, testMat, testLabel, NumList, WholeLabelList):
    testMat = mat(testMat)
    labelNum = len(testLabel)
    m, n = shape(testMat)
    ComLabel = mat(zeros((m, labelNum)))
    CalLabel = []
    WrongRate = 0
    for i in range(len(SVMList)):
        CurLabel, CurWrongRate = SVM.Test(SVMList[i], testMat, testLabel, True)
        POS_List = nonzero(CurLabel[:, 0] == 1)[0]
        Neg_List = nonzero(CurLabel[:, 0] == -1)[0]
        Add_Matrix = mat(zeros((m, labelNum)))  # 构造一个矩阵来实现测试集综合矩阵的更新
        Add_Matrix[POS_List, NumList[i][0]] = 1
        Add_Matrix[Neg_List, NumList[i][1]] = 1
        ComLabel += Add_Matrix  # 根据每个SVM分类器来更新测试集综合矩阵
    Max_index = argmax(ComLabel.A, axis=1)
    for k in range(len(Max_index)):
        CalLabel.append(WholeLabelList[Max_index[k]])
        if WholeLabelList[Max_index[k]] != testLabel[k]:
            WrongRate += 1
    WrongRate = WrongRate / m * 100
    return CalLabel, WrongRate


if __name__ == '__main__':
    trainingMat, trainingLabel, testMat, testLabel = readData('./test/iris.txt')
    SVMList, NumList, WholeLabelList = TrainMulSVM(trainingMat, trainingLabel, C=10, toler=0.00001, kTup=('lin', 1.3),
                                                   maxIter=40)
    CalLabel, WrongRate = Test(SVMList, testMat, testLabel, NumList, WholeLabelList)
    print(WrongRate)
