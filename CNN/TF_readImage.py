# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-30

"""
Tensorflow读取硬盘中的图片文件，构造数据训练所需要的batch结构

理解：*进入tensorflow的数据要转换为张量格式*
    1.tf.train.slice_input_producer定义了文件名放入列表的方式,构建张量列表（是否打乱，放入次数）
    2.tf.train.batch 定义了张量的存取队列（保证线程读取文件的顺序不会发生错乱，每次都是弹出一个张量对象）
    3.使用tf.train.Coordinator()来创建一个线程管理器（协调器）对象，管理session中启动的线程（因为读入数据是多线程读入）
    4.调用tf.train.start_queue_runners函数来启动执行文件名队列填充的线程（将tensor推入内存之中，数据流图开始计算，每次列表中弹出一个张量对象[imgae,label]，将
    5.读取数据的循环终止条件：while not coord.should_stop()当文件队列中所有文件都已经读出时会抛出OutofRangeError的异常
注意：
    1.CNN的输入输出都必须是同一个维度（故图片输入前需要进行统一裁剪）
    2.输入标签应该转换为one-hot向量，即多分类情况下对应位置的才为1
问题：
    1.numpy数组不能存放不同类型的值，自动转换为string类型
    2.TF中一些函数的使用需要初始化局部变量，故在计算图之前需要初始化变量（定义在main函数中的变量为局部变量)
    3.当样本集过少时会产生测试样本并不能满足所有标签

"""
import os
import numpy as np
import tensorflow as tf
import math
import time


# 将标签字符串文件转换为数字文件(符合sklearn的要求）
# 输入：字符串样本集和labelstring
# 输出：转换为对应的数字labelint, 集和种类：labelnum
def string2int(labelstring):
    Wholelabel = []
    labelint = []
    for label in labelstring:
        if label not in Wholelabel:
            Wholelabel.append(label)
        else:
            continue

    for cur_label in labelstring:
        label_index = int(Wholelabel.index(cur_label))
        labelint.append(label_index)  # 转换为对应的种类数字

    return labelint, int(len(Wholelabel))


# 构造one_hot向量
# 输入：文件样本标签索引:file_label, 总标签数量:label_number
# 输出：文件对应的one-hot向量:label_onehot
def getOneHotLabel(file_label, label_number):
    m = np.shape(file_label)[0]
    label_onehot = np.zeros((m, label_number))
    for j in range(m):
        label_onehot[j, file_label[j]] = 1
    return label_onehot


# 按一定比重生成文件乱序的训练样本和测试样本集和(在这里进行了one-hot标签转换)
# 输入:文件名位置:path 测试集比重:ratio
# 输出:训练样本数据:trainfile_name, 训练样本标签:trainfile_label, 训练样本标签种类:trainfile_num
#       测试样本数据:testfile_name, 测试样本标签:testfile_label,测试样本标签种类:testlabel_num
def getTrainAndTestData(path, ratio):
    # 生成文件名队列（name+label)，都为字符串格式
    """

    :param path: 文件名位置
    :param ratio: 测试集比重
    :return:
    """
    file_name_list = []
    file_label_list = []
    for forder_name in os.listdir(path):
        forder_path = path + forder_name + '/'
        for file_name in os.listdir(forder_path):
            if 'jpg' in file_name:
                file_name_list.append(forder_path + file_name)
                file_label_list.append(forder_name)  # 文件夹名称表示当前类别
            else:
                continue
    file_name_array = np.array([file_name_list, file_label_list]).T  # 转换为m*2的数组格式
    # 打乱样本集序列
    np.random.shuffle(file_name_array)
    m = np.shape(file_name_array)[0]
    m_test = int(np.ceil(m * ratio))  # 获得测试样本数量,转换为整型

    # 获得训练集样本
    trainfile_name = file_name_array[m_test:, 0]
    trainfile_label, trainfile_num = string2int(file_name_array[m_test:, 1])
    trainfile_label_onehot = getOneHotLabel(trainfile_label, trainfile_num)

    # 获得测试集样本
    testfile_name = file_name_array[0:m_test, 0]
    testfile_label, testlabel_num = string2int(file_name_array[0:m_test, 1])
    testfile_label_onehot = getOneHotLabel(testfile_label, trainfile_num)
    return trainfile_name, trainfile_label_onehot, trainfile_num,testfile_name, testfile_label_onehot,testlabel_num


# 生成输入的batch
# 输入：文件路径:filename, 文件标签:filelabel, 每次读取的文件数量:batchsize, 所需的图片样本的高度，image_height,所需的图片样本的宽度:width
# 输出：图片数据batch:imagebatch, 图片标签batch：labelbatch
def getBatch(filename, filelabel, batchsize, image_height, image_width):
    # 转换类型
    print('样本总数量为:%d'%np.shape(filename)[0])
    filename = tf.cast(filename, tf.string)
    filelabel = tf.cast(filelabel, tf.float32)
    # 构造文件名队列
    input_queue = tf.train.slice_input_producer([filename, filelabel], shuffle=False, num_epochs=1)  # 定义了样本放入文件名队列的方式
    imagename = input_queue[0]
    imagelabel = input_queue[1]
    # 读取图片文件
    image_contents = tf.read_file(imagename)
    image = tf.image.decode_jpeg(image_contents, channels=3)  # 图像解码
    image = tf.image.resize_image_with_crop_or_pad(image, image_height, image_width)    #图像裁剪
    image = tf.image.rgb_to_grayscale(image)    # 转换为灰度图
    image = tf.image.per_image_standardization(image)  # image的预处理,对图像矩阵进行归一化处理
    # 构造训练batch
    image_batch, label_batch = tf.train.batch([image, imagelabel],
                                              batch_size=batchsize,  # 出队数量
                                              num_threads=8,  # 入队线程
                                              capacity=64)  # 队列中元素最大数量

    return image_batch, label_batch


if __name__ == '__main__':
    starttime = time.time()
    # 定义需要使用到的系数
    path = './temp/'
    ratio = 0.2
    batchsize = 1
    image_height = 28
    image_weight = 28
    i = 1
    # 获得训练数据,构件图
    trainfile_name, trainfile_label_onehot, trainfile_num,testfile_name, testfile_label_onehot,testlabel_num = getTrainAndTestData(path, ratio=0.2)
    image_batch, label_batch = getBatch(testfile_name, testfile_label_onehot, batchsize, image_height, image_weight)
    # 计算图
    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)  # 把张量tensor推入内存之中
    # 输出
    try:
        while not coord.should_stop():  # 当文件队列中所有文件都已经读出时会抛出OutofRangeError的异常
            image, label = sess.run([image_batch, label_batch])
            print('第%d轮' % i )
            print(np.shape(image))
            print(label)
            i +=1
    except tf.errors.OutOfRangeError:
        print('complete')
    finally:
        coord.request_stop()  # 停止读入线程
    coord.join(threads)  # 线程加入主线程，等待主线程结束
    sess.close()
    endtime = time.time()
    print('程序运行耗时为：%.8s s' % (endtime - starttime))
