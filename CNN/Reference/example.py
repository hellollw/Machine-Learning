# -*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def weight_variable(shape):
    # 产生随机变量
    # truncated_normal：选取位于正态分布均值=0.1附近的随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride = [1,水平移动步长,竖直移动步长,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #卷积函数，stride为卷积步长，same为考虑边界


def max_pool_2x2(x):
    # stride = [1,水平移动步长,竖直移动步长,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #ksize为池化窗口大小，strides为池化窗口步长


# 读取MNIST数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)   #手写数据集（每张图片为1x784维度）

sess = tf.InteractiveSession()  #先构建一个session在构建一个计算图

# 预定义输入值X、输出真实值Y    placeholder为占位符（占位符为tf中程序需要提前赋值的常量）
x = tf.placeholder(tf.float32, shape=[None, 784])   #先预定义过程，之后在执行的时候在具体赋值，先预定义维度
y_ = tf.placeholder(tf.float32, shape=[None, 10])   #对于多分类的输出也需要将标签扩容为相应位数
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1, 28, 28, 1])    #改变张量形状——将输入的x变为原先的28*28图片格式，每个像素点都为list的一个对象（4维对象像一个体形状）

# 卷积层1网络结构定义
# 卷积核1：patch=5×5;in size 1;out size 32;激活函数reLU非线性处理
W_conv1 = weight_variable([5, 5, 1, 32])    #卷积层变量定义
b_conv1 = bias_variable([32])   #定义偏移量，为一个32维的
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28*28*32 激活函数采用relu函数，卷积层采取权值加上偏移量的函数
h_pool1 = max_pool_2x2(h_conv1)  # output size 14*14*32#卷积层2网络结构定义 网络结构计算(VALID为维度/核维度)

# 卷积核2：patch=5×5;in size 32;out size 64;激活函数reLU非线性处理
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2)  # output size 7 *7 *64 #层次降维

# 全连接层1，以向量作为输入格式（输入编程数量为向量）
W_fc1 = weight_variable([7 * 7 * 64, 1024]) # 1024个神经元
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # [n_samples,7,7,64]->>[n_samples,7*7*64] 进入全链接层与之前的线性分类器一致
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)# 还是以relu作为激活函数，矩阵相乘
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 减少计算量dropout（防止过拟合，神经元会被以一定概率选中并在这次迭代中不更新权值），只有当keep_prob = 1时，才是所有的神经元都参与工作

# 全连接层2
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2   #最后输出10个表示对应的识别结果
######################################################################################################训练代码############################################################################
# 二次代价函数:预测值与真实值的误差,同时求取均值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=prediction))    #计算labels和logits之间的交叉熵（cross entropy）

# 梯度下降法:数据太庞大,选用AdamOptimizer优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)    #梯度下降法

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))   #寻找索引
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()  # defaults to saving all variables
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist.train.next_batch(50)  #分batch训练？ 循环输出50个样本集进行训练
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step", i, "training accuracy", train_accuracy)   #

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})   #优化器可以直接执行run()函数

# 保存模型参数
saver.save(sess, './model.ckpt')

######################################################################################################测试代码############################################################################
# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))   #以列维度缩减，返回索引
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  #cast执行张量的数据转换，布尔型也可以转换,True表示1，False表示0

saver = tf.train.Saver()  # defaults to saving all variables

sess.run(tf.global_variables_initializer()) # 在含有变量型张量的情况下需要初始化所有变量

saver.restore(sess, './model.ckpt')  # 和之前保存的文件名一致

print("test accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:
    1.0})) # 计算图的输出为计算结果
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:
    1.0}))#sess.run一样

"""
读入样本
"""

# 生成图片路径和标签list
# train_dir='C:/Users/hxd/Desktop/tensorflow_study/Alexnet_dr'
zeroclass = []
label_zeroclass = []
oneclass = []
label_oneclass = []
twoclass = []
label_twoclass = []
threeclass = []
label_threeclass = []
fourclass = []
label_fourclass = []
fiveclass = []
label_fiveclass = []


# s1 获取路径下所有图片名和路径，存放到对应列表并贴标签
def get_files(file_dir, ratio):
    for file in os.listdir(file_dir + '/0'):
        zeroclass.append(file_dir + '/0' + '/' + file)
        label_zeroclass.append(0)
    for file in os.listdir(file_dir + '/1'):
        oneclass.append(file_dir + '/1' + '/' + file)
        label_oneclass.append(1)
    for file in os.listdir(file_dir + '/2'):
        twoclass.append(file_dir + '/2' + '/' + file)
        label_twoclass.append(2)
    for file in os.listdir(file_dir + '/3'):
        threeclass.append(file_dir + '/3' + '/' + file)
        label_threeclass.append(3)
    for file in os.listdir(file_dir + '/4'):
        fourclass.append(file_dir + '/4' + '/' + file)
        label_fourclass.append(4)
    for file in os.listdir(file_dir + '/5'):
        fiveclass.append(file_dir + '/5' + '/' + file)
        label_fiveclass.append(5)
    # s2 对生成图片路径和标签list打乱处理（img和label）
    image_list = np.hstack((zeroclass, oneclass, twoclass, threeclass, fourclass, fiveclass))
    label_list = np.hstack(
        (label_zeroclass, label_oneclass, label_twoclass, label_threeclass, label_fourclass, label_fiveclass))
    # shuffle打乱
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
    # 将所得List分为2部分，一部分train,一部分val，ratio是验证集比例
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))  # 验证样本数
    n_train = n_sample - n_val  # 训练样本数

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:]
    val_labels = all_label_list[n_train:]
    val_labels = [int(float(i)) for i in val_labels]
    return tra_images, tra_labels, val_images, val_labels


# 生成batch
# s1:将上面的list传入get_batch(),转换类型,产生输入队列queue因为img和lab
# 是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像
#   image_W, image_H, ：设置好固定的图像高度和宽度
#   设置batch_size：每个batch要放多少张图片
#   capacity：一个队列最大多少

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 转换类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)    #文件名队列
    # 入队
    input_queue = tf.train.slice_input_producer([image, label]) #构成文件名张量
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # 读取图像
    # s2图像解码，且必须是同一类型
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # s3预处理，主要包括旋转，缩放，裁剪，归一化
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)   # image的预处理
    # s4生成batch

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,    #出队数量
                                              num_threads=32,   #入队线程
                                              capacity=capacity)    # 队列中元素最大数量

    #构造batch结构，将文件名队列变为文件队列，训练时线程从文件名队列中读取文件队列放入内存中

    # 重新排列label，行数为[batch_size]???
    label_batch = tf.reshape(label_batch, [batch_size])
    # image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

