# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-27

"""
使用tensorflow实现卷积神经网络编程
"""

import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()  #先构建一个session在构建一个计算图
a1 = [[1,0,0,0],[0,3,1,2],[6,0,0,0]]
b = [[1,0,0,0],[3,1,1,2],[6,0,0,0]]

prediction = tf.equal(tf.argmax(a1,axis=1),tf.argmax(b,axis=1))
a = tf.reduce_mean(tf.cast(prediction,dtype=tf.float32))


# sess = tf.Session()
print(a.eval()) #需要预先定义图