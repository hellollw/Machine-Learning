# -*- coding:utf-8 -*-
# Author: Lu Liwen
# Modified Time: 2019-11-27

"""
使用tensorflow实现卷积神经网络编程

1.tensorflow的图像读取
"""

import tensorflow as tf
import numpy as np
from skimage import io

def hh():
    return tf.constant('hhh'),tf.constant('123')

image_shape = io.imread('./temp/test2/crystalline_0167.jpg')
print(np.shape(image_shape))