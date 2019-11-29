tensorflow的定义流程：
    1.构件图：
        计算图：是包含节点和边的网络。本节定义所有要使用的数据，也就是张量（tensor）对象（常量、变量和占位符），同时定义要执行的所有计算，即运算操作对象（Operation Object，简称 OP）。
        网络中的节点表示对象（张量和运算操作），每个节点可有多个输入，但只有一个输出，边表示运算操作之间流动的张量。计算图定义神经网络的蓝图，但其中的张量还没有相关的数值。
    2.计算图


图片张量格式：shape为 [ batch, in_height, in_weight, in_channel ]，其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。（也可以用其它值，但是具体含义不是很理解）
卷积核格式：shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。

变量：在初始化后可以产生改变（tensorflow中存在变量在计算图之前必须通过方法global_variables_initializer()来初始化变量，即把变量加载进入内存中）
占位符:定义一个可变的常量，占位符赋值后不用初始化就可以获取值
