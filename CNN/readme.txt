文件：
    1.Reference为写代码时候的参考例程（包含了手写数字识别，读取图片例程)
    2.CNN_BP.py是构建的图模型
    3.TF_readImage.py是tensorflow读出图片的流程
    4.result文件夹存储了训练好的卷积神经网络

tensorflow的定义流程：
    1.构件图：
        计算图：是包含节点和边的网络。本节定义所有要使用的数据，也就是张量（tensor）对象（常量、变量和占位符），同时定义要执行的所有计算，即运算操作对象（Operation Object，简称 OP）。
        网络中的节点表示对象（张量和运算操作），每个节点可有多个输入，但只有一个输出，边表示运算操作之间流动的张量。计算图定义神经网络的蓝图，但其中的张量还没有相关的数值。
    2.计算图


图片张量格式：shape为 [ batch, in_height, in_weight, in_channel ]，其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。（也可以用其它值，但是具体含义不是很理解）
卷积核格式：shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。

变量：在初始化后可以产生改变（tensorflow中存在变量在计算图之前必须通过方法global_variables_initializer()来初始化变量，即把变量加载进入内存中）
占位符:定义一个可变的常量，占位符赋值后不用初始化就可以获取值

tensorflow的图像读取：
    tf使用一个线程源源不断的将硬盘中的图片数据读入到一个内存队列中，另一个线程负责计算任务，所需数据直接从内存队列中获取。
    tf在内存队列之前，还设立了一个文件名队列，文件名队列存放的是参与训练的文件名，要训练N个epoch，则文件名队列中就含有N个批次的所有文件名。
    创建tf的文件名队列就需要使用到 tf.train.slice_input_producer 函数
    文件名队列——>内存中数据队列——>进入计算
    流程:
        调用 tf.train.slice_input_producer，从本地文件里抽取tensor，准备放入Filename Queue（文件名队列）中;——>tensor列表生成器
        调用 tf.train.batch，从文件名队列中提取tensor，使用单个或多个线程，准备放入内存队列;——>tensor队列生成器（保证多线程的时候读取顺序不发生错乱），弹出对应的读取数据后等待计算图计算
        调用 tf.train.Coordinator() 来创建一个线程协调器，用来管理之后在Session中启动的所有线程;
        调用 tf.train.start_queue_runners, 启动入队线程，由多个或单个线程，按照设定规则，把文件读入Filename Queue中。函数返回线程ID的列表，一般情况下，系统有多少个核，就会启动多少个入队线程（入队具体使用多少个线程在tf.train.batch中定义）;
        文件从 Filename Queue中读入内存队列的操作不用手动执行，由tf自动完成;
        调用 sess.run 来启动数据出列和执行计算;
        使用 coord.should_stop()来查询是否应该终止所有线程，当文件队列（queue）中的所有文件都已经读取出列的时候，会抛出一个 OutofRangeError的异常，这时候就应该停止Sesson中的所有线程了;(循环终止条件)
        使用 coord.request_stop()来发出终止所有线程的命令，使用coord.join(threads)把线程加入主线程，等待threads结束。


