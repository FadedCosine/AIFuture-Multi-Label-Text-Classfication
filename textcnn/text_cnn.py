# coding:utf-8
import tensorflow as tf
import numpy as np

class TextCNN(object):
    '''
    sequence_length: 句子的长度，我们把所有的句子都填充成了相同的长度(该数据集是59)。
    num_classes: 输出层的类别数，我们这个例子是2(正向和负向)。
    vocab_size: 我们词汇表的大小。定义 embedding 层的大小的时候需要这个参数，embedding层的形状是[vocabulary_size, embedding_size]。
    embedding_size: 嵌入的维度。
    filter_sizes: 我们想要 convolutional filters 覆盖的words的个数，对于每个size，我们会有 num_filters 个 filters。比如 [3,4,5] 表示我们有分别滑过3，4，5个 words 的 filters，总共是3 * num_filters 个 filters。
    num_filters: 每一个filter size的filters数量(见上面)
    l2_reg_lambda:正则化处理时L2的参数
    '''
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # 定义占位符，在dropout层保持的神经元的数量也是网络的输入，因为我们可以只在训练过程中启用dropout，而评估模型的时候不启用。
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # 定义常量，Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        # Embedding layer，强制操作运行在CPU上。 如果有GPU，TensorFlow 会默认尝试把操作运行在GPU上，但是embedding实现目前没有GPU支持，如果使用GPU会报错。
        # 这个 scope 添加所有的操作到一个叫做“embedding”的高阶节点，使得在TensorBoard可视化你的网络时，你可以得到一个好的层次
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                # 它将词汇表的词索引映射到低维向量表示。它基本上是我们从数据中学习到的lookup table
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            # 创建实际的embedding操作。embedding操作的结果是形状为 [None, sequence_length, embedding_size] 的3维张量积
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # print "^^^^^^^embedded_chars^^^^^^",self.embedded_chars.get_shape()
            # (?, 56, 128)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # print "^^^^^^^embedded_chars_expanded^^^^^^",self.embedded_chars_expanded.get_shape()
            # (?, 56, 128, 1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积层
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # 定义参数，也就是模型的参数变量
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # 非线性激活
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 最大值池化，h是卷积的结果，是一个四维矩阵，ksize是过滤器的尺寸，是一个四维数组，第一位第四位必须是1，第二位是长度，这里为卷积后的长度，第三位是宽度，这里为1
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        # 在 tf.reshape中使用-1是告诉TensorFlow把维度展平，作为全连接层的输入
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Add dropout，以概率1-dropout_keep_prob，随机丢弃一些节点
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # tf.nn.xw_plus_b 是进行 Wx+b 矩阵乘积的方便形式
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.sigmoid(self.scores)
            self.one_hot_prediction = self.multi_label_hot(self.predictions,name="predictions")

        # 定义损失函数
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.T = tf.multiply(self.one_hot_prediction, self.input_y)
            self.P = tf.divide(tf.reduce_sum(tf.cast(self.T, "float")), tf.reduce_sum(self.one_hot_prediction))
            self.R = tf.divide(tf.reduce_sum(tf.cast(self.T, "float")), tf.reduce_sum(self.input_y))
            self.F1 = tf.divide(2 * self.P * self.R, self.P + self.R, name="f1_score")

            correct_predictions = tf.equal(self.one_hot_prediction, self.input_y )

            self.accuracy = tf.reduce_mean(tf.cast(tf.reduce_all(correct_predictions, 1), "float"), name="accuracy")


    def multi_label_hot(self, prediction, name, threshold=0.5):
        prediction = tf.cast(prediction, "float")
        threshold = float(threshold)
        return tf.cast(tf.greater(prediction, threshold), "float", name=name)