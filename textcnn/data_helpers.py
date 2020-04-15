# coding:utf-8
import numpy as np
import re
import itertools
from collections import Counter

# 剔除英文的符号


def load_data_and_labels(data_file):
    train_examples = open(data_file, "rb")
    train_examples.readline()
    x = []
    y = []
    max_len = 0
    for line in train_examples:
        line = line.decode('utf-8', 'ignore')
        line_split = line.strip('\n').split(',')
        x.append(line_split[-1])
        max_len = max(max_len, len(line_split[-1]))
        # if line_split[0] == '1':
        #     y.append([0, 1])
        # elif line_split[0] == '0':
        #     y.append([1, 0])
        y.append([int(i) for i in line_split[:-1]])

    return np.array(x), np.array(y), max_len


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    利用迭代器从训练数据的回去某一个batch的数据
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # 每回合打乱顺序
        if shuffle:
            # 随机产生以一个乱序数组，作为数据集数组的下标
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        # 划分批次
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]





