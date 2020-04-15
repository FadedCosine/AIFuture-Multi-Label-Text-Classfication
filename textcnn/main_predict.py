# coding:utf-8
# ! /usr/bin/env python
import jieba
import re
import tensorflow as tf
import numpy as np
import os
from optparse import OptionParser
from text_cnn import TextCNN
from tensorflow.contrib import learn

# 参数：我们这里使用命令行传入参数的方式执行该脚本
oparser = OptionParser()

oparser.add_option("-d", "--test-dataset", dest="test_dataset", help="the file path for test file.", default = "./dataset/test/texts.csv")

oparser.add_option("-m", "--model", dest="checkpoint_dir", help="Checkpoint directory from trained model", default = "./model/")

oparser.add_option("-o", "--prediction-file", dest="prediction_file", help="the file path for predict file", default = "./dataset/test/submit.csv")

oparser.add_option("-b", "--batch_size", dest="batch_size", help="Batch Size (default: 128)", default = 128)

oparser.add_option("-a", "--allow_soft_placement", dest="allow_soft_placement", help="Allow device soft device placement", default = True)

oparser.add_option("-l", "--log_device_placement", dest="log_device_placement", help="Log placement of ops on devices", default = False)

(options, args) = oparser.parse_args()


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

def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', file)
    return chinese

def load_fenci(read_filename):
    read_file = open(read_filename, 'rb')
    test_text = []
    for line in read_file:
        line = line.decode('utf-8', 'ignore').strip('\n').strip('\r')
        newline = jieba.cut(find_chinese(line), cut_all=False)
        test_text.append(' '.join(newline))
    read_file.close()
    return np.array(test_text)

x_text = load_fenci(options.test_dataset)
print("finish load.")
# Map data into vocabulary
vocab_path = os.path.join(options.checkpoint_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_text)))
print("\nTesting...\n")

# Testing
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(options.checkpoint_dir)
print("checkpoint_file========", checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=options.allow_soft_placement,
        log_device_placement=options.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))

        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = batch_iter(list(x_test), options.batch_size, 1, shuffle=False)

        # 存储模型预测结果
        all_predictions = []
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            for res in batch_predictions:
                all_predictions.append(res)

mood_list = ['恐慌', '悲伤', '惧怕', '愤怒', '无助', '焦虑']

write_file = open(options.prediction_file, 'w')
for one_pre in all_predictions:
    count = 0
    for i in range(len(one_pre)):
        if one_pre[i] == 1:
            write_file.write(mood_list[i])
            if count < 2:
                write_file.write(',')
                count += 1
    while count < 2:
        write_file.write(',')
        count += 1
    write_file.write('\n')
write_file.close()



