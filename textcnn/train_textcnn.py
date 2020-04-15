#coding:utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from text_cnn import TextCNN
from data_helpers import load_data_and_labels, batch_iter
from optparse import OptionParser

oparser = OptionParser()

oparser.add_option("-m", "--model", dest="checkpoint_dir", help="Checkpoint directory from trained model", default = "./model/")

oparser.add_option("-t", "--training-dataset", dest="traning_dataset", help="the file path for training file.", default = "./dataset/training/fenci_train_examples.csv")

oparser.add_option("-d", "--dev-dataset", dest="dev_dataset", help="the file path for dev file.", default = "./dataset/dev/fenci_dev_examples.csv")

oparser.add_option("-b", "--batch_size", dest="batch_size", help="Batch Size (default: 128)", default = 128)

oparser.add_option("-a", "--allow_soft_placement", dest="allow_soft_placement", help="Allow device soft device placement", default = True)

oparser.add_option("-l", "--log_device_placement", dest="log_device_placement", help="Log placement of ops on devices", default = False)

(options, args) = oparser.parse_args()

#默认的超参数
EMBEDDING_DIM = 300 #词嵌入维度
FILTER_SIZE = "3,4,5" #cnn的filter size
NUM_FILTERS = 128 #number of filters
DROPOUT_KEEP = 0.5 #dropout的保留几率
L2_REG = 0.0 #l2正则化的参数
LEARNING_RATE = 1e-4 #学习率
NUM_EPOCHS = 50
EVAL_STEP = 100 #多少个step验证一次
CHECKPOINT_STEP = 100 #多少个step存一次模型
NUM_CHECKPOINT = 5 #最大保留的模型数

print("Loading data...")
x_train_text, y_label, max_document_length = load_data_and_labels(options.traning_dataset)
x_dev_text, y_dev, _ = load_data_and_labels(options.dev_dataset)
print(" =========Loading data x_text==============  ")
print(max_document_length)

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x_train_idx = np.array(list(vocab_processor.fit_transform(x_train_text)))

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y_label)))
x_train = np.array(list(vocab_processor.fit_transform(x_train_text)))[shuffle_indices]
y_train = y_label[shuffle_indices]

x_dev = np.array(list(vocab_processor.fit_transform(x_dev_text)))


print(x_train.shape)

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=options.allow_soft_placement,
      log_device_placement=options.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=EMBEDDING_DIM,
            filter_sizes=list(map(int, FILTER_SIZE.split(","))),
            num_filters=NUM_FILTERS,
            l2_reg_lambda=L2_REG)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = options.checkpoint_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_prefix = os.path.join(out_dir, "model")

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=NUM_CHECKPOINT)

        vocab_processor.save(os.path.join(out_dir, "vocab"))

        sess.run(tf.global_variables_initializer())

        # 训练模型
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: DROPOUT_KEEP
            }
            _, step, loss, F1, T, one_hot_prediction, input_y, accuracy, scores, predictions, input_y= sess.run(
                [train_op, global_step, cnn.loss, cnn.F1, cnn.T, cnn.one_hot_prediction, cnn.input_y, cnn.accuracy, cnn.scores, cnn.predictions, cnn.input_y],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, F1 {:g}, acc {:g}".format(time_str, step, loss, F1, accuracy))

        def dev_step(x_batch, y_batch):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, loss, F1, T, one_hot_prediction, input_y, accuracy, scores, predictions, input_y = sess.run(
                [global_step, cnn.loss, cnn.F1, cnn.T, cnn.one_hot_prediction, cnn.input_y, cnn.accuracy, cnn.scores, cnn.predictions, cnn.input_y],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, F1 {:g}, acc {:g}".format(time_str, step, loss, F1, accuracy))

        batches = batch_iter(list(zip(x_train, y_train)), options.batch_size, NUM_EPOCHS)
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % EVAL_STEP == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev)
                print("")
            if current_step % CHECKPOINT_STEP == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


