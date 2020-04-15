# coding:utf-8
import jieba
import numpy as np
import re
import pandas as pd

mood_list = ['恐慌', '悲伤', '惧怕', '愤怒', '无助', '焦虑']

def load_data(path, text_name, label_name):
    texts = pd.read_csv(path + text_name,header=None, names=["text"])
    labels = pd.read_csv(path + label_name,header=None, names=["label1","label2","label3"])
    texts_labels = pd.concat([texts, labels], axis=1)
    labels = []
    texts = []
    for row in texts_labels.iterrows():
        #examples_dic["text"].append(row[1]["text"])
        row_mood = [row[1]["label1"],row[1]["label2"], row[1]["label3"]]
        one_hot_mood = []
        for mood in mood_list:
            if mood in row_mood:
                one_hot_mood.append('1')
            else:
                one_hot_mood.append('0')
        labels.append(one_hot_mood)
        texts.append(row[1]["text"])
    return texts, labels

def shuffle(d):
    return np.random.permutation(d)

def shuffle2(d):
    len_ = len(d)
    times = 2
    for i in range(times):
        index = np.random.choice(len_, 2)
        d[index[0]],d[index[1]] = d[index[1]],d[index[0]]
    return d

def dropout(d, p=0.3):
    len_ = len(d)
    index = np.random.choice(len_, int(len_ * p))
    for i in index:
        d[i] = ' '
    return d


def clean(xx):
    xx2 = re.sub(r'\?', "", xx)
    xx1= xx2.split(' ')
    return xx1


def dataaugment(X, X_label):
    l = len(X)
    for i in range(l):
        item = clean(X[i])
        d1 = shuffle2(item)
        d11 = ' '.join(d1)
        d2 = dropout(item)
        d22 = ' '.join(d2)
        X.extend([d11,d22])
        X_label.extend([X_label[i], X_label[i]])
    return X, X_label

def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese = re.sub(pattern, '', file)
    return chinese

def FenCi(path, read_text_name, read_label_name, write_filename, is_train_file):

    write_file = open(write_filename, 'wb')
    write_file.write("label,ques\n".encode('utf-8'))
    texts, labels = load_data(path, read_text_name, read_label_name)
    ones_x = []
    ones_label = []
    zeros_x = []
    for i in range(len(texts)):
        newline = jieba.cut(find_chinese(texts[i]), cut_all=False)
        if ','.join(labels[i]) == "0,0,0,0,0,0":
            zeros_x.append(' '.join(newline))
        else:
            ones_x.append(' '.join(newline))
            ones_label.append(','.join(labels[i]))
    if is_train_file:
        #数据增强
        ones_x_p, ones_label_p = dataaugment(ones_x, ones_label)
    else:
        ones_x_p, ones_label_p = ones_x, ones_label
    for i in range(len(ones_x_p)):

        write_line = ones_label_p[i] + ',' + ones_x_p[i] + '\n'
        write_line = write_line.encode('utf-8', 'ignore')
        write_file.write(write_line)
    for text in zeros_x:
        write_line = '0,0,0,0,0,0,' + text + '\n'
        write_line = write_line.encode('utf-8', 'ignore')
        write_file.write(write_line)
    write_file.close()

from optparse import OptionParser

oparser = OptionParser()

oparser.add_option("-t", "--training-dir", dest="traning_dir", help="the file path for training.", default = "./dataset/training/")

oparser.add_option("-d", "--dev-dir", dest="dev_dir", help="the file path for dev.", default = "./dataset/dev/")

oparser.add_option("--fenci-train", dest="fenci_train", help="file path for precessed training file.", default = "./dataset/training/fenci_train_examples.csv")

oparser.add_option("--fenci-dev", dest="fenci_dev", help="file path for precessed dev file.", default = "./dataset/dev/fenci_dev_examples.csv")

(options, args) = oparser.parse_args()

TEXT_FILE = "texts.csv"
LABEL_FILE = "labels.csv"

fenci_train_file = "fenci_train_examples.csv"
fenci_dev_file = "fenci_dev_examples.csv"
FenCi(options.traning_dir, TEXT_FILE, LABEL_FILE, options.fenci_train, is_train_file=True)
FenCi(options.dev_dir, TEXT_FILE, LABEL_FILE, options.fenci_dev, is_train_file=False)
