```
本文档模板仅供参考，参赛选手可根据实际需要增删和修改。
```

# 未来杯“战疫”特别赛事 作品

* 轮次：第{1}轮
* 战队编号：{2681}
* 战队名称: {午后红茶}
* 战队成员：{杨志贤}

## 概述

在数据预处理阶段使用jieba分词对文本数据进行分词；

在模型搭建和预测阶段，使用tensorflow深度学习架构，实现了textcnn模型

## 系统要求

### 硬件环境要求

* CPU: Intel(R) Core(TM) i7-6900K CPU @ 3.20GHz
* GPU: NVIDIA TITAN RTX
* 内存: 12G
* 硬盘: 无
* 其他: 无

### 软件环境要求

* 操作系统: {Ubuntu} {16.04.3 LTS}

如有特殊编译/安装步骤，请列明。

### 数据集

无其他数据集

## 数据预处理

### 方法概述
调用jieba分词库对文本进行分词

### 操作步骤
分词的实现在 fenci_jieba.py 中，参数 --training-dir 指定训练数据集的文件路径，--dev-dir 指定验证数据集的文件路径，--fenci-train 指定训练数据集分词处理之后的文件名，--fenci-dev 指定验证数据集分词处理之后的文件名。
```python
python fenci_jieba.py \
--training-dir="./dataset/training/" \
--dev-dir="./dataset/dev/" \
--fenci-train="./dataset/training/fenci_train_examples.csv" \
--fenci-dev="./dataset/dev/fenci_dev_examples.csv"

```
### 模型

训练后的模型存储地址："./model/"

模型文件大小：118M


## 训练

### 训练方法概述

使用tensorflow框架搭建了textcnn模型，模型代码在 text_cnn.py 文件中，训练程序在 train_textcnn.py 文件中，文件中设定了默认的超参数如下：

```python
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
```



### 训练操作步骤
参数--training-dataset指定训练集文件，注意，需要传入分词之后的文件，--model指定模型存储位置
```python
python train_textcnn.py \
--training-dataset="./dataset/training/fenci_train_examples.csv" \
--model="./model/"

```

### 训练结果保存与获取

训练的模型存储在指定的--model参数文件夹下

## 测试

### 方法概述
预测的代码实现在 main_predict.py 文件中，即加载模型，并做出预测

### 操作步骤
参数--test-dataset指定测试数据集， --model指定模型存储位置， --prediction-file 指定预测结果输出的文件名
```python
python main_predict.py \
--test-dataset="./dataset/test/texts.csv" \
--model="./model/" \
--prediction-file="./dataset/test/submit.csv"
```

## 其他
该项目上传的文件结构如下，最终预测的结果在submit.csv文件中。

```
├── dataset
│   ├── training
│	│	├── texts.csv
│	│	├── labels.csv
│	│	└── fenci_train_examples.csv
│   ├── dev
│	│	├── texts.csv
│	│	├── labels.csv
│	│	└── fenci_dev_examples.csv
│   └── test
│	│	├── texts.csv
│	│	└── submit.csv
├── mdoel
│   ├── checkpoint
│   ├── model-24300.data-00000-of-00001
│   ├── model-24300.index
│   ├── model-24300.meta
│   └── vocab
├── data_helpers.py
├── fenci_jieba.py
├── main_predict.py
├── README.md
├── requirements.txt
├── text_cnn.py
└── train_textcnn.py
```