# coding=utf-8
import torch
device='cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np

from optparse import OptionParser

oparser = OptionParser()

oparser.add_option("-m", "--model", dest="checkpoint_dir", help="Checkpoint directory from trained model", default = "./model/")

oparser.add_option("-t", "--training-dataset", dest="traning_dataset", help="the file path for training file.", default = "./dataset/training/fenci_train_examples.csv")

oparser.add_option("-d", "--dev-dataset", dest="dev_dataset", help="the file path for dev file.", default = "./dataset/dev/fenci_dev_examples.csv")

(options, args) = oparser.parse_args()

def load_data_and_labels(data_file):
    train_examples = open(data_file, "rb")
    train_examples.readline()
    x = []
    y = []
    max_len = 0
    for line in train_examples:
        line = line.decode('utf-8', 'ignore')
        line_split = line.strip('\n').split(',')
        x.append(''.join(line_split[-1].split(' ')))
        max_len = max(max_len, len(x[-1]))
        y.append([int(i) for i in line_split[:-1]])
    return np.array(x), np.array(y), max_len
x_train_text, y_label, max_train_length = load_data_and_labels(options.traning_dataset)
x_dev_text, y_dev, max_dev_length = load_data_and_labels(options.dev_dataset)
print(x_train_text[:10])
print(y_label[:10])

from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

MAX_LEN=150
batch_size = 32
epochs = 20

input_ids = [tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in x_train_text]
dev_input_ids=[tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in x_dev_text]

from keras.preprocessing.sequence import pad_sequences
print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                          value=0, truncating="post", padding="post")

dev_input_ids = pad_sequences(dev_input_ids, maxlen=MAX_LEN, dtype="long",
                          value=0, truncating="post", padding="post")

# Create attention masks
attention_masks = []

for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)

dev_attention_masks = []

for sent in dev_input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    dev_attention_masks.append(att_mask)

"""创建数据集和dataloader"""

train_inputs = torch.tensor(input_ids)
validation_inputs = torch.tensor(dev_input_ids)

train_labels = torch.tensor(y_label)
validation_labels = torch.tensor(y_dev)

train_masks = torch.tensor(attention_masks)
validation_masks = torch.tensor(dev_attention_masks)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

"""创建模型"""
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel

class BertForMultiLable(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultiLable, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def unfreeze(self,start_layer,end_layer):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())
        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b
        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)
        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))
        set_trainable(self.bert, False)
        for i in range(start_layer, end_layer+1):
            set_trainable(self.bert.encoder.layer[i], True)

model = BertForMultiLable.from_pretrained(
    "bert-base-chinese",
    num_labels = 6,
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

model.cuda()

"""设置optimizer和scheduler"""
from transformers import AdamW, BertConfig
optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )

from transformers import get_linear_schedule_with_warmup

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer,
                      num_warmup_steps = 0, # Default value in run_glue.py
                      num_training_steps = total_steps)
"""写一些函数帮助训练"""

import numpy as np

eps = 1e-3


def flat_accuracy(preds, labels):
    pred_flat = preds.view(-1, 6)
    labels_flat = labels.view(-1, 6)
    pred_flat = pred_flat.to('cpu').numpy()
    labels_flat = labels_flat.to('cpu').numpy()

    right_num = 0.
    for i in range(len(pred_flat)):
        if (pred_flat[i] == labels_flat[i]).all():
            right_num += 1.
    return right_num / len(pred_flat)


def pr_re_f1(preds, labels):
    pred_flat = preds.view(-1, 6)
    labels_flat = labels.view(-1, 6)
    pred_flat = pred_flat.to('cpu').numpy()
    labels_flat = labels_flat.to('cpu').numpy()
    TP, sum_y, sum_y_hat = 0, 0, 0
    for i in range(len(pred_flat)):
        for j in range(len(pred_flat[i])):
            if abs(pred_flat[i][j] - labels_flat[i][j]) < eps and abs(pred_flat[i][j] - 1) < eps:
                TP += 1
            if abs(labels_flat[i][j] - 1) < eps:
                sum_y_hat += 1
            if abs(pred_flat[i][j] - 1) < eps:
                sum_y += 1
    if sum_y == 0:
        P = 0.
    else:
        P = float(TP) / float(sum_y)
    if sum_y_hat == 0:
        R = 0.
    else:
        R = float(TP) / float(sum_y_hat)
    if P == 0. or R == 0.:
        F1 = 0.
    else:
        F1 = 2 * P * R / (P + R)
    return P, R, F1


import time
import datetime


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss


__call__ = ['CrossEntropy','BCEWithLogLoss']

class CrossEntropy(object):
    def __init__(self):
        self.loss_f = CrossEntropyLoss()

    def __call__(self, output, target):
        loss = self.loss_f(input=output, target=target)
        return loss

class BCEWithLogLoss(object):
    def __init__(self):
        self.loss_fn = BCEWithLogitsLoss()

    def __call__(self,output,target):
        output = output.float()
        target = target.float()
        loss = self.loss_fn(input = output,target = target)
        return loss


import random

seed_val = 10

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

loss_values = []
best_F1 = -1
criterion = BCEWithLogLoss()

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()

    total_loss = 0

    model.train()
    for step, batch in enumerate(train_dataloader):
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        b_labels = batch[2].to(device)
        model.zero_grad()

        logits = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask)
        loss = criterion(output=logits, target=b_labels)

        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}, loss: {}.'.format(step, len(train_dataloader), elapsed,
                                                                                  loss.item()))
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)

    loss_values.append(avg_train_loss)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()

    eval_loss, eval_accuracy, eval_precision, eval_recall, eval_f1 = 0, 0, 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask)

        rounded_preds = torch.round(torch.sigmoid(logits))
        tmp_eval_accuracy = flat_accuracy(rounded_preds, b_labels)
        tmp_eval_precision, tmp_eval_recall, tmp_eval_f1 = pr_re_f1(rounded_preds, b_labels)
        eval_accuracy += tmp_eval_accuracy
        eval_precision += tmp_eval_precision
        eval_recall += tmp_eval_recall
        eval_f1 += tmp_eval_f1
        nb_eval_steps += 1

    print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
    print("  Precisoin: {0:.2f}".format(eval_precision / nb_eval_steps))
    print("  Recall: {0:.2f}".format(eval_recall / nb_eval_steps))
    print("  F1: {0:.2f}".format(eval_f1 / nb_eval_steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))
    if eval_f1 / nb_eval_steps > best_F1:
        print("Saving model!")
        best_F1 = eval_f1 / nb_eval_steps
        torch.save(model.state_dict(), options.checkpoint_dir + '20FutureAI-Bert-based-model2.pt')