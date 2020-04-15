# coding=utf-8
import torch
from trainer import BertForMultiLable
device='cuda' if torch.cuda.is_available() else 'cpu'

from optparse import OptionParser

oparser = OptionParser()

oparser.add_option("-d", "--test-dataset", dest="test_dataset", help="the file path for test file.", default = "./dataset/test/texts.csv")

oparser.add_option("-m", "--model", dest="checkpoint_dir", help="Checkpoint directory from trained model", default = "./model/")

oparser.add_option("-o", "--prediction-file", dest="prediction_file", help="the file path for predict file", default = "./dataset/test/submit.csv")

(options, args) = oparser.parse_args()

import numpy as np

def load_data_and_labels(data_file):
    train_examples = open(data_file, "rb")
    x = []
    max_len = 0
    for line in train_examples:
        line = line.decode('utf-8', 'ignore')
        line_split = line.strip('\n').split(',')
        x.append(line_split[-1])
        max_len = max(max_len, len(x[-1]))
    return np.array(x), max_len
x_test_text, max_test_length = load_data_and_labels(options.test_dataset)
print(x_test_text.shape)

from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

MAX_LEN=150
batch_size = 32

test_input_ids=[tokenizer.encode(sent,add_special_tokens=True,max_length=MAX_LEN) for sent in x_test_text]

print('Original: ', x_test_text[0])
print('Token IDs:', test_input_ids[0])

from keras.preprocessing.sequence import pad_sequences

test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long",
                          value=0, truncating="post", padding="post")

# Create attention masks

test_attention_masks = []

for sent in test_input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    test_attention_masks.append(att_mask)
print(len(test_attention_masks))

"""创建数据集和dataloader"""

test_inputs=torch.tensor(test_input_ids)

test_masks=torch.tensor(test_attention_masks)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# Create the DataLoader for our test set.
test_data = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

model = BertForMultiLable.from_pretrained(
    "bert-base-chinese",
    num_labels = 6,
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)


model.cuda()

def write_pre(preds, write_file):
    mood_list = ['恐慌', '悲伤', '惧怕', '愤怒', '无助', '焦虑']
    rounded_preds = torch.round(torch.sigmoid(logits)).view(-1,6).to('cpu').numpy()
    for one_pre in rounded_preds:
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


"""加载模型"""

model.load_state_dict(torch.load(options.checkpoint_dir + '20FutureAI-Bert-based-model.pt'))

"""测试集进行测试"""
import time

t0 = time.time()
model.eval()

write_file = open(options.prediction_file, 'w', encoding='utf-8')

for step, batch in enumerate(test_dataloader):
    if step % 40 == 0 and not step == 0:
        print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))

    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask = batch

    with torch.no_grad():
        logits = model(b_input_ids,
                       token_type_ids=None,
                       attention_mask=b_input_mask)

    write_pre(logits, write_file)
write_file.close()
