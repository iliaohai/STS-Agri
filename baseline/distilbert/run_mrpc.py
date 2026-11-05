import pandas as pd
import transformers as tfs
from torch import nn
from torch import optim
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from transformers import AdamW, BertModel, BertTokenizer

from baseline.distilbert.distilbert import DistilBERTClassificationModel
from baseline.unilm.unilm import UnilmClassificationModel
from baseline.ernie.ernie import ERNIEClassificationModel
from util.logutil import Logger
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差

# 记录日志
filename = "../../output/distilbert/run_mrpc"
time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
filepath = filename + "_" + time_stamp + ".log"
log = Logger(filepath, level='debug')

'''
    0.Create model
'''
bert_classifier_model = DistilBERTClassificationModel().cuda()

'''
    1.Load dateset
'''
train_set = pd.read_csv('../../dataset/mrpc/train.csv', header=0)  # train set
dev_set = pd.read_csv('../../dataset/mrpc/dev.csv', header=0)  # dev set
test_set = pd.read_csv('../../dataset/mrpc/test.csv', header=0)  # test set

# train
train_inputs = [train_set['#1 String'].values, train_set['#2 String'].values]  # train input
train_targets = train_set['Quality'].values  # train label
# dev
dev_inputs = [dev_set['#1 String'].values, dev_set['#2 String'].values]
dev_targets = dev_set['Quality'].values
# test
test_inputs = [test_set['#1 String'].values, test_set['#2 String'].values]
# test_targets = test_set['Quality'].values

'''
    2.Set batch
'''
batch_size = 32
batch_count = int(len(train_inputs[0]) / batch_size)
batch_train_inputs, batch_train_targets = [], []
sentences1, sentences2 = train_inputs[0], train_inputs[1]  # 第一个句子的所有列数组, 第二个句子的所有列数组
for i in range(batch_count):
    j = i * batch_size
    batch_sentences = []
    while j < (i+1) * batch_size:
        batch_sentences.append('[CLS]' + sentences1[j] + '[SEP]' + sentences2[j] + '[SEP]')
        j = j + 1
    batch_train_inputs.append(batch_sentences)
    batch_train_targets.append(train_targets[i*batch_size: (i+1)*batch_size])

'''
    3.训练模型
'''
epochs = 4
# lr = 0.01
print_every_batch = 1
optimizer = AdamW(bert_classifier_model.parameters(), lr=2e-5)

# 记录最好结果
max_acc, max_f1, max_p, max_r, best_epoch = 0.0, 0.0, 0.0, 0.0, 0
for epoch in range(epochs):
    print_avg_loss = 0
    bert_classifier_model.train()
    for i in range(batch_count):
        # 正向传播
        inputs = batch_train_inputs[i]  # inputs中有64条训练数据
        labels = batch_train_targets[i]  # labels中有64条标签
        optimizer.zero_grad()
        outputs = bert_classifier_model(inputs, labels)
        loss = outputs[0]
        print_avg_loss += loss.item()

        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bert_classifier_model.parameters(), 1.0)
        optimizer.step()

        if i % print_every_batch == (print_every_batch - 1):
            log.logger.info("epoch:%d" % epoch)
            log.logger.info("Batch: %d, Loss: %.4f" % ((i + 1), print_avg_loss / print_every_batch))
            print_avg_loss = 0

    '''
        4.评价模型
    '''
    bert_classifier_model.eval()
    total = len(dev_targets)
    y_pred = []
    y_true = []
    with torch.no_grad():
        for j in range(total):
            dev_strs1, dev_strs2 = dev_inputs[0], dev_inputs[1]
            dev_input = "[CLS]" + dev_strs1[j] + "[SEP]" + dev_strs2[j] + "[SEP]"
            outputs = bert_classifier_model([dev_input], dev_targets[j])  # 此处需要传数组，所以加一个[]
            loss = outputs[0]
            logits = outputs[1]
            y_pred.append(torch.argmax(logits, axis=1)[0].item())
            y_true.append(dev_targets[j])

    print(y_true)
    print(y_pred)
    f1score = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')  # 0.2222222222222222
    recall = recall_score(y_true, y_pred, average='macro')
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    log.logger.info("***************** Result ******************")
    log.logger.info("epoch:%d" % epoch)
    log.logger.info("Accuracy:%s" % acc)
    log.logger.info("F1:%s" % f1score)
    log.logger.info("precision:%s" % precision)
    log.logger.info("recall:%s" % recall)
    log.logger.info("mse:%s" % mse)
    log.logger.info("mae:%s" % mae)
    if (acc + f1score)/2 > (max_acc + max_f1) / 2:
        max_acc, max_f1, max_p, max_r, best_epoch = acc, f1score, precision, recall, epoch

        # 导出最佳预测结果
        result = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
        result.to_csv('../../output/distilbert/run_mrpc_result.csv', index=False)

    log.logger.info("max_accuracy:%s" % max_acc)
    log.logger.info("max_f1:%s" % max_f1)
    log.logger.info("max_p:%s" % max_p)
    log.logger.info("max_r:%s" % max_r)
    log.logger.info("best_epoch:%d" % best_epoch)
    log.logger.info("******************************************")

    # # 绘制roc图
    # Roc_figure.plot_ROC(y_true, y_pred, "(a) ROC based on HDFS dataset")

    # 导出预测结果
    # result = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    # result.to_csv('../../../Dataset/HDFS/result_BERT-Log.csv', index=False)
