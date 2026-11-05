'''
    使用transformers进行文本分类
    Date: 20220124
    Author: Hai Liao
    Version: 1.0
'''
import pandas as pd
import transformers as tfs
from torch import nn
from torch import optim
import torch
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from baseline.electra.electra_regression import ElectraRegressionModel
from baseline.ernie.ernie_regression import ErnieRegressionModel
from baseline.spanbert.spanbert_regression import SpanBertRegressionModel


import numpy as np
from transformers import AdamW
# 皮尔逊相关系数
from scipy.stats import pearsonr
from util.logutil import Logger
from util.logutil import Logger
from scipy.stats import spearmanr
from scipy.stats.stats import kendalltau
import time

# 记录日志
filename = "../../output/ernie/run_biossess"
time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
filepath = filename + "_" + time_stamp + ".log"
log = Logger(filepath, level='debug')

'''
    1.加载数据集
'''
train_set = pd.read_csv('../../dataset/BIOSSES/train.tsv', sep='\t', header=0, )  # 训练集
dev_set = pd.read_csv('../../dataset/BIOSSES/dev.tsv', sep='\t', header=0)  # 开发集
test_set = pd.read_csv('../../dataset/BIOSSES/test.tsv', sep='\t', header=0)  # 测试集


# 训练集
train_inputs = [train_set['sentence1'].values, train_set['sentence2'].values]  # 训练集输入
train_targets = train_set['score'].values  # 训练集标签
# 开发集
dev_inputs = [dev_set['sentence1'].values, dev_set['sentence2'].values]
dev_targets = dev_set['score'].values
# 测试集
test_inputs = [test_set['sentence1'].values, test_set['sentence2'].values]
test_targets = test_set['score'].values

bert_classifier_model = ErnieRegressionModel().cuda()


'''
    2.训练数据集分批
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
epochs = 500
# lr = 0.01
print_every_batch = 1
# optimizer = optim.SGD(bert_classifier_model.parameters(), lr=lr, momentum=0.9)
optimizer = AdamW(bert_classifier_model.parameters(), lr=2e-5)

# 记录最好结果
max_pearson, max_pvalue, best_epoch = 0.0, 0.0, 0
max_mae, max_mse = 0.0, 0.0
max_spearson = 0.0
max_spvalue = 0.0
max_kend = 0.0
max_kpvalue = 0.0
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
            print("epoch:", epoch)
            print("Batch: %d, Loss: %.4f" % ((i + 1), print_avg_loss / print_every_batch))
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
            dev_strs1, dev_strs2 = test_inputs[0], test_inputs[1]
            dev_input = "[CLS]" + dev_strs1[j] + "[SEP]" + dev_strs2[j] + "[SEP]"
            outputs = bert_classifier_model([dev_input], test_targets[j])  # 此处需要传数组，所以加一个[]
            loss = outputs[0]
            logits = outputs[1]
            y_pred.append(np.squeeze(logits).item())
            y_true.append(test_targets[j])

    print(y_pred)
    print(y_true)
    pearson, pvalue = pearsonr(y_true, y_pred)
    spearson, spvalue = spearmanr(y_true, y_pred)
    kend, kpvalue = kendalltau(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    log.logger.info("***************** Result ******************")
    log.logger.info("epoch:%d" % epoch)
    log.logger.info("Pearson correlation:%s" % pearson)
    log.logger.info("p-value:%s" % pvalue)
    log.logger.info("spearmanr correlation:%s" % spearson)
    log.logger.info("spearmanr p-value:%s" % spvalue)
    log.logger.info("kend correlation:%s" % kend)
    log.logger.info("kend p-value:%s" % kpvalue)
    log.logger.info("mse:%s" % mse)
    log.logger.info("mae:%s" % mae)

    if pearson > max_pearson:
        max_pearson = pearson
        max_pvalue = pvalue
        best_epoch = epoch
        max_spearson = spearson
        max_spvalue = spvalue
        max_kend = kend
        max_kpvalue = kpvalue
        max_mae = mae
        max_mse = mse

     # 进行测试集评估
    log.logger.info("max_pearson:%s" % max_pearson)
    log.logger.info("max_pvalue:%s" % max_pvalue)
    log.logger.info("max_spearmanr:%s" % max_spearson)
    log.logger.info("max_spearmanr_pvalue:%s" % max_spvalue)
    log.logger.info("max_kend:%s" % max_kend)
    log.logger.info("max_k_pvalue:%s" % max_kpvalue)
    log.logger.info("best_epoch:%d" % best_epoch)
    log.logger.info("max_mae:%s" % max_mae)
    log.logger.info("max_mse:%s" % max_mse)


