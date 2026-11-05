import transformers as tfs
from torch import nn
import torch

from dpsimilarity import dparsing


class DPsimilarityRegressionModel(nn.Module):
    def __init__(self):
        super(DPsimilarityRegressionModel, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (tfs.DebertaV2Model, tfs.DebertaV2Tokenizer, '../pre-models/deberta-v3-base')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert = model_class.from_pretrained(pretrained_weights, num_labels=1)
        self.dense = nn.Linear(868, 1)  #bert默认的隐藏单元数是768， 输出单元是2，表示二分类

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True, max_length=66,  padding='longest')  # tokenize、add special token、pad
        input_ids = torch.tensor(batch_tokenized['input_ids']).cuda()
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda()
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        dps_output = dparsing.getDependencyParsing(batch_sentences, self.tokenizer, bert_output[0]).cuda()
        cls = dps_output[:, 0, :]
        # kernel, bias = compute_kernel_bias(numpy.array(cls), 256)  # 线性变换
        # cls = transform_and_normalize(numpy.array(cls), kernel=kernel, bias=bias)  # 标准化流
        # output = torch.tensor(cls)
        linear_output = self.dense(cls)
        return linear_output