import transformers as tfs
from torch import nn
import torch


class T5RegressionModel(nn.Module):
    def __init__(self):
        super(T5RegressionModel, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (tfs.T5EncoderModel, tfs.T5Tokenizer, '../../pre-models/t5-base')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert = model_class.from_pretrained(pretrained_weights, num_labels=1)
        self.dense = nn.Linear(768, 1)  #bert默认的隐藏单元数是768， 输出单元是2，表示二分类

    def forward(self, batch_sentences):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True, max_length=512,  padding='longest')  # tokenize、add special token、pad
        input_ids = torch.tensor(batch_tokenized['input_ids']).cuda()
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda()
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :].cuda()   # 提取[CLS]对应的隐藏状态
        linear_output = self.dense(bert_cls_hidden_state).cuda()
        return linear_output
