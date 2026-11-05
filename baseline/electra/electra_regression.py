import transformers as tfs
from torch import nn
import torch


class ElectraRegressionModel(nn.Module):
    def __init__(self):
        super(ElectraRegressionModel, self).__init__()
        model_class, tokenizer_class, pretrained_weights = (tfs.ElectraForSequenceClassification, tfs.ElectraTokenizer, '../../pre-models/electra-base-discriminator')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.bert = model_class.from_pretrained(pretrained_weights, num_labels=1)

    def forward(self, batch_sentences, batch_labels):
        batch_tokenized = self.tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True, max_length=66,  padding='longest')  # tokenize、add special token、pad
        input_ids = torch.tensor(batch_tokenized['input_ids']).cuda()
        attention_mask = torch.tensor(batch_tokenized['attention_mask']).cuda()
        batch_labels = torch.tensor(batch_labels,dtype=torch.float).cuda()  # 回归任务设为float
        bert_output = self.bert(input_ids, attention_mask=attention_mask, labels=batch_labels)
        return bert_output
