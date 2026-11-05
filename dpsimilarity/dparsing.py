'''
    获取句子依存句法表示
    输入:
    batch_sentences: 批次数据
    tokenizer: bert的tokenizer
    bert_output: bert的get_sequence_output
    输出:
    out_put: 具有语义依存句法分析的文本表示, 可以和bert的输出拼接使用
'''

from ddparser import DDParser
import torch
from graph import get_arcs_and_head_in_wordpiece
from graph import pad_batch_graphs
from graph import get_adj_of_two_sent_in_ernie
from graph import transfor_head_id_for_ernie
import gnn
import paddle
import unicodedata
import regex

def getDependencyParsing(batch_sentences, tokenizer, bert_output):
    batch_adjacency_matrix = []  # 存放带批次邻接矩阵的数组
    batch_head_ids = []  # 存放带批次核心词索引的数组
    seq_length = bert_output.shape[1]  # 获取句子的长度,用于动态设置邻接矩阵的长度
    # print(seq_length)

    # 获取单个句子对
    for sentences in batch_sentences:
        str_a = sentences[sentences.find("[CLS]")+5: sentences.find("[SEP]")]
        str_b = sentences[len(str_a) + 10: len(sentences) - 5]

        # print(str_a)

        # 去掉字母的英标
        str_a = unicodedata.normalize('NFKD', str_a).encode('ascii', 'ignore').decode('utf-8')
        str_b = unicodedata.normalize('NFKD', str_b).encode('ascii', 'ignore').decode('utf-8')
        # print(str_a)

        # 去掉特殊字符
        str_a = regex.sub('[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+','', str_a)
        str_b = regex.sub('[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+','', str_b)

        # 获取句子的依存分析信息
        ddp = DDParser()
        ddp_res_a = ddp.parse(str_a)[0]
        ddp_res_b = ddp.parse(str_b)[0]

        # print(ddp.parse(str_a))

        # 直接将分词结果作为输入
        # ddp_res_a = ddp.parse_seg(str_a)[0]
        # ddp_res_b = ddp.parse_seg(str_b)[0]

        # 删除空格
        space_a = [i for i, j in enumerate(ddp_res_a['word']) if j == ' ']
        ddp_res_a['word'] = [i for num, i in enumerate(ddp_res_a['word']) if num not in space_a]
        ddp_res_a['head'] = [i for num, i in enumerate(ddp_res_a['head']) if num not in space_a]
        ddp_res_a['deprel'] = [i for num, i in enumerate(ddp_res_a['deprel']) if num not in space_a]
        space_b = [i for i, j in enumerate(ddp_res_b['word']) if j == ' ']
        ddp_res_b['word'] = [i for num, i in enumerate(ddp_res_b['word']) if num not in space_b]
        ddp_res_b['head'] = [i for num, i in enumerate(ddp_res_b['head']) if num not in space_b]
        ddp_res_b['deprel'] = [i for num, i in enumerate(ddp_res_b['deprel']) if num not in space_b]

        # 获取句子的token
        tokens_a = tokenizer.tokenize(str_a)
        tokens_b = tokenizer.tokenize(str_b)

        # 获取句子a的依存分析弧
        arcs_a, head_id_a = get_arcs_and_head_in_wordpiece(ddp_res_a, tokens_a)
        # 获取句子b的依存分析弧
        arcs_b, head_id_b = get_arcs_and_head_in_wordpiece(ddp_res_b, tokens_b)
        # 获取句子a和b组成的邻接矩阵
        adjacency_matrix = get_adj_of_two_sent_in_ernie(arcs_a, len(tokens_a), arcs_b, len(tokens_b))
        # 获取映射后核心词索引
        head_ids = transfor_head_id_for_ernie(head_id_a, len(tokens_a), head_id_b, len(tokens_b))

        # 存放批次
        batch_adjacency_matrix.append(adjacency_matrix)
        batch_head_ids.append(head_ids)

    # 使用GAT对bert输出和依存句法信息进行表示
    gat = gnn.GAT(input_size=768, hidden_size=100, output_size=100, dropout=0.0, alpha=0.1, heads=12, layer=4)
    sequence = bert_output.detach().cpu().numpy()  # 转paddle的输入格式
    batch_adjacency_matrix = pad_batch_graphs(batch_adjacency_matrix, seq_length)  # 长度填充
    adjmat = paddle.to_tensor(batch_adjacency_matrix, dtype='int32')  # 转paddle的输入格式
    gat_emb = gat.forward(sequence, adjmat)
    # 转torch的tensor
    gat_emb = torch.tensor(gat_emb.numpy()).cuda()
    # 将gat的输出与bert输出拼接
    out_put = torch.cat((bert_output, gat_emb), 2)

    return out_put

