import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import numpy as np

from models.capsule import CapsNet


class Config:
    def __init__(self, dataset, embedding) -> None:
        self.model_name = "CapsuleNet"
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + \
            self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.m_train = True
class Model(nn.Module):
    def __init__(self, config):
        super(Model,self).__init__()
        print("vocab size",config.n_vocab)
        print("embed",config.embed)
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(
                config.n_vocab, config.embed)
        # Embedding层的输出[128, 32, 300]
        
        self.capsule = CapsNet(300, 23*16,num_class=config.num_classes)
        self.linear = nn.Linear(in_features=140, out_features=9)
        self.batch = nn.BatchNorm1d(num_features=140)
        self.m_train = config.m_train

    def forward(self, x):
        embed = self.embedding(x[0])
        print("embed's shape", embed.shape)
        caps_output = self.capsule(embed)
        output = self.batch(caps_output)
        output = F.dropout(output, p=0.5, training=self.m_train)
        output = self.linear(output)
        return output
