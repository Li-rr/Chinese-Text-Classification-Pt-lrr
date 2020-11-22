# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
from gensim.models import Word2Vec
from data_transmformer import DataTransformer

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[
                            1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx,
                     word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    if ues_word:
        def tokenizer(x): return x.split(' ')  # 以空格隔开，word-level
    else:
        def tokenizer(x): return [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer,
                            max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index *
                                   self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index *
                                   self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def get_wiki_word2vec(word_to_id, pretrain_dir,
                      emd_dim=300,
                      filename_trimmed_dir='./THUCNews/data/embedding_wiki_w2c'):
    # 初始化一个随机矩阵
    embeddings = np.random.rand(len(word_to_id), emd_dim)
    w2c_model = Word2Vec.load(pretrain_dir)
    vocab = w2c_model.wv.vocab
    # for k,v in vocab.items():
    #     print(k,v)
    # print(embeddings[7])
    oov = []
    cnt = 0
    for i, (k, v) in enumerate(word_to_id.items()):
        # if i ==10:
        #     break
        if k in vocab:

            emb = w2c_model.wv[k]  # 获取emb
            # print(emb.shape,type(emb),emb.dtype)
            embeddings[v] = emb
            cnt += 1
            # print(k,v)
        else:
            oov.append(k)
    # print(embeddings[7])
    # print(oov)
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
    # 读取wiki w2c词向量完成，共获取 4576 个词向量
    print("读取wiki w2c词向量完成，共获取 {} 个词向量".format(cnt))


def get_wiki_cw2vec(word_to_id, pretrain_dir,
                    emd_dim=300,
                    filename_trimmed_dir='./THUCNews/data/embedding_wiki_cw2c'
                    ):
    data_tran = DataTransformer(
        stroke2word_path=os.path.join(pretrain_dir, "processed/idx2word.pkl"),
        embedding_path=os.path.join(pretrain_dir, "embedding_300d/gensim_word_vector.bin")
        )
    
    print(len(data_tran.stroke2word))
    print(len(data_tran.stroke2word.keys()))
    data_tran.get_similar_words("男人")
    print("===")
    embeddings = np.random.rand(len(word_to_id), emd_dim)
    oov = []
    cnt = 0
    for i, (k, v) in enumerate(word_to_id.items()):
        if k in data_tran.embeddings_index:
            emb = data_tran.embeddings_index[k]  # 获取emb
            embeddings[v] = emb
            cnt += 1
        else:
            oov.append(k)
    print(oov)
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
        # 读取wiki w2c词向量完成，共获取 4006 个词向量
    print("读取wiki w2c词向量完成，共获取 {} 个词向量".format(cnt))   


def get_sogo_w2c(word_to_id, pretrain_dir, emb_dim=300):
    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    n_oov = []
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        if i == 3:
            break
        lin = line.strip().split(" ")
        # print(lin)
        if lin[0] in word_to_id:  # 遍历每个字
            # print("--?",lin[0])
            if "叔" == lin[0]:
                print("fuck ")
            n_oov.append(lin[0])
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
    # print(embeddings[word_to_id['叔']])
    tmp = set(list(word_to_id.keys()))
    tmp2 = set(n_oov)
    # # print(type(tmp),tmp)
    tmp3 = tmp-tmp2
    print(tmp3)
    if "叔" in tmp3:
        print("aa")


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    wiki_w2c = "/home/stu/Documents/dataset/wiki/w2c_300d/wiki.model"
    wiki_cw2c = "/home/stu/Documents/dataset/wiki/cw2vec_300d"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        def tokenizer(x): return [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(
            train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    # get_sogo_w2c(word_to_id=word_to_id,pretrain_dir=pretrain_dir)
    # get_wiki_word2vec(
    #     word_to_id=word_to_id,
    #     pretrain_dir=wiki_w2c
    # )
    get_wiki_cw2vec(word_to_id=word_to_id, pretrain_dir=wiki_cw2c)
