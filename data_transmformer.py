#encoding:utf-8
import os
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def pkl_read(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

class DataTransformer(object):
    def __init__(self,
                 # vocab_path,
                 stroke2word_path,
                 embedding_path):
        # self.vocab_path = vocab_path
        self.embedding_path = embedding_path
        self.stroke2word_path = stroke2word_path
        self.reset()

    def reset(self):
        # if os.path.isfile(self.vocab_path):
        #     self.vocab = pkl_read(self.vocab_path)
        # else:
        #     raise FileNotFoundError("vocab file not found")
        self.stroke2word = pkl_read(str(self.stroke2word_path))
        self.load_embedding()


    # 加载词向量矩阵
    def load_embedding(self, ):
        print(" load emebedding weights")
        self.embeddings_index = {}
        self.words = []
        self.vectors = []
        f = open(str(self.embedding_path), 'r',encoding = 'utf8')
        for line in f:
            values = line.split(' ')
            try:
                word  = self.stroke2word[values[0]]
                self.words.append(word)
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
                self.vectors.append(coefs)
            except:
                print("Error on ", values[:2])
        f.close()
        self.vectors = np.vstack(self.vectors)
        print(f'Total {len(self.embeddings_index)} word vectors.')

    # 计算相似度
    def get_similar_words(self, word, w_num=10):
        if word not in self.embeddings_index:
            raise ValueError('%d not in vocab')
        current_vector = self.embeddings_index[word]
        result = cosine_similarity(current_vector.reshape(1, -1), self.vectors)
        result = np.array(result).reshape(self.vectors.shape[0], )
        idxs = np.argsort(result)[::-1][:w_num]
        print("<<<" * 7)
        print("{} 前 {} 个最相似的词".format(word,w_num))
        for i in idxs:
            print(f"{self.words[i]} : {result[i]:.3f}",end=", ")
        print(">>>" * 7)
    
    

