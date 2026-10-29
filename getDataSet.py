
from torch.utils.data import Dataset
import torch
import re                                          # 正则表达式，用来清洗文本中的标点
from collections import Counter


def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()


def build_vocab(texts,min_freq=2,max_vocab = 30000):
    
    counter = Counter()
   
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
  # 构建词表
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    idx2word = {0: '<PAD>', 1: '<UNK>'}


    for word, count in counter.most_common(max_vocab):
        if len(word2idx) >= max_vocab:   # 总数限制 包括pad 和 unk
            break
        if count >= min_freq and word not in word2idx:
            idx = len(word2idx)
            word2idx[word] = idx
            idx2word[idx] = word

    return word2idx,idx2word



#目的是通过getitem函数返回  ids，label  length

class getDataSet(Dataset):
    def __init__(self,inputdata,word2idx,idx2word,max_seq_len = 250):
        self.texts = inputdata['text']
        self.labels = inputdata['label']
        self.max_seq_len = max_seq_len
        self.word2idx = word2idx   # 由外部传入，train/test 共用同一份
        self.idx2word = idx2word

      



        
    def __len__(self):
        return len(self.labels)

    
    def __getitem__(self,idx):
        # text =  torch.tensor(self.texts[idx])
        # label = torch.tensor(self.labels[idx])

        text =  self.texts[idx]
        label = self.labels[idx]

        idxs = [self.word2idx.get(token,self.word2idx['<UNK>']) for token in tokenize(text)]
        idxs = idxs[:self.max_seq_len]

        length = len(idxs)

        if length < self.max_seq_len:
            idxs = idxs + [self.word2idx['<PAD>']] * (self.max_seq_len - length)

        idxs = torch.tensor(idxs,dtype=torch.long)
        length = torch.tensor(length,dtype=torch.long)
        label = torch.tensor(label,dtype=torch.long)

        return idxs,length,label
         







# if __name__ == '__main__':

        




