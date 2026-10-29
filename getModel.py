# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:40:11 2026

@author: huawei
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
class getModel(nn.Module):
    def __init__(self,max_vocab,embbing_size=256,hidden_len=256):
        super(getModel,self).__init__()

        self.embedding = nn.Embedding(max_vocab,embbing_size,padding_idx=0)#词表目前 
        #可以这么理解，词矩阵维度是max_vocab * embbing_size 词向量矩阵 在idx=0 代表“pad词”（占位），这个位置的向量不更新，永远为0
        #idx= 1 代表 “unk"词，这个在定义词表的时候已经确定好了
        self.lstm = nn.LSTM(embbing_size,hidden_len,num_layers=1,batch_first=True,bidirectional=True)

        self.fc1 = nn.Linear(hidden_len*2,128)

        self.fc2 = nn.Linear(128,64)

        self.fc3 = nn.Linear(64,2)


    def forward(self,x,length):
        #x batch_size,seq_len
        x = self.embedding(x) # batch_size,seq_len,embedding_dim
        embed = x
        packed = nn.utils.rnn.pack_padded_sequence(x,length.to('cpu'),batch_first=True,enforce_sorted=False)# batch_size,seq_len,embedding_dim

        _,(h,_) = self.lstm(packed)#h: num_layers batch_size,hidden_size

        x = torch.cat([h[-1],h[-2]],dim=1)#batch_size hidden_size*2

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        logits = self.fc3(x)

        return logits,embed


if __name__ =='__main__':

    x = torch.tensor([[2, 3, 0, 0],      # "good bad <PAD> <PAD>"
                    [3, 0, 0, 0]])     # "bad <PAD> <PAD> <PAD>"
    lenght = torch.tensor([2,1])

    mymodel = getModel(30000)
    out,embed = mymodel(x,lenght)
    print(embed.shape)   # torch.Size([2, 4, embedding_dim])
    print(embed[0])

   