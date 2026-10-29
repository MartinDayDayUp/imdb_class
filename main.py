
from datasets import Dataset,load_dataset
from getDataSet import getDataSet,build_vocab
from torch.utils.data import DataLoader
import torch.nn as nn
from getModel import getModel
import torch
from disp import plot_and_save
from init_config import getConfig

import os


config = getConfig('./config.yaml')
batch_size = config.batch_size
epochs = config.epochs
max_vocab = config.max_vocab

check_path = config.checkpoint_path
data_dir = config.data_path




# data_dir = "./imdb_data/"

if os.path.exists(f"./{data_dir}/imdb-train.arrow"):
    print('data from local')
    train_dataset = Dataset.from_file(
        f"{data_dir}/imdb-train.arrow"
    )
    val_dataset = Dataset.from_file(
        f"{data_dir}/imdb-test.arrow"
    )
else:
    print('data from hugging face ')
    tmp_dataset = load_dataset(
        "stanfordnlp/imdb", 
        cache_dir="./imdb_data"
    )
    train_dataset = tmp_dataset['train']
    val_dataset = tmp_dataset['test']




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

word2idx,idx2word = build_vocab(train_dataset['text'],max_vocab=max_vocab)


train_data  = getDataSet(train_dataset,word2idx,idx2word)
val_data = getDataSet(val_dataset,word2idx,idx2word)

idxs,lengh,label = train_data[0]

text = [idx2word.get(id,'1') for id in idxs.numpy()]

# print(text)
# print(idxs,label)

train_ld = DataLoader(train_data,batch_size =batch_size,shuffle=True)
val_ld = DataLoader(train_data,batch_size =batch_size,shuffle=False)

# idxs,lengths,labels = next(iter(train_ld))
# embedding = nn.Embedding(max_vocab,300,padding_idx=0)

# embedding.weight.shape()
# x = embedding(idxs)




def train_oneEp(model,criterion,optimizer):
    model.train()
    total_loss = 0
    average_loss = 0
    correct = 0
    for idxs,lengths,labels in train_ld:
        idxs = idxs.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        logits,_ = model(idxs,lengths)#batch_size*num_classes
        loss = criterion(logits,labels)

        preds = torch.argmax(logits,1)#batch_size  每个值都是最大概率的索引 0 ~numclasses
        correct += (preds == labels).sum().item()

        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数

        total_loss += loss.item()*len(labels)
    average_loss = total_loss/len(train_ld.dataset)  
    acc = correct/len(train_ld.dataset) 
    return average_loss,acc


def val_oneEp(model,criterion):    

    model.eval()
    total_loss = 0
    average_loss = 0
    correct = 0

    with torch.no_grad():
        for idxs,lengths,labels in val_ld:
            idxs = idxs.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            logits,_ = model(idxs,lengths)
            loss = criterion(logits,labels)

            preds = torch.argmax(logits,1)#batch_size  每个值都是最大概率的索引 0 ~numclasses
            correct += (preds == labels).sum().item()
            
            total_loss += loss.item()*len(labels)
        average_loss = total_loss/len(val_ld.dataset)  
        acc = correct/len(val_ld.dataset) 

    return average_loss,acc



def run_train():
    lstmModel = getModel(max_vocab).to(device)

    criterion = nn.CrossEntropyLoss()                  # 交叉熵损失，适合分类任务
    optimizer = torch.optim.Adam(lstmModel.parameters(), lr=1e-4)  

    train_result ={'loss':[],
                   'acc':[]}
    val_result ={'loss':[],
                   'acc':[]}

    for ep in range(0,epochs):
        train_loss,train_acc = train_oneEp(lstmModel,criterion,optimizer)
        val_loss,val_acc = val_oneEp(lstmModel,criterion)
        train_result['loss'].append(train_loss)
        train_result['acc'].append(train_acc)

        val_result['loss'].append(val_loss)
        val_result['acc'].append(val_acc)

       
        print(f'epcId={ep},train_loss = {train_loss:.4f} train_acc = {train_acc:.4f},val_loss = {val_loss:.4f} val_acc = {val_acc:.4f}')


    save_check_point = {
        'train_result':train_result,
        'val_result':val_result,
        'state_dict':lstmModel.state_dict()
    }
    torch.save(save_check_point,check_path)
    plot_and_save(train_result, val_result, './out/curve.png')


def run_test():

    if os.path.exists(check_path):
        print('no model load')
        return
    check_obj = torch.load(check_path)
    lstmModel = getModel(max_vocab).to(device)
    lstmModel.load_state_dict(check_obj['state_dict'])


    test_list = []
    test_list.append('i love this movie it is very good')
    test_list.append('Absolutely brilliant from start to finish. The acting was superb and the story kept me on the edge of my seat. Highly recommended!')
    test_list.append('i hate this movie')
    test_list.append('Boring, predictable, and way too long. I checked my watch at least five times. Save your money.')
    test_label = []
    test_label.append(1)
    test_label.append(1)
    test_label.append(0)
    test_label.append(0)


    test_dict = {'text':test_list,
                'label':test_label}
    lstmModel.eval()
    lstmModel.to(device)
    test_data = getDataSet(test_dict,word2idx,idx2word)
    preds_list =[]
    with torch.no_grad():
        for idxs,lengths,labels in test_data:
            idxs = idxs.to(device)#(seq_len,)
            lengths = lengths.to(device)#(length,)
            labels = labels.to(device)#(1,)
            idxs = torch.unsqueeze(idxs,0)  #(1,seq_len,)
            lengths = torch.unsqueeze(lengths,0)#(1,length,)
            labels =  torch.unsqueeze(labels,0)#(1,1,)
            logits,_ = lstmModel(idxs,lengths)#1,num_class
            preds = torch.argmax(logits,1) #1
            actual_pred = preds.item()
            preds_list.append(actual_pred)
            print_str = '正类' if actual_pred == 1 else '负类'
            print(print_str)


    # with torch.no_grad():
    #     for idxs,lengths,labels in test_data:
    #         idxs = idxs.to(device)
    #         lengths = lengths.to(device)
    #         labels = labels.to(device)
    #         idxs = torch.unsqueeze(idxs,0)
    #         lengths = torch.unsqueeze(lengths,0)
    #         labels = torch.unsqueeze(labels,0)

    #         logits,_ = lstmModel(idxs,lengths)#batch,num_class
    #         preds = torch.argmax(logits,1) #bact
    #         actual_pred = (preds == labels).item()
    #         preds_list.append(actual_pred)
    #         print_str = '正类' if actual_pred == 1 else '负类'
    #         print(print_str)    





run_train()
run_test()






