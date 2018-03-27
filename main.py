from __future__ import print_function

import os, time, argparse
from tqdm import tqdm
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var

#import utils
import gc
import sys
import Constants
from model import *
from tree import Tree
from vocab import Vocab

from dataset import *
from trainer import Trainer

from config import parse_args   

# MAIN BLOCK
def main():
    
    

    args=parse_args()
    print(args)


    num_classes = 7
    
    data_dir = args.data_dir #,'train_texts.blk')
    train_file=os.path.join(data_dir,'train_data.pth')
    
    #val_dir = args.val_data #'val_texts.blk')
    val_file= os.path.join(data_dir,'val_data.pth')

    
    vocab_file="../data/vocab.txt"
    vocab = Vocab(filename=vocab_file)
    

    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
        
    else:
        train_dataset = WebKbbDataset(vocab, num_classes,os.path.join(data_dir,'train_texts.blk'),os.path.join(data_dir,'train_labels.blk'))

        torch.save(train_dataset, train_file)
    
    if os.path.isfile(val_file):
        val_dataset = torch.load(val_file)
        
    else:
        val_dataset = WebKbbDataset(vocab, num_classes,os.path.join(data_dir,'val_texts.blk'),os.path.join(data_dir,'val_labels.blk'))
        torch.save(val_dataset, val_file)
    


    
    
    

    vocab_size=vocab.size()
    in_dim=200
    mem_dim=200
    hidden_dim=200
    num_classes=7
    sparsity=True
    freeze=args.freeze_emb
    epochs=args.epochs
    lr=args.lr
    pretrain=args.pretrain
    
    
    
    cuda_flag=True

    if not torch.cuda.is_available():
        cuda_flag=False
                        
    model = DomTreeLSTM(vocab_size,in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze)
    criterion = nn.CrossEntropyLoss()

    if pretrain:
        
        emb_file = os.path.join('../data', 'emb.pth')
        if os.path.isfile(emb_file):
            emb = torch.load(emb_file)
            print(emb.size())
            print("Embedding weights loaded")
        else:
            print("Embedding file not found")
        
        model.emb.weight.data.copy_(emb)

    optimizer = optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    trainer = Trainer(model, criterion, optimizer,train_dataset,val_dataset,cuda_flag=cuda_flag)
    
    for epoch in range(epochs):

        trainer.train()
        
        trainer.test()
        
            
    #trainer.train(train_dataset)
    



if __name__ == "__main__":
    
    main()

