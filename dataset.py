import os
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.utils.data as data
from torch.autograd import Variable as Var

from tree import Tree
from vocab import Vocab
import Constants
import utils
from nltk import word_tokenize

# Dataset class for SICK dataset
class SICKDataset(data.Dataset):
    def __init__(self, path, vocab, num_classes):
        super(SICKDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes

        self.lsentences = self.read_sentences(os.path.join(path,'a.toks'))
        self.rsentences = self.read_sentences(os.path.join(path,'b.toks'))

        self.ltrees = self.read_trees(os.path.join(path,'a.parents'))
        self.rtrees = self.read_trees(os.path.join(path,'b.parents'))

        self.labels = self.read_labels(os.path.join(path,'sim.txt'))

        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        ltree = deepcopy(self.ltrees[index])
        rtree = deepcopy(self.rtrees[index])
        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])
        return (ltree,lsent,rtree,rsent,label)

    def read_sentences(self, filename):
        with open(filename,'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_trees(self, filename):
        with open(filename,'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents = map(int,line.split())
        trees = dict()
        root = None
        for i in xrange(1,len(parents)+1):
            #if not trees[i-1] and parents[i-1]!=-1:
            if i-1 not in trees.keys() and parents[i-1]!=-1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx-1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx-1] = tree
                    tree.idx = idx-1
                    #if trees[parent-1] is not None:
                    if parent-1 in trees.keys():
                        trees[parent-1].add_child(tree)
                        break
                    elif parent==0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self, filename):
        with open(filename,'r') as f:
            labels = map(lambda x: float(x), f.readlines())
            labels = torch.Tensor(labels)
        return labels

class WebKbbDataset(data.Dataset):
    def __init__(self, vocab, num_classes,data_dir,label_dir):
        super(WebKbbDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes
        #self.word_set=[line.strip() for line in open('../data/vocab.txt','r',encoding='latin-1').readlines()]

        #self.tsentences = self.read_sentences(os.path.join(path,'a.toks'))
        self.labels = self.read_labels(label_dir)
        self.texts = self.read_texts(data_dir)

        #self.trees = self.read_trees(os.path.join(path,'parents.txt'))
        
        
        self.size = self.labels.size(0)
        

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        
        #tree = deepcopy(self.trees[index])
        text = deepcopy(self.texts[index])
        
        label = deepcopy(self.labels[index])
        #return (tree,text,label)
        return (text,label)

    def read_texts(self, filename):
        with open(filename,'r',encoding='latin-1') as f:
            texts = [self.read_text(line) for line in tqdm(f.readlines())]
        return texts

    def read_text(self, line):
        blocks=line.strip().split('|||')
        blocks=[block.split(' ') for block in blocks]
        indices=[]
        for block in blocks:
            
            idx=self.vocab.convertToIdx(block, Constants.UNK_WORD)
            
            indices.append(Var(torch.LongTensor(idx)))
           
        return indices


    def read_trees(self, filename):
        with open(filename,'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents =list(map(int,line.split())) 
        
        trees = dict()
        root = None
        for i in range(1,len(parents)+1):
            #if not trees[i-1] and parents[i-1]!=-1:
            if i-1 not in trees.keys() and parents[i-1]!=-1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx-1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx-1] = tree
                    tree.idx = idx-1
                    #if trees[parent-1] is not None:
                    if parent-1 in trees.keys():
                        trees[parent-1].add_child(tree)
                        break
                    elif parent==0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self, filename):
        with open(filename,'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.Tensor(labels)
        return labels

