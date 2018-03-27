import os
import sys
import re
import queue
from bs4 import BeautifulSoup
import bs4
import copy
from shutil import copyfile
import torch

#from model import DomTreeLSTM

from tree import Tree
from nltk import word_tokenize
from vocab import Vocab


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)




def dfs(root):
    if root is not None:
        
        print(root.words)
        for child in root.children:
            dfs(child)

def clean_tree(root):

    unhandled_children=queue.Queue()
    for child in root.children:
        unhandled_children.put(child)
    handled_children=[]
    while unhandled_children.qsize()>0:
        child=unhandled_children.get()  # not processed
        
        if len(child.words)>0:
            handled_children.append(child)  # it is okay to keep
        else:
            for grandchild in child.children:
                unhandled_children.put(grandchild)
    root.clear_children()
    #root.children=handled_children
    for child in handled_children:
        root.add_child(child)

    for child in root.children:
        clean_tree(child)
    return root
        
def build_tree(soup):
    
    q_text=queue.Queue()
    q_tree=queue.Queue()
    
    if soup.name is None:
        return None
    root= Tree(words=str(soup.string).lower().split(' '))
    q_text.put(soup)
    q_tree.put(root)
    
    while q_text.qsize()>0:
        current_dom_node=q_text.get()
        current_tree_node=q_tree.get()
        
        if not isinstance(current_dom_node, bs4.element.NavigableString) and len(list(current_dom_node.children))>0 :
            
            for child in current_dom_node.children:              
                
                block_words=[]
                if child.string is not None and  not (child.parent.string== child.string):
                    print(type(child))
         
                new_child=Tree(words=block_words)
                current_tree_node.add_child(new_child)
                q_text.put(child)
                q_tree.put(new_child)
    return root
    #printprint(root.depth())
    #dfs(root)
    
def generate_line_reprensentation(root):
   
    block_texts=[]
    parent_idx=[]
    q_tree=queue.Queue()
    q_tree.put(root)
    idx=0
    while q_tree.qsize()>0:
        idx+=1
        node=q_tree.get()
        node.idx=idx
        block_texts.append(" ".join(node.words))
        for child in node.children:
            q_tree.put(child)
    parent_idx=[0]*idx
    
    q_tree.put(root)
    #print(idx)
    while q_tree.qsize()>0:
        node=q_tree.get()
        parent_idx[node.idx-1]=0 # assume it is root node
        if node.parent is not None:
            parent_idx[node.idx-1]=node.parent.idx    
        for child in node.children:
            q_tree.put(child)
    
    return parent_idx,block_texts

def length_compare():
    dirs=os.listdir('htmls/faculty')
    for dir in dirs:
        if dir.find('Jessica.K.Hodgins.html') > -1:
            path=os.path.join('plain-data-c-5/faculty',dir)
            break
    
    text1=open('tree-text.txt','r',encoding='latin-1').read()
    print(text1)
    return 0
    text1=text1.replace('\n',' ')
    text1=re.sub(' +',' ',text1)
    text2=open(path+'.txt','r').read()
    text2=text1.replace('\n',' ')
    text2=re.sub(' +',' ',text2)
    #print(len(text2.split(' ')))
    #print(len(text1.split(' ')))
def get_soup(path):

    html=open(path,'r',encoding='latin-1').read()
    soup = BeautifulSoup(html,"html5lib") 
    return soup
def htmls2trees():
    
    path='data/raw-data/train/'
    all_cate=os.listdir(path)
    idx=-1
    fp=open('data/train/parents.txt','w',encoding='latin-1')
    ft=open('data/train/texts.txt','w',encoding='latin-1')
    fl=open('data/train/labels.txt','w')
    for cate in all_cate:
        dirs=os.listdir(os.path.join(path,cate))
        if  len(dirs)>=0:
            idx+=1
            print("Processing {}, cate ID {}".format(cate,idx))
            for fn in dirs:
                
                html_path=os.path.join(path,cate,fn)
                #print(html_path)
                soup=get_soup(html_path)
                tree=build_tree(soup)
                tree=clean_tree(tree)
                #print(tree.depth())
                parents_idx,block_text=generate_line_reprensentation(tree)
                
                block_str="|||".join(block_text)+'\n'
                
                fp.write(" ".join(map(str,parents_idx))+"\n")
                ft.write(block_str)
                fl.write(str(idx)+'\n')
    fp.close()
    ft.close()
    fl.close()
    

    path='data/raw-data/val/'
    all_cate=os.listdir(path)
    idx=-1
    fp=open('data/val/parents.txt','w',encoding='latin-1')
    ft=open('data/val/texts.txt','w',encoding='latin-1')
    fl=open('data/val/labels.txt','w')
    for cate in all_cate:
        dirs=os.listdir(os.path.join(path,cate))
        if  len(dirs)>=0:
            idx+=1
            print("Processing {}, cate ID {}".format(cate,idx))
            for fn in dirs:
                
                html_path=os.path.join(path,cate,fn)
                print(html_path)
                soup=get_soup(html_path)
                tree=build_tree(soup)
                
                tree=clean_tree(tree)
                #print(tree.depth())
                parents_idx,block_text=generate_line_reprensentation(tree)
                fp.write(" ".join(map(str,parents_idx))+"\n")
                ft.write("|||".join(block_text)+'\n')
                fl.write(str(idx)+'\n')
    fp.close()
    ft.close()
    fl.close()

def getvocab():
    lines1=open('data/train/texts.txt','r',encoding='latin-1').readlines()
    lines2=open('data/val/texts.txt','r',encoding='latin-1').readlines()
    lines=lines1+lines2

    lines= [line.split('|||') for line in lines]
    word_set=set()
    for line in lines:
        for block in line:
            words=block.split(' ')
            for word in words:
                word_set.add(word)
    others=['<blank>','<unk>','<s>','</s>']
    for word in others:
        if word not in word_set:
            word_set.add(word)

    print(len(word_set))
    f=open('data/vocab.txt','w',encoding='latin-1')
    for word in word_set:
        f.write(word+'\n')

'''
none---[]--[]
        ---[]-date,tue...
             -jessica k, hodgins,information,page...-[]
             -[]
             -wa0
             -[]-wa1
                -[]
             -[]
             -[]-graphics visualizatio
                -jessica k  hodgins
             -[]
             -[]
             -[]
             -[wa2]
             -[]
             -[]-jessica is an assitant prof..
                -wa3...
                -the leg lab...
                -at the mit lab
                -wa4
                -computer animation
                -by using comtrol alg
             -[]
             -[]
             -contact information..-[]
             -[]
             -jessica k hodgins-[]
             -[]
             -graphics visual
             -[]
             -college of computing
             -[]
             -801 atlantic dirve
             -[]
             -georgia institue of tech
             -[]
             -atlanta ga 303332
             -[]
             -404 894-..
             -[]
             -email ..
             -[]
             -wa5
             -[]
             -[]
             -[]
            
'''      
def load_word_vectors(path):
    if os.path.isfile(path+'.pth') and os.path.isfile(path+'.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path+'.pth')
        vocab = Vocab(filename=path+'.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    count = sum(1 for line in open(path+'.txt',encoding='latin-1'))
    with open(path+'.txt','r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words = [None]*(count)
    vectors = torch.zeros(count,dim)
    with open(path+'.txt','r',encoding='latin-1') as f:
        idx = 0
        for line in f:
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            #print(contents[1:])
            vectors[idx] = torch.Tensor(list(map(float, contents[1:])))
            idx += 1
    with open(path+'.vocab','w',encoding='latin-1') as f:
        for word in words:
            f.write(word+'\n')
    vocab = Vocab(filename=path+'.vocab')
    torch.save(vectors, path+'.pth')
    return vocab, vectors

def getembd():
    vocab_file='data/vocab.txt'
    vocab = Vocab(filename=vocab_file)
    emb_file = os.path.join('data/', 'webkbb_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
        print(emb.size())
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join('data/glove','glove.6B.200d'))
        print('==> GLOVE vocabulary size: %d ' % glove_vocab.size())

        emb = torch.zeros(vocab.size(),glove_emb.size(1))

        for word in vocab.labelToIdx.keys():
            
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
            else:
                emb[vocab.getIndex(word)] = torch.Tensor(emb[vocab.getIndex(word)].size()).normal_(-0.05,0.05)
        torch.save(emb, emb_file)
        is_preprocessing_data = True # flag to quit
        print('done creating emb, quit')

     
def test():
    #print('testing')
    dirs=os.listdir('htmls/faculty')
    for dir in dirs:
        if dir.find('Jessica.K.Hodgins.html') > -1:
            path=os.path.join('htmls/faculty',dir)
            break
    tree=build_tree(get_soup(path))
    print(tree.depth())
    #print(tree.children[0].children[1].children[19].children[0].words)
    #print(tree.depth())
    #tree=clean_tree(tree)
    #print(tree.depth())
    #dfs(tree)
    #parent_idx,block_text=generate_line_reprensentation(tree)
    #dfs(tree)
    #print(len(parent_idx))
    #print(block_text)
    

    #dfs(tree)
    #vocab_size=10000
    #in_dim=300
    #mem_dim=100
    #hidden_dim=200
    #num_classes=5
    #sparsity=True
    #freeze=False
    #sent=['', 'by', 'using', 'control', 'algorithms', 'in', 'combination', 'with', 'physically', 'realistic', 'simulation.', 'in', '1994', 'she', 'received', 'a', 'nsf', 'young', 'investigator', 'award', 'and', 'a', 'packard', 'fellowship.', 'in', '1995', 'she', 'received', 'a', 'sloan', 'fellowship.', 'her', 'recent', 'research', 'funding', 'has', 'been', 'from', 'the', 'national', 'science', 'foundation,', 'the', 'jet', 'propulsion', 'laboratory,', 'and', 'the','mitsubishi', 'electric', 'research', 'laboratory.', '']
    #model=DomTreeLSTM(vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze)
    
    #s=model.forward(tree,sent)
    #print('testing')

    return 0
def train_val_copy():
    
    source_base='data/htmls/'
    cate_names=os.listdir(source_base)

    for cate_name in cate_names:
        fns=os.listdir(os.path.join(source_base,cate_name))
        train_dirs=fns[0:int(len(fns)*0.8):]
        dest=os.path.join('data/raw/train/',cate_name)
        if not os.path.exists(dest):
            os.makedirs(dest)
        for fn in train_dirs:
            copyfile(os.path.join(source_base,cate_name,fn),os.path.join(dest,fn))


    
if __name__=='__main__':
    #getembd()
    #train_val_copy()

    #main()
    
    #test()
    #htmls2trees()
    #getvocab()

    #length_compare()
    