from tqdm import tqdm
import sys
import torch
from torch.autograd import Variable as Var

from utils import map_label_to_target
import gc

from tree import Tree
def dfs(root):
    print(root.idx)
    for child in root.children:
        dfs(child)

class Trainer(object):
    def __init__(self, model, criterion, optimizer,train_data,val_data,cuda_flag=False):
        super(Trainer, self).__init__()
        #self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0
        self.num_classes =7
        self.train_data=train_data
        self.val_data=val_data

    # helper function for training
    def train(self,batch_size=50):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = torch.randperm(len(self.train_data))

        nb_samples=len(self.train_data)

        l2_reg = None
        
        
        for idx in tqdm(range(nb_samples),desc='Training epoch ' + str(self.epoch + 1) + '',ascii=True, file=sys.stdout):
            text, label = self.train_data[indices[idx]]
            tree=None
            if len(text)<3:
                nb_samples-=1
                continue
            
            target=map_label_to_target(label,self.num_classes)
            #print(target)
                
            output = self.model(tree, text)
            
            if output is None:
                continue
                nb_samples-=1
            
            loss = self.criterion(output, target)
            #params = self.model.childsumtreelstm.getParameters()
            #0.5*self.args.reg*params_norm*params_norm 
            

            total_loss += loss.data[0]
            loss.backward()

            if idx % batch_size == 0 and idx > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            #del tree, text, label, output
            gc.collect()
        self.epoch += 1
        
        train_loss= total_loss / nb_samples
        print("Train loss:{} ".format(train_loss))

    # helper function for testing
    def test(self):
        
        self.model.eval()
        
        loss = 0
        predictions = torch.zeros(len(self.val_data))
        predictions = predictions
        indices = torch.range(1,self.val_data.num_classes)
        correct = 0
        total = 0
        nb_samples=len(self.val_data)
        for idx in tqdm(range(len(self.val_data)),desc='Testing epoch  '+str(self.epoch)+'',ascii=True, file=sys.stdout):
            text, label = self.val_data[idx]
            tree=None
            if len(text)<3:
                nb_samples-=1
                continue

            target = map_label_to_target(label,self.num_classes)
            
            outputs = self.model(tree, text) # size(1,5)
            if outputs is None:
                continue
                nb_samples-=1
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += target.size(0)
           # print(type(predicted))
           # print(type(target))
            correct += (predicted == target.data).sum()
            err = self.criterion(outputs, target)
            loss += err.data[0]
        loss=loss/nb_samples
        acc=correct/total
            
        #val_loss=loss/len(self.val_data)
        print("Val loss:{} Acc:{}".format(loss,acc))
