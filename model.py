import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

import Constants


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.hidden_states=[]

    def node_forward(self, inputs, child_c, child_h):
      
        child_h_sum = torch.mean(child_h, dim=0, keepdim=True)
      
        
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
                self.fh(child_h) +
                self.fx(inputs).repeat(len(child_h), 1)
            )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h
    def clear_states(self):
        self.hidden_states=[]
    def get_states(self):
        return self.hidden_states

    def forward(self, tree, inputs):
        
        
        _ = [self.forward(tree.children[idx], inputs) for idx in range(tree.num_children)]
        # leaf node
        if tree.num_children == 0:
            child_c = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            child_h = Var(inputs[0].data.new(1, self.mem_dim).fill_(0.))
            
        else:
            # get childern's cell state and hidden
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            #concate child c and child h
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        # compute value of current node
        
        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        self.hidden_states.append(tree.state[1])
        
        return tree.state


# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        out = F.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out))
        return out


# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze):
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes)

    def forward(self, ltree, linputs, rtree, rinputs):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)
        lstate, lhidden = self.childsumtreelstm(ltree, linputs)
        rstate, rhidden = self.childsumtreelstm(rtree, rinputs)
        output = self.similarity(lstate, rstate)
        return output

#classifier
class Classifier(nn.Module):
    def __init__(self, mem_dim, num_classes):
        super(Classifier, self).__init__()
        
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        
        self.l1 = nn.Linear(self.mem_dim, self.num_classes)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, input):
        out=self.logsoftmax(self.l1(input))
        return out


# putting the whole model together
class DomTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity=True, freeze=True,mean_only=False):
        super(DomTreeLSTM, self).__init__()
        self.in_dim=in_dim
        self.emb = nn.Embedding(vocab_size, in_dim,sparse=sparsity)
        self.mean_only=False
        self.CNN_flag=False
        self.num_classes=num_classes
       

        self.zero_var=Var(torch.Tensor(self.in_dim).zero_()).unsqueeze(0)

        if freeze:
            self.emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)

        
        self.filter_frames_block = [1,2,3]
        self.filter_dims_block = [100,100,100]
        ##########################################

        self.filter_frames = [3]
        self.filter_dims = [200]
        
        
        self.conv_models = []
        for i in range(len(self.filter_frames)):
            conv_tmp = nn.Conv1d(200, self.filter_dims[i], self.filter_frames[i], stride=1)
            self.conv_models.append(conv_tmp)
        #######################################
        self.conv_models_block = []
        for i in range(len(self.filter_frames_block)):
            conv_tmp = nn.Conv1d(in_dim, self.filter_dims_block[i], self.filter_frames_block[i], stride=1)
            self.conv_models_block.append(conv_tmp)
        self.drop_prob = 0.1
        

        if self.CNN_flag:
            self.output =  Classifier(200,self.num_classes) 
        else:
            self.output =  Classifier(mem_dim,self.num_classes) 

    def forward(self,tree,block_texts):

        all_blocks = []  
        nb_blocks=0
        if True:
            for block_text in block_texts:
                block_len=len(block_text)
                if len(block_text)==0:    
                    all_blocks.append(self.zero_var)
                    continue
                block_embs = self.emb(block_text)
                block_rep = torch.max(block_embs,0)[0]
                #print(type(block_rep))
                #print(block_rep.size())
                #block_rep = torch.mean(block_embs,dim=0)
                all_blocks.append(block_rep.unsqueeze(0))
            nb_blocks=len(all_blocks)    
            all_blocks=torch.cat(all_blocks).unsqueeze(0).transpose(1,2)
        else:
            for block_text in block_texts:
                block_len=len(block_text)
                if len(block_text)<3:    
                    continue
                    
                block_embs = self.emb(block_text)
                block_embs=block_embs.transpose(0,1).unsqueeze(0)
                #block_rep = torch.mean(block_embs,dim=0)
                #all_blocks.append(block_rep.unsqueeze(0))
                cs = []
                for i in range(len(self.filter_frames_block)):    
                    ci = F.relu(self.conv_models_block[i](block_embs))
                    ci = F.avg_pool1d(ci,block_len -self.filter_frames_block[i]+1).view(-1, self.filter_dims_block[i])
                    cs.append(ci)
                
                cs = torch.cat(cs, 1)
        
                all_blocks.append(cs)
            nb_blocks=len(all_blocks)
            if nb_blocks<3:
                return None
            all_blocks=torch.cat(all_blocks).unsqueeze(0).transpose(1,2)
            
            #print(all_blocks.size())
            
        
        cs = []
        for i in range(len(self.filter_frames)):    
            ci = F.relu(self.conv_models[i](all_blocks))
            ci = F.avg_pool1d(ci,nb_blocks -self.filter_frames[i]+1).view(-1, self.filter_dims[i])
            cs.append(ci)
        
        cs = torch.cat(cs, 1)

        output = self.output(cs)
        return output

        
        if self.mean_only:
            mean_rep=torch.mean(all_blocks,dim=0)
            mean_rep=mean_rep.unsqueeze(0)
            
            output = self.output(mean_rep)

        else:
            self.childsumtreelstm.clear_states()
            state, hidden = self.childsumtreelstm(tree, all_blocks)
            hidden_states=self.childsumtreelstm.get_states()
            
            hidden_states=torch.cat(hidden_states)

            
            output = self.output(state)
        
        return output

