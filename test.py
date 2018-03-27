import torch.nn as nn
from torch.autograd import Variable as Var
fro
emb= nn.Embedding(80000, 200)
block_idx=torch.tensor([1,2,3,4,5,6,10])
block_idx=Var(block_idx)
emb.weight.requires_grad=True

res=emb(block_idx.unsqueeze(0))


