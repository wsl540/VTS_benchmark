import torch.nn as nn
from layers.others import BNornot,Add
from layers.ConvBlock import ConvBlock

class ResBlock(nn.Module):
    def __init__(self,ni,nf,ksize,bn=True):
        super().__init__()
        self.convblock1=ConvBlock(ni,nf,ksize[0],bn)
        self.convblock2=ConvBlock(nf,nf,ksize[1],bn)
        self.convblock3=ConvBlock(nf,nf,ksize[2],bn,act=False)

        self.shortcut=BNornot(bn,num_features=nf) if ni==nf else ConvBlock(ni,nf,kernel_size=1,bn=bn,act=False)
        self.add=Add()
        self.act=nn.ELU()

    def forward(self,x):
        res=x
        x=self.convblock1(x)
        x=self.convblock2(x)
        x=self.convblock3(x)
        x=self.add(x,self.shortcut(res))
        x=self.act(x)

        return x

