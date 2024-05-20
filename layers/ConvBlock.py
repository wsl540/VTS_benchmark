import torch.nn as nn
from layers.others import BNornot
class ConvBlock(nn.Module):
    def __init__(self,c_in,c_out,kernel_size,bn=False,act=True):
        super().__init__()
        self.Conv1d=nn.Conv1d(in_channels=c_in,out_channels=c_out,kernel_size=kernel_size,padding=(kernel_size-1)//2,groups=1,bias=True)
        self.bn=BNornot(bn,c_out)
        self.seq=nn.Sequential(self.Conv1d,self.bn)
        if act:
            self.act=nn.GELU()
            self.seq.add_module('activation',self.act)

    def forward(self,x):
        x=self.seq(x)
        return x