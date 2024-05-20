import torch.nn as nn
class Linear(nn.Module):
    def __init__(self,d_in,d_out,act=True):
        super().__init__()
        self.linear=nn.Linear(d_in,d_out)
        self.act=nn.GELU()
        self.seq=nn.Sequential(self.linear,self.act) if act else nn.Sequential(self.linear)

    def forward(self,x):
        x=self.seq(x)
        return x