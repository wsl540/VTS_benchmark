import torch.nn as nn
import torch
from layers.ConvBlock import ConvBlock
from layers.others import BNornot,Add,Concat,Nothing
from layers.Spectralpooling import Spectralpooling
from layers.Avgpooling_no import Avgpooling
from layers.Maxpooling_no import Maxpooling

class InceptionBlock(nn.Module):
    def __init__(self, ni, nf,ks, depth,bn,pos,pooling_mode,pooling_output,norm,trans):
        super().__init__()
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        self.pooling_mode=pooling_mode
        self.pooling_output=pooling_output
        self.pooling_1= Nothing()
        self.trans=trans
        self.norm=norm
        if self.pooling_mode== "avg":
            self.pooling_1=Avgpooling(self.pooling_output,self.trans)
        elif self.pooling_mode=="max":
            self.pooling_1=Maxpooling(self.pooling_output,self.trans)
        elif self.pooling_mode=="sp":
            self.pooling_1=Spectralpooling(self.pooling_output,norm=self.norm,trans=self.trans)

        self.timepooling=Nothing() if pos==-1 else self.pooling_1
        self.pos=pos
        self.depth=depth

        for d in range(depth):
            self.inception.append(InceptionModule(ni if d == 0 else nf * 4, nf,ks,bn=bn))
            if d % 3 == 2:
                n_in, n_out = ni if d == 2 else nf * 4, nf * 4
                self.shortcut.append(BNornot(bn,n_out) if n_in == n_out else ConvBlock(n_in, n_out, 1,bn=bn,act=False))
        self.add = Add()
        self.act=nn.GELU()

    def forward(self, x,length):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            if d % 3 == 2:
                res = x = self.act(self.add(x, self.shortcut[d // 3](res)))
                if (d == 2 and self.pos == 1) or (d == 5 and self.pos == 2):
                    self.timepooling.set_length(length)
                    x = self.timepooling(x)
                    res = x
        return x

class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks, bn,bottleneck=True):
        super().__init__()
        ks = [ks // (2 ** i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        bottleneck = bottleneck if ni > 1 else False
        self.bottleneck = nn.Conv1d(ni, nf, 1, bias=False) if bottleneck else Nothing()
        self.convs = nn.ModuleList([nn.Conv1d(nf if bottleneck else ni, nf, k, bias=False,padding=(k-1)//2) for k in ks])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(ni, nf, 1, bias=False)])
        self.concat = Concat()
        self.bn = BNornot(bn,num_features=nf * 4)
        self.act=nn.ELU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.concat([l(x) for l in self.convs] + [self.maxconvpool(input_tensor)])
        return self.act(x)
