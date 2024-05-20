import torch.nn
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.ResBlock import ResBlock
from layers.others import Nothing
from layers.Spectralpooling import Spectralpooling
from layers.Avgpooling_no import Avgpooling
from layers.Maxpooling_no import Maxpooling


class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()

        self.pos=config.pos
        self.pooling_mode = config.pooling_mode

        self.resblock1=ResBlock(config.c_in,config.nf,config.ksize,bn=config.bn)
        self.resblock2=ResBlock(config.nf,config.nf*2,config.ksize,bn=config.bn)
        self.resblock3=ResBlock(config.nf*2,config.nf*2,config.ksize,bn=config.bn)


        self.gpooling = nn.AdaptiveAvgPool1d(1)

        self.pooling_1=Nothing()
        if self.pooling_mode=="avg":
            self.pooling_1=Avgpooling(config.pooling_output,config.trans)
        elif self.pooling_mode=="max":
            self.pooling_1=Maxpooling(config.pooling_output,config.trans)
        elif self.pooling_mode=="sp":
            self.pooling_1=Spectralpooling(config.pooling_output,norm=config.norm,trans=config.trans)



        self.timepooling=Nothing() if self.pos==-1 else self.pooling_1
        self.flat = nn.Flatten()

        self.fc=nn.Linear(config.nf*2,config.class_num)

        self.seq = nn.Sequential(self.resblock1, self.resblock2, self.resblock3)
        if self.pos!=-1:
            self.seq = nn.Sequential(*list(self.seq.children())[:self.pos] + [self.timepooling] + list(self.seq.children())[self.pos:])

    def forward(self, x,mask=None,length=None):
        x=x if len(x.shape)==3 else x.unsqueeze(-1)
        x=x.permute(0,2,1)
        if self.pos!=-1:
            self.timepooling.set_length(length)
        x=self.seq(x)
        if mask!=None:
            x=x*mask.unsqueeze(1)
        x=self.gpooling(x)
        x=self.flat(x)
        x=self.fc(x)
        return x
