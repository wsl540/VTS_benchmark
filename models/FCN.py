import torch
import torch.nn as nn
from layers.ConvBlock import ConvBlock
from layers.Spectralpooling import Spectralpooling
from layers.others import Nothing
from layers.Avgpooling_no import Avgpooling
from layers.Maxpooling_no import Maxpooling


class Model(nn.Module):
    def __init__(self, config):
        super(Model,self).__init__()
        self.pooling_mode = config.pooling_mode
        self.pos = config.pos
        self.convblock1 = ConvBlock(config.c_in, config.c_out[0], config.ksize[0],bn=config.bn)
        self.convblock2 = ConvBlock(config.c_out[0], config.c_out[1], config.ksize[1],bn=config.bn)
        self.convblock3 = ConvBlock(config.c_out[1], config.c_out[2], config.ksize[2],bn=config.bn)
        self.pooling_1 =Nothing()
        if self.pooling_mode == "avg":
            self.pooling_1 = Avgpooling(config.pooling_output,config.trans)
        elif self.pooling_mode == "max":
            self.pooling_1 =Maxpooling(config.pooling_output,config.trans)
        elif self.pooling_mode == "sp":
            self.pooling_1=Spectralpooling(config.pooling_output,norm=config.norm,trans=config.trans)

        self.gpooling=nn.AdaptiveAvgPool1d(1)
        self.timepooling=Nothing() if self.pos==-1 else self.pooling_1
        self.flat = nn.Flatten()
        self.fc=nn.Linear(config.c_out[2],config.class_num)

        self.seq = nn.Sequential(self.convblock1, self.convblock2, self.convblock3)
        if self.pos!=-1:
            self.seq = nn.Sequential(*list(self.seq.children())[:self.pos] + [self.timepooling] + list(self.seq.children())[self.pos:])

    def forward(self, x,mask=None,length=None):
        x=x if len(x.shape)==3 else x.unsqueeze(-1)
        x=x.permute(0,2,1)
        if length!=None and self.pos!=-1:
            self.timepooling.set_length(length)
        x=self.seq(x)
        if mask!=None:
            x=x*mask.unsqueeze(1)
        x=self.gpooling(x)
        x=self.flat(x)
        x=self.fc(x)
        return x


