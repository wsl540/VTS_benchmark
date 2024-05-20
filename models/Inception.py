import torch
from torch import nn
from layers.InceptionBlock import InceptionBlock


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos = config.pos
        self.inceptionblock = InceptionBlock(config.c_in, config.nf,config.ks,depth=config.depth, bn=config.bn,pos=config.pos,pooling_mode=config.pooling_mode,
                                             pooling_output=config.pooling_output,norm=config.norm,trans=config.trans)

        self.gpooling=nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(config.nf*4, config.class_num)

    def forward(self, x,mask=None,length=None):
        x = x if len(x.shape) == 3 else x.unsqueeze(-1)
        x=x.permute(0,2,1)
        x = self.inceptionblock(x,length)
        if mask!=None:
            x=x*mask.unsqueeze(1)
        x=self.gpooling(x)
        x=self.flatten(x)
        x = self.fc(x)
        return x

