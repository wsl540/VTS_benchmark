import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.Linear import Linear

class Model(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self,config):
        super(Model, self).__init__()
        self.pos=config.pos
        self.pooling_mode = config.pooling_mode

        self.fc1=Linear(config.input_len,256)
        self.fc2=Linear(256,512)
        self.fc3=Linear(512,256)

        # self.gpooling=nn.AdaptiveAvgPool1d(1)
        # self.flat = nn.Flatten()
        self.fc=nn.Linear(256,config.class_num)
        self.seq = nn.Sequential(self.fc1, self.fc2, self.fc3)


    def forward(self, x,mask=None,length=None):

        x=self.seq(x)
        x=self.fc(x)
        return x