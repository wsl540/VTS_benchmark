import torch
import torch.nn as nn

from layers.Embed import DataEmbedding
from layers.Spectralpooling import Spectralpooling
from layers.others import Nothing
from layers.Avgpooling_no import Avgpooling
from layers.Maxpooling_no import Maxpooling

class Model(nn.Module):
    def __init__(self, config):
        super(Model,self).__init__()
        self.pos = config.pos
        self.pooling_mode = config.pooling_mode

        self.lstm_1=nn.LSTM(input_size=config.c_in,hidden_size=config.hidden_size,batch_first=True)
        self.lstm_2=nn.LSTM(input_size=config.hidden_size,hidden_size=config.hidden_size,batch_first=True)
        self.gpooling =nn.AdaptiveMaxPool1d(1)

        self.pooling_1 = Nothing()
        if self.pooling_mode == "avg":
            self.pooling_1 = Avgpooling(config.pooling_output,trans=config.trans)
        elif self.pooling_mode == "max":
            self.pooling_1 = Maxpooling(config.pooling_output,trans=config.trans)
        elif self.pooling_mode == "sp":
            self.pooling_1=Spectralpooling(config.pooling_output,norm=config.norm,trans=config.trans)

        self.flat=nn.Flatten()
        self.fc=nn.Linear(config.hidden_size,config.class_num)
        self.timepooling=Nothing() if config.pos==-1 else self.pooling_1

    def forward(self, x,mask,length):
        x = x if len(x.shape) == 3 else x.unsqueeze(-1)
        if self.pos == -1:
            output_lstm_1, _ = self.lstm_1(x)
            output_lstm_2, _ = self.lstm_2(output_lstm_1)
            x=output_lstm_2.permute(0, 2, 1)
        elif self.pos == 1:
            self.timepooling.set_length(length)
            output_lstm_1, _ = self.lstm_1(x)
            x = self.timepooling(output_lstm_1)
            output_lstm_2, _ = self.lstm_2(x)
            x=output_lstm_2.permute(0, 2, 1)
        elif self.pos == 2:
            self.timepooling.set_length(length)
            output_lstm_1, _ = self.lstm_1(x)
            output_lstm_2, _ = self.lstm_2(output_lstm_1)
            x = self.timepooling(output_lstm_2)
            x=x.permute(0, 2, 1)
        if mask != None:
            x=x*mask.unsqueeze(1)

        x = self.gpooling(x)
        x = self.flat(x)
        x = self.fc(x)
        return x