import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Spectralpooling import Spectralpooling
from layers.others import Nothing
from layers.Maxpooling_no import Maxpooling
from layers.Avgpooling_no import Avgpooling


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, mask=None):
        new_x, attn = self.attention(
            x, x, x, attn_mask=mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers,pooling_output,pos,norm_layer=None,pooling_mode="sp",norm_fft='backward',trans=True):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer
        self.pos=pos
        self.pooling_mode=pooling_mode
        self.pooling_output=pooling_output
        self.norm_fft=norm_fft
        self.trans=trans

        if self.pooling_mode == "avg":
            self.pooling_1 = Avgpooling(self.pooling_output,trans=self.trans)
        elif self.pooling_mode == "max":
            self.pooling_1 = Maxpooling(self.pooling_output,trans=self.trans)
        elif self.pooling_mode == "sp":
            self.pooling_1=Spectralpooling(self.pooling_output,norm=self.norm_fft,trans=self.trans)
        self.timepooling=Nothing() if pos==-1 else self.pooling_1

    def forward(self, x,attn_mask,length):
        attns = []
        for i,attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x,attn_mask)
            attns.append(attn)
            if i+1==self.pos:
                self.timepooling.set_length(length)
                x=self.timepooling(x)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

