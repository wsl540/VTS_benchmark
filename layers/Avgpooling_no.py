import torch
import torch.nn as nn

class Avgpooling(nn.Module):
    def __init__(self, outputlength,trans=False):
        super(Avgpooling, self).__init__()
        self.output = outputlength
        self.length = None
        self.pooling=torch.nn.AdaptiveAvgPool1d(self.output)
        self.trans=trans

    def set_length(self, length):
        self.length = length

    def forward(self,x):
        if self.trans:
            x = x.permute(0, 2, 1)
        output=self.pooling(x.contiguous())
        if self.trans:
            output=output.permute(0,2,1)
        return output