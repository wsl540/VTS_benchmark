import torch
import torch.nn as nn

class Spectralpooling(nn.Module):
    def __init__(self, outputlength,pad_pos='post',norm='backward',trans=False):
        super(Spectralpooling, self).__init__()
        self.output = outputlength
        self.need = torch.div(self.output, 2, rounding_mode='floor') + 1
        self.length = None
        self.pad_pos=pad_pos
        self.norm=norm
        self.trans=trans

    def set_length(self, length):
        self.length = length

    def forward(self,x):
        if self.trans:
            x=x.permute(0,2,1)
        four=torch.fft.rfft(x,dim=-1,norm=self.norm)
        output=torch.fft.irfft(four[:,:,:self.need],dim=-1,norm=self.norm)
        if self.trans:
            output=output.permute(0,2,1)
        return output