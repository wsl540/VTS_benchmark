import torch
import torch.nn as nn

class Spectralpooling(nn.Module):
    def __init__(self, outputlength,pad_pos='post',norm='backward',trans=False):
        super(Spectralpooling, self).__init__()
        self.output =outputlength if outputlength%2==0 else outputlength-1
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
        length=self.length.reshape(-1)
        output=torch.zeros(x.shape[0],x.shape[1],self.output).to(x.device)
        for i in range(0,x.shape[0]):
            cur = x[i, :, :length[i]]
            if length[i]<self.output:
                output[i,:,:length[i]]=x[i,:,:length[i]]
            else:
                four=torch.fft.rfft(cur,dim=-1,norm=self.norm)
                four=four[:,:self.need]
                output[i]=torch.fft.irfft(four,dim=-1,norm=self.norm)
        if self.trans:
            output=output.permute(0,2,1)
        return output