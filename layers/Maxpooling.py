import torch
import torch.nn as nn

class Maxpooling(nn.Module):
    def __init__(self, outputlength,trans=False):
        super(Maxpooling, self).__init__()
        self.output = outputlength
        self.length = None
        self.pooling=torch.nn.AdaptiveMaxPool1d(self.output)
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
                output[i, :, :length[i]] = x[i, :, :length[i]]
            else:
                output[i]=self.pooling(cur)
        if self.trans:
            output=output.permute(0,2,1)
        return output