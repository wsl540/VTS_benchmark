import torch.nn as nn
import torch

class Add(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x,y):
        return x+y

class BNornot(nn.Module):
    def __init__(self,bn,num_features=None):
        super().__init__()
        if bn:
            self.layer=nn.BatchNorm1d(num_features)
        else:
            self.layer=Nothing()
    def forward(self,x):
        return self.layer(x)

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *x):
        return torch.cat(*x, dim=self.dim)

class  Nothing(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x