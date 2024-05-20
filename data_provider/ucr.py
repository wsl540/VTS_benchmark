import numpy as np
import pandas as pd
import torch

from data_provider.preprocessing import predeal
from utils.tools import count_non_missing

def collate_fn(data,args):
    batch_size=len(data)
    features,labels=zip(*data)
    features=np.array(features)
    labels=torch.Tensor(labels).reshape(batch_size,1)
    lengths,masking,pre_features=None,None,None
    if args.data=='VARY_UCR':
        if args.pos!=-1:
            lengths=np.apply_along_axis(count_non_missing, 1, features).reshape(batch_size,1)
            lengths=torch.tensor(lengths)
        pre_features = predeal(features, length=args.seq_len, way=args.way, norm=args.norm)
        if args.use_masking==True:
            assert args.way=='zeropad_post'
            masking=np.zeros((batch_size,args.seq_len))
            if pre_features.shape[1]>=features.shape[1]:
                masking[:,:features.shape[1]] = np.isnan(features).astype(np.float32)
                masking[:,:features.shape[1]]= 1 -masking[:,:features.shape[1]]  # Invert the mask: NaN becomes 0, non-NaN becomes 1
            else:
                masking= np.isnan(features).astype(np.float32)
                masking = 1 - masking
                masking= masking[:, :pre_features.shape[1]]
            masking = torch.tensor(masking)


        pre_features=torch.tensor(pre_features)
    elif args.data=='DWT':
        pre_features=torch.tensor(features)


    return pre_features,labels,masking,lengths
