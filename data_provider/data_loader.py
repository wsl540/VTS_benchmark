import glob
import os.path
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.tools import count_non_missing
from tqdm import tqdm
from data_provider import dtw

class UCRloader(Dataset):
    def __init__(self,args,flag):
        self.data,self.label=self.load_all(args,flag=flag)

    def load_all(self,args,flag=None):
        data_path=args.data_path
        file_paths=glob.glob(os.path.join(data_path,'*'))
        if len(file_paths)==0:
            raise Exception('No files found using {}.'.format(os.path.join(data_path,'*')))
        if flag is not None:
            file_paths=list(filter(lambda x:re.search(flag,x),file_paths))
        input_paths=[p for p in file_paths if os.path.isfile(p) and p.endswith('.tsv')]

        if len(input_paths)==0:
            raise Exception('No .tsv file found using {}.'.format(data_path))
        features,labels=self.load_single(input_paths[0])
        lengths=np.apply_along_axis(count_non_missing, 1, features)

        if flag=='TRAIN' and args.seq_len==0:
            if args.mode=='max':
                args.seq_len=int(np.max(lengths))
            elif args.mode=='min':
                args.seq_len=int(np.min(lengths))
        return features,labels

    def load_single(self,file_path):
        df = pd.read_csv(file_path, sep="\t", header=None)
        features=df.iloc[:,1:].values
        labels=df.iloc[:,0].values
        labels=labels-labels.min()
        class_names=list(set(labels))
        class_names.sort()
        self.class_names=class_names
        return features,labels

    def __getitem__(self, ind):
        return self.data[ind],self.label[ind]

    def __len__(self):
        return len(self.data)

class DWTloader(Dataset):
    def __init__(self,args,flag):
        self.data,self.label=self.load_all(args,flag=flag)

    def load_all(self,args,flag=None):
        data_path=args.data_path
        file_paths=glob.glob(os.path.join(data_path,'*'))
        if len(file_paths)==0:
            raise Exception('No files found using {}.'.format(os.path.join(data_path,'*')))
        if flag is not None:
            file_paths=list(filter(lambda x:re.search(flag,x),file_paths))
        input_paths=[p for p in file_paths if os.path.isfile(p) and p.endswith('.tsv')]
        if len(input_paths)==0:
            raise Exception('No .tsv file found using {}.'.format(data_path))
        features,labels=self.load_single(args,input_paths[0],flag)

        return features,labels


    def nearest_guided_warping(self,x, ret_x, nan_starts, prototypes, ids, slope_constraint="asymmetric"):
        for t, s_t in enumerate(tqdm(ids)):
            dtw_dists = np.zeros(np.shape(prototypes)[0])
            nan_start = int(nan_starts[s_t])
            sample = x[s_t][:nan_start]
            if nan_start > 2. * np.shape(prototypes)[1]:
                ret_x[s_t] = np.interp(np.linspace(0, nan_start - 1, num=np.shape(prototypes)[1]), np.arange(nan_start),
                                       sample)
                print('interpolating long sample')
            else:
                for i, p_i in enumerate(prototypes):
                    dtw_dists[i] = dtw.dtw(p_i.reshape((-1, 1)), sample.reshape((-1, 1)), dtw.RETURN_VALUE,
                                           slope_constraint=slope_constraint)
                smallest_p = np.argmin(dtw_dists)
                path = dtw.dtw(prototypes[smallest_p].reshape((-1, 1)), sample.reshape((-1, 1)), dtw.RETURN_PATH,
                               slope_constraint=slope_constraint)
                ret_x[s_t] = sample[path[1]]
        return ret_x
    def load_single(self,args,file_path,flag):
        os.makedirs("../dwt", exist_ok=True)
        prototype_file="../dwt/"+args.data_name+args.way+str(args.alpha)+str(args.beta)+'.npy'
        data_save = os.path.join("../dwt/",args.data_name + args.way+str(args.alpha)+str(args.beta)+ '_' + flag + '.npy')

        df = pd.read_csv(file_path, sep="\t", header=None)
        features=df.iloc[:,1:].values
        labels=df.iloc[:,0].values
        labels=labels-args.label_min
        lengths = np.apply_along_axis(count_non_missing, 1, features)
        ret_x = np.zeros((features.shape[0], args.u_quant))
        if os.path.exists(prototype_file)==False or os.path.exists(data_save)==False:
            if flag=='TRAIN':
                prototype_ids = np.where(lengths >= args.l_quant)[0]
                ids = np.where(lengths < args.l_quant)[0]

                prototypes = np.zeros((len(prototype_ids), args.u_quant))

                for t,p_t in enumerate(prototype_ids):
                    nan_start = int(lengths[p_t])
                    prot = features[p_t][:nan_start]
                    prototypes[t] = np.interp(np.linspace(0, nan_start - 1, num=args.u_quant), np.arange(nan_start), prot)
                    ret_x[p_t] = prototypes[t]

                prototypes.tofile(prototype_file)
                args.id_len=len(prototype_ids)

            else:
                ids=np.arange(len(features))
                prototypes=np.fromfile(prototype_file).reshape((-1,args.u_quant))

            ret_x=self.nearest_guided_warping(features,ret_x,lengths,prototypes,ids,args.slope_constraint)
            ret_x.tofile(data_save)
        else:
            ret_x=np.fromfile(data_save).reshape((-1,args.u_quant))
        return ret_x,labels


    def __getitem__(self, ind):
        return self.data[ind],self.label[ind]

    def __len__(self):
        return len(self.data)


