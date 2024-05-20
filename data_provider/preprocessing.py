import math
import numpy as np
import pandas as pd
from utils.tools import count_non_missing
import glob
import os
import re


def read_file(path):
    df = pd.read_csv(path, sep="\t", header=None)
    return df

def get_content(data_path,flag='TRAIN'):
    file_path=glob.glob(os.path.join(data_path,'*'))
    file_paths=list(filter(lambda x:re.search(flag,x),file_path))
    input_path=[p for p in file_paths if os.path.isfile(p) and p.endswith('.tsv')]
    data=read_file(input_path[0])
    x_data=np.array(data.iloc[:,1:].values)
    labels=data.iloc[:,0].values
    label_min=labels.min()
    labels=labels-label_min
    class_num=len(set(labels))

    return x_data,class_num,label_min


def truncate_pre(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_= np.zeros((np.shape(x_data)[0], length))
    x_data=np.nan_to_num(x_data,nan=0.0)
    for t, x_t in enumerate(x_data):
        if lengths[t]<length:
            arr_[t]=x_t[:length]
        else:
            cur = x_t[:lengths[t]]
            arr_[t] = cur[-length:]
    return arr_

def truncate_outer(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_= np.zeros((np.shape(x_data)[0], length))
    x_data=np.nan_to_num(x_data,nan=0.0)
    for t, x_t in enumerate(x_data):
        if lengths[t]<length:
            arr_[t]=x_t[:length]
        else:
            cur = x_t[:lengths[t]]
            stat = lengths[t] // 2 - length // 2
            end = lengths[t] // 2 + math.ceil(length / 2)
            arr_[t] = cur[stat:end]
    return arr_

def truncate_post(x_data,length):
    x_data=np.nan_to_num(x_data,nan=0.0)
    arr_ = x_data[:, :length]
    return arr_

def zeropad_pre(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_ = np.zeros((np.shape(x_data)[0], length))
    for t, x_t in enumerate(x_data):
        if lengths[t]>=length:
            arr_[t] = x_t[:length]
        else:
            zeropad=np.zeros(length-lengths[t])
            arr_[t] = np.concatenate((zeropad,x_t[:lengths[t]]))
    return arr_

def zeropad_outer(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_ = np.zeros((np.shape(x_data)[0], length))
    for t, x_t in enumerate(x_data):
        if lengths[t]>=length:
            arr_[t] = x_t[:length]
        else:
            zeropad=length-lengths[t]
            padpre=np.zeros(zeropad//2)
            padpost=np.zeros(zeropad-len(padpre))
            arr_[t] = np.concatenate((padpre,x_t[:lengths[t]],padpost))
    return arr_

def zeropad_mid(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_ = np.zeros((np.shape(x_data)[0], length))
    for t, x_t in enumerate(x_data):
        if lengths[t]>=length:
            arr_[t] = x_t[:length]
        else:
            mid=lengths[t]//2
            zeropad=np.zeros(length-lengths[t])
            arr_[t] = np.concatenate((x_t[:mid],zeropad,x_t[mid:lengths[t]]))
    return arr_

def zeropad_post(x_data,length):
    arr_ = np.zeros((np.shape(x_data)[0], length))
    x_data=np.nan_to_num(x_data,nan=0.0)
    for t, x_t in enumerate(x_data):
        if len(x_t)>=length:
            arr_[t] = x_t[:length]
        else:
            arr_[t,:len(x_t)] = x_t
    return arr_

def noisepad_pre(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_ = np.zeros((np.shape(x_data)[0], length))
    for t, x_t in enumerate(x_data):
        if lengths[t]>=length:
            arr_[t] = x_t[:length]
        else:
            noise = np.random.uniform(0, 0.001, length - lengths[t])
            arr_[t] = np.concatenate((noise, x_t[:lengths[t]]))
    return arr_


def noisepad_outer(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_ = np.zeros((np.shape(x_data)[0], length))
    for t, x_t in enumerate(x_data):
        if lengths[t]>=length:
            arr_[t] = x_t[:length]
        else:
            pad=length-lengths[t]
            noise_pre = np.random.uniform(0, 0.001, pad//2)
            noise_post = np.random.uniform(0, 0.001, pad - len(noise_pre))
            arr_[t] = np.concatenate((noise_pre, x_t[:lengths[t]], noise_post))
    return arr_

def noisepad_post(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_ = np.zeros((np.shape(x_data)[0], length))
    for t, x_t in enumerate(x_data):
        if lengths[t]>=length:
            arr_[t] = x_t[:length]
        else:
            noise = np.random.uniform(0, 0.001, length - lengths[t])
            arr_[t] = np.concatenate((x_t[:lengths[t]], noise))
    return arr_

def edgepad_pre(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_ = np.zeros((np.shape(x_data)[0], length))
    for t, x_t in enumerate(x_data):
        if lengths[t]>=length:
            arr_[t] = x_t[:length]
        else:
            pad = np.ones(length - lengths[t]) * x_t[0]
            arr_[t] = np.concatenate((pad, x_t[:lengths[t]]))
    return arr_

def edgepad_outer(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_ = np.zeros((np.shape(x_data)[0], length))
    for t, x_t in enumerate(x_data):
        if lengths[t]>=length:
            arr_[t] = x_t[:length]
        else:
            pad = length - lengths[t]
            pad_pre = np.ones(pad // 2) * x_t[0]
            pad_post = np.ones(pad - len(pad_pre)) * x_t[lengths[t]-1]
            arr_[t] = np.concatenate((pad_pre, x_t[:lengths[t]], pad_post))
    return arr_

def edgepad_post(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_ = np.zeros((np.shape(x_data)[0], length))
    for t, x_t in enumerate(x_data):
        if lengths[t]>=length:
            arr_[t] = x_t[:length]
        else:
            pad = np.ones(length - lengths[t]) * x_t[lengths[t]-1]
            arr_[t] = np.concatenate((x_t[:lengths[t]], pad))
    return arr_

def interpolate(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_ = np.zeros((np.shape(x_data)[0], length))
    for t, x_t in enumerate(x_data):
        if lengths[t]>=length:
            arr_[t] = x_t[:length]
        else:
            arr_[t] = np.interp(np.linspace(0, lengths[t]-1, length), np.arange(lengths[t]), x_t[:lengths[t]])
    return arr_

def strf_pad(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_ = np.zeros((np.shape(x_data)[0], length))
    for t, x_t in enumerate(x_data):
        if lengths[t]>=length:
            arr_[t] = x_t[:length]
        else:
            pad=length-lengths[t]
            arr_[t] = np.insert(x_t[:lengths[t]], np.linspace(1, lengths[t]-1, num=pad).astype(int), 0)
    return arr_

def random_pad(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_ = np.zeros((np.shape(x_data)[0], length))
    for t, x_t in enumerate(x_data):
        if lengths[t]>=length:
            arr_[t] = x_t[:length]
        else:
            pad=length-lengths[t]
            arr_[t] = np.insert(x_t[:lengths[t]], np.random.randint(0, lengths[t]-1, size=pad), 0)
    return arr_

def zoom_pad(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    arr_ = np.zeros((np.shape(x_data)[0], length))
    for t, x_t in enumerate(x_data):
        if lengths[t]>=length:
            arr_[t] = x_t[:length]
        else:
            cur=x_t[:lengths[t]]
            arr_[t] = cur[np.linspace(0, lengths[t]-1, num=length).astype(int)]
    return arr_

def s_fourselect(x_data,length):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    length=length if length%2==0 else length-1
    N=length//3
    arr_ = np.zeros((np.shape(x_data)[0], N*2+2))
    x_data=np.nan_to_num(x_data,nan=0.0)
    for t, x_t in enumerate(x_data):
        if lengths[t]<=length:
            cur=x_t[:length]
            four = np.fft.fft(cur, axis=-1, norm='backward')
            phase=np.angle(four)
            amp=np.abs(four)
            amp_m=amp[length//2].reshape(-1)
            phase_m=phase[length//2].reshape(-1)
            amp_map=np.concatenate((amp_m,amp[:N]),axis=-1)
            phase_map=np.concatenate((phase_m,phase[:N]),axis=-1)
            arr_[t] = np.concatenate((amp_map,phase_map),axis=-1)
        else:
            cur = x_t[:lengths[t]]
            delet=lengths[t]%length
            gap=lengths[t]//length
            delet_start=delet//2
            delet_end=delet-delet_start
            if delet!=0:
                cur=cur[delet_start:-delet_end]
            four = np.abs(np.fft.fft(cur, axis=-1, norm='backward'))
            phase=np.angle(four)
            amp=np.abs(four)
            index=np.arange(0,len(four),gap)
            amp_m=amp[cur.shape[0]//2].reshape(-1)
            phase_m=phase[cur.shape[0]//2].reshape(-1)
            amp_map=np.concatenate((amp_m,amp[index][:N]),axis=-1)
            phase_map=np.concatenate((phase_m,phase[index][:N]),axis=-1)
            arr_[t] = np.concatenate((amp_map,phase_map),axis=-1)
    return arr_

def spectral(x_data,length,norm='backward'):
    lengths = np.apply_along_axis(count_non_missing, 1, x_data)
    if length % 2 != 0: length -= 1
    need=length//2+1
    arr_ = np.zeros((np.shape(x_data)[0], length))
    for t, x_t in enumerate(x_data):
        cur=x_t[:lengths[t]]
        if lengths[t]<length:
            pad = length - lengths[t]
            cur=np.pad(cur,(0,pad),'constant')
        four=np.fft.rfft(cur, axis=-1, norm=norm)
        arr_[t]=np.fft.irfft(four[:need],axis=-1,norm=norm)
    return arr_

def getngw_len(x_data_train,alpha,beta=0):
    train_lengths = np.apply_along_axis(count_non_missing, 1, x_data_train)
    lower_quantile = alpha
    upper_quantile =beta if beta!=0 else 1. - (1. - alpha) / 2.
    l_quant = int(np.quantile(train_lengths, lower_quantile))
    u_quant = int(np.quantile(train_lengths, upper_quantile))
    return l_quant,u_quant

def predeal(data,length,way,norm='backward'):
    features=None
    if way=='truncate_pre':
        features=truncate_pre(data,length)
    elif way=='truncate_outer':
        features=truncate_outer(data,length)
    elif way=='truncate_post':
        features=truncate_post(data,length)
    elif way=='zeropad_pre':
        features=zeropad_pre(data,length)
    elif way=='zeropad_outer':
        features=zeropad_outer(data,length)
    elif way=='zeropad_mid':
        features=zeropad_mid(data,length)
    elif way=='zeropad_post':
        features=zeropad_post(data,length)
    elif way=='noisepad_pre':
        features=noisepad_pre(data,length)
    elif way=='noisepad_outer':
        features=noisepad_outer(data,length)
    elif way=='noisepad_post':
        features=noisepad_post(data,length)
    elif way=='edgepad_pre':
        features=edgepad_pre(data,length)
    elif way=='edgepad_outer':
        features=edgepad_outer(data,length)
    elif way=='edgepad_post':
        features=edgepad_post(data,length)
    elif way=='interpolate':
        features=interpolate(data,length)
    elif way=='strf_pad':
        features=strf_pad(data,length)
    elif way=='random_pad':
        features=random_pad(data,length)
    elif way=='zoom_pad':
        features=zoom_pad(data,length)
    elif way=='s_fourselect':
        features=s_fourselect(data,length)
    elif way=='spectral':
        features=spectral(data,length,norm=norm)
    else:
        print("wrong way")
    return features


