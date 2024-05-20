import argparse
import os
import time

import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 1024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--data_name', type=str, required=True, help='data name')
    parser.add_argument('--model', type=str, required=True, default='FCN',
                        help='model name, options: [MLP,LSTM,FCN, Resnet,Inception,Transformer,Informer,TimesNet]')
    parser.add_argument('--use_masking',action="store_true",default=False)
    parser.add_argument('--mode',type=str,default='max')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='VARY_UCR', help='dataset type')
    parser.add_argument('--data_path', type=str, default='./data/AllGestureWiimoteX/', help='root path of the data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--way',type=str,default='zeropad_post',help='way of preprocessing')
    parser.add_argument('--date', type=str, default='20231107', help='date of the experiment')
    parser.add_argument('--pooling_mode',type=str,default='max',help='pooling mode')
    parser.add_argument('--norm',type=str,default='backward',help='fft norm')
    parser.add_argument('--top_k',type=int,default=3,help='k for timesnet')

    parser.add_argument('--seq_len', type=int, default=0, help='input sequence length')
    parser.add_argument('--class_num',type=int,default=0)
    parser.add_argument('--input_len', type=int, default=0,help='model input length')
    parser.add_argument('--factor',type=float,default=0.5,help='factor of the learning rate')
    parser.add_argument('--lr_patience',type=int,default=10,help='patience of the learning rate')
    parser.add_argument('--min_lr',type=float,default=1e-5)
    parser.add_argument('--label_min',type=int,default=0)

    parser.add_argument('--lpatience',type=int,default=50,help='lpatience of the learning rate')

    # model define
    parser.add_argument('--pos',type=int,default=-1)
    parser.add_argument('--pooling_output',type=int,default=0)
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--c_in',type=int,default=1,help='input channel')
    parser.add_argument('--c_out', nargs='+', default=[128,256,128],help='output channel of FCN')
    parser.add_argument('--ksize',nargs='+',default=[7,5,3],help='kernel size of FCN or Resnet')
    parser.add_argument('--bn',action="store_true",default=False,help='batch normalization')
    parser.add_argument('--nf',type=int,default=32,help='output channel of Inception or Resnet')
    parser.add_argument('--depth',type=int,default=6,help='for Inception')
    parser.add_argument('--ks',type=int,default=40,help='for Inception')
    parser.add_argument('--hidden_size',type=int,default=128,help='for LSTM')
    parser.add_argument('--trans',action="store_true",default=False,help='for pooling')

    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of transformer')

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    parser.add_argument('--l_quant',type=int,default=0,help='for dwt')
    parser.add_argument('--u_quant',type=int,default=0,help='for dwt')
    parser.add_argument('--alpha',type=float,default=0.4,help='for dwt')
    parser.add_argument('--beta',type=float,default=0.7,help='for dwt')
    parser.add_argument('--id_len',type=int,default=0,help='for dwt')
    parser.add_argument('--slope_constraint',type=str,default="asymmetric",help='for dwt')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.data_name,
                args.model,
                args.way,
                args.use_masking,
                args.seq_len,
                args.pos,
                args.date,
                args.pooling_output,
                args.pooling_mode,
            )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            args.data_name,
            args.model,
            args.way,
            args.use_masking,
            args.seq_len,
            args.pos,
            args.date,
            args.pooling_output,
            args.pooling_mode,
        )
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()