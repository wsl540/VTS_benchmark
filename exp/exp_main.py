import wandb
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer, FCN, Inception, LSTM, Resnet, MLP, TimesNet, Informer
from utils.tools import EarlyStopping, cal_accuracy
from data_provider.preprocessing import get_content,getngw_len

import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import numpy as np

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark=True




class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'MLP': MLP,
            'LSTM': LSTM,
            'FCN': FCN,
            'Resnet': Resnet,
            'Inception': Inception,
            'Transformer': Transformer,
            'Informer': Informer,
            'TimesNet': TimesNet,
        }


        print(self.args.way)
        if 'dwt' in self.args.way:
            self.args.l_quant,self.args.u_quant,self.args.class_num,self.args.label_min=self._get_lu(flag='TRAIN')
            self.args.input_len=self.args.u_quant
        else:
            train_data,train_loader=self._get_data(flag='TRAIN')
            features,labels,mask,lengths=next(iter(train_loader))
            self.args.input_len=features.shape[1]
            self.args.class_num=len(train_data.class_names)

        print("input_len:",self.args.input_len)
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_lu(self,flag):
        x_data,class_num,label_min=get_content(self.args.data_path,flag)
        l_quant, u_quant=getngw_len(x_data,self.args.alpha,self.args.beta)
        return l_quant,u_quant,class_num,label_min
    def _get_data(self, flag):
        # print("come in")
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def _select_scheduler(self,optimizer):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,patience=10,min_lr=self.args.min_lr,verbose=True)
        return scheduler

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mark, x_lengths) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                if self.args.use_masking:
                    padding_mark = padding_mark.float().to(self.device)
                if self.args.pos != -1:
                    x_lengths = x_lengths.to(self.device)

                outputs = self.model(batch_x, padding_mark, x_lengths)
                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze(-1).cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)
        total_loss = np.average(total_loss)
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        probs=torch.nn.functional.softmax(preds,dim=1)
        predictions=torch.argmax(probs,dim=1).cpu().numpy()
        trues=trues.flatten().cpu().numpy()
        accuracy=cal_accuracy(predictions,trues)

        return total_loss,accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.date,self.args.model,self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = self._select_scheduler(model_optim)

        for epoch in range(self.args.train_epochs):
            print("learning rate:{}".format(model_optim.param_groups[0]['lr']))
            iter_count=0
            train_loss = []
            preds = []
            trues = []

            epoch_time = time.time()
            self.model.train()
            for i, (batch_x, label, padding_mask,length) in enumerate(train_loader):
                iter_count+=1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                if self.args.use_masking:
                    padding_mask=padding_mask.float().to(self.device)
                if self.args.pos!=-1:
                    length=length.to(self.device)

                outputs = self.model(batch_x, padding_mask, length)

                preds.append(outputs.detach())
                trues.append(label)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            preds = torch.cat(preds, dim=0)
            trues = torch.cat(trues, dim=0)
            probs=torch.nn.functional.softmax(preds)
            predictions=torch.argmax(probs,dim=1).cpu().numpy()
            trues=trues.flatten().cpu().numpy()
            train_acc=cal_accuracy(predictions,trues)
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            scheduler.step(train_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Train Acc:{3:.3f} Vali Loss: {4:.3f} Vali Acc: {5:.3f}".format(epoch + 1, train_steps, train_loss,train_acc, vali_loss, val_accuracy))
            early_stopping(-val_accuracy, self.model, path)



        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.date,self.args.model,'checkpoints', setting,'checkpoint.pth')))

        preds = []
        trues = []

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, label, padding_mask,length) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)
                if self.args.use_masking:
                    padding_mask = padding_mask.float().to(self.device)
                if self.args.pos!=-1:
                    length = length.to(self.device)

                outputs = self.model(batch_x, padding_mask, length)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)
        print("accuracy:{}".format(accuracy))

        # result save
        folder_path =self.args.date+'/'+self.args.model+ '/results/' + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_path=folder_path + 'result_classification.txt'
        f = open(file_path, 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return