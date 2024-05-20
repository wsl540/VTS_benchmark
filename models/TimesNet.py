import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from layers.Avgpooling_no import Avgpooling
from layers.Maxpooling_no import Maxpooling
from layers.others import Nothing
from layers.Spectralpooling import Spectralpooling


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.ks),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.ks)
        )


    def forward(self, x):
        B, T, N = x.size()
        self.seq_len = T
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len) % period != 0:
                length = (((self.seq_len ) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len )
                out = x
            # reshape
            if period==0:period=1
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # self.configs = configs
        # self.seq_len = configs.seq_len
        self.pos=configs.pos
        # self.pooling_output=configs.pooling_output
        self.pooling_mode=configs.pooling_mode
        # self.norm=configs.norm

        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.c_in, configs.d_model, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.gpl = nn.AdaptiveMaxPool1d(1)
        # self.gpl = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(configs.d_model, configs.class_num)
        self.pooling_1 =Nothing()
        if self.pooling_mode == "avg":
            self.pooling_1 = Avgpooling(configs.pooling_output,trans=configs.trans)
        elif self.pooling_mode == "max":
            self.pooling_1 =Maxpooling(configs.pooling_output,trans=configs.trans)
        elif self.pooling_mode == "sp":
            self.pooling_1=Spectralpooling(configs.pooling_output,norm=configs.norm,trans=configs.trans)
        self.timepooling = Nothing() if self.pos == -1 else self.pooling_1

    def classification(self, x, mask,length):
        # embedding
        x = x if len(x.shape) == 3 else x.unsqueeze(-1)
        x = self.enc_embedding(x)
        if self.pos!=-1:
            self.timepooling.set_length(length)
         # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            x = self.layer_norm(self.model[i](x))
            if self.pos==i+1:
                x=self.timepooling(x)

        output = self.act(x)
        output = self.dropout(output)
        output = output.permute(0, 2, 1)

        if mask!=None:
            output=output*mask.unsqueeze(1)

        output = self.gpl(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mask_enc,length):
        dec_out = self.classification(x_enc,x_mask_enc,length)
        return dec_out  # [B, N]