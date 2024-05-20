import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_Enc import Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.others import Nothing


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, config):
        super(Model, self).__init__()
        self.pos = config.pos
        self.pooling_mode = config.pooling_mode
        self.enc_embedding = DataEmbedding(config.c_in, config.d_model, config.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False,  attention_dropout=config.dropout,
                                      output_attention=None), config.d_model, config.n_heads),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                ) for l in range(config.e_layers)
            ],
            pooling_output=config.pooling_output,
            pos=self.pos,
            norm_layer=torch.nn.LayerNorm(config.d_model),
            pooling_mode = config.pooling_mode,
            norm_fft=config.norm,
            trans=config.trans
        )
        self.gpooling = nn.AdaptiveAvgPool1d(1)

        self.act=nn.GELU()
        self.dropout =Nothing() if config.dropout==0.0 else nn.Dropout(config.dropout)
        self.projection = nn.Linear(config.d_model, config.class_num)

    def forward(self, x,mask,length):
        x=x if len(x.shape)==3 else x.unsqueeze(-1)
        x = self.enc_embedding(x)

        enc_out,_ = self.encoder(x,attn_mask=None,length=length)
        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.permute(0, 2, 1)
        if mask != None:
            output = output * mask.unsqueeze(1)
        output = self.gpooling(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, class_numes)
        return output  # [B, N]