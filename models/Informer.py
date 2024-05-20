import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_Enc import Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention import ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    def __init__(self, configs):
        super(Model, self).__init__()


        # Embedding
        self.enc_embedding = DataEmbedding(configs.c_in, configs.d_model, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor=1, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                ) for l in range(configs.e_layers)
            ],
            pooling_output=configs.pooling_output,
            pos=configs.pos,
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            pooling_mode=configs.pooling_mode,
            norm_fft=configs.norm,
            trans=configs.trans
        )
        # Decoder
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.gpooling = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(configs.d_model, configs.class_num)

    def classification(self, x, mask,length):
        # enc
        enc_out = self.enc_embedding(x)
        enc_out, _ = self.encoder(enc_out, attn_mask=None,length=length)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output=output.permute(0, 2, 1)

        if mask != None:
            output = output * mask.unsqueeze(1)
        output = self.gpooling(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x, mask,length):
        x = x if len(x.shape) == 3 else x.unsqueeze(-1)
        dec_out = self.classification(x, mask,length)
        return dec_out  # [B, N]
