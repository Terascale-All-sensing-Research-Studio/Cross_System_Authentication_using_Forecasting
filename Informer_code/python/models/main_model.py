import torch
import torch.nn as nn
import torch.nn.functional as F

from models.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.informer import Informer

from models.MJ_FCN import FCN as mj_FCN
from models.MJ_transformer import Encoder as mj_endoder


class exp3_model(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
        num_feats=4, num_class=2, classification_model="TF",
        d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=2048, 
        dropout=0.0, activation='gelu', output_attention = True,
        device=torch.device('cuda:0')
    ):
        super(exp3_model, self).__init__()
        self.classification_model = classification_model

        self.informer = Informer(
            enc_in, dec_in, c_out, seq_len, label_len, out_len, 
            factor=5, d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_layers=d_layers, d_ff=d_ff, 
            dropout=0.0, attn='full', embed='timeF', freq='h', activation='gelu', 
            output_attention=True, distil=False, mix=False,
            device=device
        ).float()

        self.mj_FCN = mj_FCN(
            data_len = seq_len + out_len, 
            num_features=num_feats,
            num_class=num_class
        )

        self.transformer = mj_endoder(
            n_head=n_heads,
            d_k=64,
            d_v=64,
            seq_len=seq_len + out_len,
            num_features=num_feats, 
            d_model=d_model, 
            d_ff=d_ff,
            n_layers=e_layers,
            n_class=num_class,
            dropout=0.1
        )


    def forward(self, en_in, en_time, de_in, de_time, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        forecasting_output, attn_map = self.informer(en_in, en_time, de_in, de_time)

        forecasting_input = torch.concat((en_in, forecasting_output), dim=1)
        if self.classification_model == "FCN":
            classification_output = self.mj_FCN(forecasting_input)
        elif self.classification_model == "TF":
            classification_output = self.transformer(forecasting_input)
        else:
            raise ValueError("classification_model must be FCN or TF")

        return attn_map, forecasting_output, classification_output


class V3_Forecast_Model(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
        num_feats=4, d_model=512, n_heads=8, e_layers=3,
        d_layers=2, d_ff=2048, dropout=0.0, activation='gelu', 
        output_attention=True, device=torch.device('cuda:0')
    ):
        super(V3_Forecast_Model, self).__init__()

        self.informer = Informer(
            enc_in, dec_in, c_out, seq_len, label_len, out_len, 
            factor=5, d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_layers=d_layers, d_ff=d_ff, 
            dropout=0.0, attn='full', embed='timeF', freq='h', activation='gelu', 
            output_attention=True, distil=False, mix=False,
            device=device
        ).float()


    def forward(self, en_in, en_time, de_in, de_time, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        forecasting_output, attn_map = self.informer(en_in, en_time, de_in, de_time)

        return attn_map, forecasting_output


if __name__ == '__main__':
    model = exp3_model(
        enc_in=8, dec_in=8, c_out=8, seq_len=96, label_len=48, out_len=48,
        num_feats=8, num_class=2,
        d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
        dropout=0.0, activation='gelu', output_attention = True,
        device=torch.device('cuda:0')
    )

    en_in = torch.randn(1, 96, 8)
    en_time = torch.randn(1, 96, 1)
    de_in = torch.randn(1, 48, 8)
    de_time = torch.randn(1, 48, 1)

    attn_map, forecasting_output, classification_output = model(en_in, en_time, de_in, de_time)

    print(len(attn_map))                    # forecasting attention map
    print(forecasting_output.shape)         # forecasting output
    print(classification_output[0].shape)   # classification output
    print(len(classification_output[1]))    # classification attention map
