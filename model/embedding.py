import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from model.attn_layer import (PositionalEmbedding,
                              AttentionLayer,
                              complex_dropout,
                              complex_operator)
from model.Conv_Blocks import Inception_Block
from model.multi_attention_blocks import Inception_Attention_Block

from model.RevIN import RevIN

class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        # d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attn_layer = attn
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        """
        x : N x L x C(=d_model)
        """

        out, attn = self.attn_layer(x)
        y = complex_dropout(self.dropout, out)

        return y


# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        """
        x : N x L x C(=d_model)
        """

        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


class TokenEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, n_window=100, n_layers=1, branch_layers=['fc_linear', 'intra_fc_transformer'],
                 group_embedding='False', match_dimension='first', kernel_size=[5], multiscale_patch_size=[10, 20],
                 init_type='normal', gain=0.02, dropout=0.1):
        super(TokenEmbedding, self).__init__()

        self.window_size = n_window
        self.d_model = d_model
        self.n_layers = n_layers
        self.branch_layers = branch_layers
        self.group_embedding = group_embedding
        self.match_dimension = match_dimension
        self.kernel_size = kernel_size
        self.multiscale_patch_size = multiscale_patch_size

        # For the input is data in the frequency domain, n_network is two for real and imagery
        component_network = ['real_part', 'imaginary_part']
        num_in_fc_networks = len(component_network)

        self.encoder_layers = nn.ModuleList([])
        self.norm_layers = nn.ModuleList([])

        for i, e_layer in enumerate(branch_layers):
            if self.match_dimension == 'none':
                updated_in_dim = in_dim
                extended_dim = in_dim
            elif (i == 0 and self.match_dimension == 'first') or (len(branch_layers) < 2):
                updated_in_dim = in_dim
                extended_dim = d_model
            elif (i == 0) and (not self.match_dimension == 'first'):
                updated_in_dim = in_dim
                extended_dim = in_dim
            elif (i + 1 < len(branch_layers)) and (self.match_dimension == 'middle'):
                updated_in_dim = extended_dim
                extended_dim = d_model
            elif i + 1 == len(branch_layers):
                updated_in_dim = extended_dim
                extended_dim = d_model
            else:
                updated_in_dim = extended_dim
                extended_dim = extended_dim

            if 'conv1d' in e_layer or 'deconv1d' in e_layer:
                if self.group_embedding == 'False':
                    groups = 1
                else:
                    if extended_dim >= updated_in_dim and extended_dim % updated_in_dim == 0:
                        groups = updated_in_dim
                    elif extended_dim < updated_in_dim and updated_in_dim % extended_dim == 0:
                        groups = extended_dim
                    else:
                        print(f"The conv1d/deconv1d layer {i} of encoder is non-grouped convolution!")
                        groups = 1

            if e_layer == 'dropout':
                self.encoder_layers.append(nn.Dropout(p=dropout))
                self.norm_layers.append(nn.Identity())
            elif e_layer == 'fc_linear':
                self.encoder_layers.append(nn.ModuleList([nn.Linear(updated_in_dim, extended_dim, bias=False)
                                                         for _ in range(num_in_fc_networks)])
                                           )
                self.norm_layers.append(nn.ModuleList([nn.LayerNorm(extended_dim) for _ in range(num_in_fc_networks)]))
            elif e_layer == 'linear':
                self.encoder_layers.append(nn.ModuleList([nn.Linear(updated_in_dim, extended_dim, bias=False)
                                                          for _ in range(num_in_fc_networks)]))
                self.norm_layers.append(nn.ModuleList([nn.LayerNorm(extended_dim) for _ in range(num_in_fc_networks)]))
            elif e_layer == 'multiscale_conv1d':
                for _ in range(n_layers):
                    self.encoder_layers.append(Inception_Block(in_channels=updated_in_dim,
                                                               out_channels=extended_dim,
                                                               kernel_list=kernel_size,
                                                               groups=groups
                                                               )
                                                   )
                    self.norm_layers.append(nn.ModuleList([nn.LayerNorm(self.window_size)
                                                           for _ in range(num_in_fc_networks)]))
            elif e_layer == 'inter_fc_transformer':
                w_model = self.window_size // 2 + 1
                attention_layer = AttentionLayer(w_size=extended_dim, d_model=w_model, n_heads=1, dropout=dropout)
                self.encoder_layers.append(nn.ModuleList([EncoderLayer(attn=attention_layer,
                                                                       d_model=w_model,
                                                                       d_ff=128,
                                                                       dropout=dropout,
                                                                       activation='gelu'
                                                                       )
                                                          for _ in range(num_in_fc_networks)])
                                           )
                self.norm_layers.append(nn.ModuleList([nn.LayerNorm(self.window_size)
                                                       for _ in range(num_in_fc_networks)]))

            elif e_layer == 'intra_fc_transformer':
                w_model = self.window_size // 2 + 1
                attention_layer = AttentionLayer(w_size=w_model, d_model=extended_dim, n_heads=1, dropout=dropout)
                self.encoder_layers.append(nn.ModuleList([EncoderLayer(attn=attention_layer,
                                                                       d_model=extended_dim,
                                                                       d_ff=128,
                                                                       dropout=dropout,
                                                                       activation='gelu'
                                                                       )
                                                          for _ in range(num_in_fc_networks)])
                                           )
                self.norm_layers.append(nn.ModuleList([nn.LayerNorm(self.window_size)
                                                       for _ in range(num_in_fc_networks)]))

            elif e_layer == 'multiscale_ts_attention':
                self.encoder_layers.append(Inception_Attention_Block(w_size=self.window_size,
                                                                     in_dim=extended_dim,
                                                                     d_model=extended_dim,
                                                                     patch_list=multiscale_patch_size))
                # self.norm_layers.append(nn.LayerNorm(extended_dim))
                self.norm_layers.append(nn.Identity())
            else:
                raise ValueError(f'The specified model {e_layer} is not supported!')

        self.dropout = nn.Dropout(p=dropout)
        self.criterion = nn.MSELoss(reduction='none')
        self.activation = nn.GELU()
        # self.activation = nn.LeakyReLU(0.2, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)

                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):

                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        B, L, C = x.size()

        latent_list = []

        residual = None

        amplitudeRevIN = RevIN(int(L//2 + 1))

        for i, (embedding_layer, norm_layer) in enumerate(zip(self.encoder_layers, self.norm_layers)):
            if self.branch_layers[i] not in ['linear', 'fc_linear', 'multiscale_ts_attention']:
                x = x.permute(0, 2, 1)

            if self.branch_layers[i] == 'multiscale_conv1d':
                x = complex_operator(embedding_layer, x)
            elif self.branch_layers[i] == 'multiscale_ts_attention':
                x = complex_operator(embedding_layer, x)
            elif self.branch_layers[i] in ['fc_linear']:
                x = torch.fft.rfft(x, dim=-2)
                x = complex_operator(embedding_layer, x)
                x = torch.fft.irfft(x, dim=-2)
            elif self.branch_layers[i] in ['inter_fc_transformer']:
                x = torch.fft.rfft(x, dim=-1)
                x = complex_operator(embedding_layer, x)
                x = torch.fft.irfft(x, dim=-1)
            elif self.branch_layers[i] in ['intra_fc_transformer']:
                x = torch.fft.rfft(x, dim=-1)
                x = x.permute(0, 2, 1)
                x = complex_operator(embedding_layer, x)
                x = x.permute(0, 2, 1)
                x = torch.fft.irfft(x, dim=-1)
            else:
                x = complex_operator(embedding_layer, x)

            x = complex_operator(norm_layer, x)

            # x = self.activation(x)
            # x = self.dropout(x)
            if self.branch_layers[i] not in ['linear', 'fc_linear', 'multiscale_ts_attention']:
                x = x.permute(0, 2, 1)

            latent_list.append(x)

            # After each transformer layer, a residual connection is used
            if residual is not None:
                if x.shape == residual.shape and 'transformer' in self.branch_layers[i]:
                    x += residual

            if self.branch_layers[i] in ['linear', 'fc_linear']:
                residual = x

        return x, latent_list

class InputEmbedding(nn.Module):
    def __init__(self, in_dim, d_model, n_window, device, dropout=0.1, n_layers=1, use_pos_embedding='False',
                 group_embedding='False', kernel_size=5, init_type='kaiming', match_dimension='first',  branch_layers=['linear']):
        super(InputEmbedding, self).__init__()
        self.device = device
        self.token_embedding = TokenEmbedding(in_dim=in_dim, d_model=d_model, n_window=n_window,
                                              n_layers=n_layers, branch_layers=branch_layers,
                                              group_embedding=group_embedding, match_dimension=match_dimension,
                                              init_type=init_type, kernel_size=kernel_size,
                                              dropout=0.1)
        self.pos_embedding = PositionalEmbedding(d_model=d_model)
        self.use_pos_embedding = use_pos_embedding
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.to(self.device)

        x, latent_list = self.token_embedding(x)

        if self.use_pos_embedding == 'True':
            x = x + self.pos_embedding(x).to(self.device)

        return self.dropout(x), latent_list

