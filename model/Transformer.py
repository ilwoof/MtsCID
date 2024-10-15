from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

from model.embedding import InputEmbedding
from model.loss_functions import sce_loss

from utils.dataplot import plot_time_series_comparison
from model.RevIN import RevIN


class Decoder(nn.Module):
    def __init__(self, w_size, d_model, c_out, networks=['linear'], n_layers=1,
                 group_embedding='False', kernel_size=[1], patch_size=-1, activation='gelu', dropout=0.0, device='cpu'):
        super(Decoder, self).__init__()

        self.decoder = InputEmbedding(in_dim=d_model, d_model=c_out, n_window=w_size,
                                      dropout=dropout, n_layers=n_layers,
                                      branch_layers=networks,
                                      match_dimension='last',
                                      group_embedding=group_embedding,
                                      kernel_size=kernel_size, init_type='normal',
                                      device=device)

    def forward(self, x):
        """
        x : N x L x C(=d_model)
        """
        out = self.decoder(x)
        return out  # N x L x c_out


class TransformerVar(nn.Module):

    DEFAULTS = {}

    def __init__(self, config, n_heads=1, d_ff=128, dropout=0.3, activation='gelu', gain=0.02):
        super(TransformerVar, self).__init__()

        self.__dict__.update(TransformerVar.DEFAULTS, **config)

        # Encoding
        branch1_group = self.branches_group_embedding.split('_')[0]
        branch2_group = self.branches_group_embedding.split('_')[1]

        branch1_dim = self.input_c if self.branch1_match_dimension == 'none' else self.d_model
        branch2_dim = self.input_c if self.branch2_match_dimension == 'none' else self.d_model

        self.encoder_branch1 = InputEmbedding(in_dim=self.input_c, d_model=branch1_dim, n_window=self.win_size,
                                              dropout=dropout, n_layers=self.encoder_layers,
                                              branch_layers=self.branch1_networks,
                                              match_dimension=self.branch1_match_dimension,
                                              group_embedding=branch1_group,
                                              kernel_size=self.multiscale_kernel_size, init_type=self.embedding_init,
                                              device=self.device)

        self.encoder_branch2 = InputEmbedding(in_dim=self.input_c, d_model=branch2_dim, n_window=self.win_size,
                                              dropout=dropout, n_layers=self.encoder_layers,
                                              branch_layers=self.branch2_networks,
                                              match_dimension=self.branch2_match_dimension,
                                              group_embedding=branch2_group,
                                              kernel_size=self.multiscale_kernel_size,
                                              init_type=self.embedding_init, device=self.device)

        self.activate_func = nn.GELU()

        self.dropout = nn.AlphaDropout(p=dropout)

        self.loss_func = nn.MSELoss(reduction='none')

        self.mem_R, self.mem_I = create_memory_matrix(N=branch2_dim,
                                                      L=self.win_size,
                                                      mem_type=self.memory_guided,
                                                      option='options2')

        branch1_out_dim = self.output_c if self.branch1_match_dimension == 'none' else self.d_model

        model_dim = branch1_out_dim

        self.weak_decoder = Decoder(w_size=self.win_size,
                                    d_model=model_dim,
                                    c_out=self.output_c,
                                    networks=self.decoder_networks,
                                    n_layers=self.decoder_layers,
                                    group_embedding=self.decoder_group_embedding,
                                    kernel_size=self.multiscale_kernel_size,
                                    activation='gelu',
                                    dropout=0.0,       # The dropout in decoder is set as zero
                                    device=self.device)

        if self.branch1_match_dimension == 'none':
            self.feature_prj = lambda x: x
        else:
            self.feature_prj = nn.Linear(branch1_out_dim, self.output_c)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.embedding_init == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif self.embedding_init == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif self.embedding_init == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif self.embedding_init == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)

                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                if self.embedding_init == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif self.embedding_init == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif self.embedding_init == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif self.embedding_init == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                else:
                    torch.nn.init.uniform_(m.weight.data, a=-0.5, b=0.5)

                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input_data, mode='train'):
        """
        x (input time window) : B x L x enc_in
        """

        z1 = z2 = input_data

        t_query, t_latent_list = self.encoder_branch1(z1)

        i_query, _ = self.encoder_branch2(z2)

        # use dot production with static sinusoid basis
        mem = self.mem_R.T.to(self.device)
        # differencing_q = (i_query - torch.roll(i_query, shifts=1, dims=-2))
        # It seems that using differencing is better than using i_query
        attn = torch.einsum('blf,jl->bfj', i_query, self.mem_R.to(self.device).detach())
        attn = torch.softmax(attn / self.temperature, dim=-1)

        queries = i_query

        combined_z = t_query

        combined_z = self.feature_prj(combined_z)

        out, _ = self.weak_decoder(combined_z)

        return {"out": out, "queries": queries, "mem": mem, "attn": attn}

    def get_attn_score(self, query, key, scale=None):
        """
        Calculating attention score with sparsity regularization
        query (initial features) : (NxL) x C or N x C -> T x C
        key (memory items): M x C
        """
        scale = 1. / sqrt(query.size(-1)) if scale is None else 1. / scale

        attn = torch.matmul(query, torch.t(key.to(self.device)))  # (TxC) x (CxM) -> TxM

        attn = attn * scale

        # attn = F.softmax(attn / self.temperature, dim=-1)
        # attn = torch.einsum('tl,kfl->tkf', query, key.to(self.device))  # (TxC) x (CxM) -> TxM
        # attn = attn.max(dim=1)[0]

        return attn

def generate_rolling_matrix(input_matrix):
    F, L = input_matrix.size()
    # Initialize an empty tensor of shape [L, F, L] to store the result
    output_matrix = torch.empty(L, F, L)

    # Iterate over each step from 0 to L-1
    for step in range(L):
        # Roll the rows of the input tensor along the last dimension
        rolled_matrix = input_matrix.roll(shifts=step, dims=1)
        # Assign the rolled tensor to the appropriate slice in the output tensor
        output_matrix[step] = rolled_matrix

    return output_matrix

def create_memory_matrix(N, L, mem_type='sinusoid', option='option1'):

    with torch.no_grad():
        if mem_type  == 'sinusoid' or mem_type  == 'cosine_only':
            row_indices = torch.arange(N).reshape(-1, 1)
            col_indices = torch.arange(L)
            grid = row_indices * col_indices
            # Calculate the period values using the grid
            init_matrix_r = torch.cos((1 / L) * 2 * torch.tensor([torch.pi]) * grid)
            init_matrix_i = torch.sin((1 / L) * 2 * torch.tensor([torch.pi]) * grid)
        elif mem_type  == 'uniform' or mem_type  == 'uniform_only':
            init_matrix_r = torch.rand((N, L), dtype=torch.float)
            init_matrix_i = torch.rand((N, L), dtype=torch.float)
        elif mem_type  == 'orthogonal_uniform' or mem_type  == 'orthogonal_uniform_only':
            init_matrix_r = torch.nn.init.orthogonal_(torch.rand((N, L), dtype=torch.float))
            init_matrix_i = torch.nn.init.orthogonal_(torch.rand((N, L), dtype=torch.float))
        elif mem_type  == 'normal' or mem_type  == 'normal_only':
            init_matrix_r = torch.randn((N, L), dtype=torch.float)
            init_matrix_i = torch.randn((N, L), dtype=torch.float)
        elif mem_type  == 'orthogonal_normal' or mem_type  == 'orthogonal_normal_only':
            init_matrix_r = torch.nn.init.orthogonal_(torch.randn((N, L), dtype=torch.float))
            init_matrix_i = torch.nn.init.orthogonal_(torch.randn((N, L), dtype=torch.float))

        # rolling the wave
        if option == 'option4':
            init_matrix_r = generate_rolling_matrix(init_matrix_r)
            init_matrix_i = generate_rolling_matrix(init_matrix_i)

        if 'only' not in mem_type:
            return init_matrix_r, init_matrix_i
        else:
            return init_matrix_r, torch.zeros_like(init_matrix_r)
