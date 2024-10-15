import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os

def complex_operator(net_layer, x):
    if not torch.is_complex(x):
        return net_layer[0](x) if isinstance(net_layer, nn.ModuleList) else net_layer(x)
    else:
        if isinstance(net_layer[0], nn.LSTM):
            return torch.complex(net_layer[0](x.real)[0], net_layer[1](x.imag)[0]) if isinstance(net_layer, nn.ModuleList) else torch.complex(net_layer(x.real)[0], net_layer(x.imag)[0])
        else:
            return torch.complex(net_layer[0](x.real), net_layer[1](x.imag)) if isinstance(net_layer, nn.ModuleList) else torch.complex(net_layer(x.real), net_layer(x.imag))

def complex_einsum(order, x, y):
    x_flag = True
    y_flag = True
    if not torch.is_complex(x):
        x_flag = False
        x = torch.complex(x, torch.zeros_like(x).to(x.device))
    if not torch.is_complex(y):
        y_flag = False
        y = torch.complex(y, torch.zeros_like(y).to(y.device))
    if x_flag or y_flag:
        return torch.complex(torch.einsum(order, x.real, y.real) - torch.einsum(order, x.imag, y.imag),
                             torch.einsum(order, x.real, y.imag) + torch.einsum(order, x.imag, y.real))
    else:
        return torch.einsum(order, x.real, y.real)

def complex_softmax(x, dim=-1):
    if not torch.is_complex(x):
        return torch.softmax(x, dim=dim)
    else:
        return torch.complex(torch.softmax(x.real, dim=dim), torch.softmax(x.imag, dim=dim))

def complex_dropout(dropout_func, x):
    if not torch.is_complex(x):
        return dropout_func(x)
    else:
        # return torch.complex(dropout_func(x.real), dropout_func(x.imag))
        return torch.complex(x.real, x.imag)

def complex_layernorm(norm_func, x):
    if not torch.is_complex(x):
        return norm_func(x)
    else:
        return torch.complex(norm_func(x.real), norm_func(x.imag))


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.pe = torch.zeros((max_len, d_model), dtype=torch.float)
        self.pe.requires_grad = False

        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()

        self.pe[:, ::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        if d_model % 2 == 0:
            self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        else:
            self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))[:, :-1]

        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class Attention(nn.Module):
    def __init__(self, window_size, mask_flag=False, scale=None, dropout=0.0):
        super(Attention, self).__init__()
        self.window_size = window_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        '''
        queries : N x L x Head x d
        keys : N x L(s) x Head x d
        values : N x L x Head x d
        '''
        N, L, Head, C = queries.shape

        scale = self.scale if self.scale is not None else 1. / sqrt(C)

        attn_scores = complex_einsum('nlhd,nshd->nhls', queries, keys)    # N x Head x L x L

        attn_weights = complex_dropout(self.dropout, complex_softmax(scale * attn_scores, dim=-1))

        updated_values = complex_einsum('nhls,nshd->nlhd', attn_weights, values)  # N x L x Head x d

        return updated_values.contiguous(), attn_weights.permute(0, 2, 1, 3).mean(dim=-2)

    
class AttentionLayer(nn.Module):
    def __init__(self, w_size, d_model, n_heads, d_keys=None, d_values=None, mask_flag=False,
                 scale=None, dropout=0.0):
        super(AttentionLayer, self).__init__()

        n_heads = n_heads if (d_model % n_heads) == 0 else 1

        z = d_model % n_heads if (d_model // n_heads) == 0 else (d_model // n_heads)

        self.d_keys = d_keys if d_keys is not None else z
        self.d_values = d_values if d_values is not None else z
        self.n_heads = n_heads
        self.d_model = d_model  # d_model = C

        self.pos_embedding = PositionalEmbedding(d_model=d_model)

        # Linear projections to Q, K, V
        self.W_Q = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_K = nn.Linear(self.d_model, self.n_heads * self.d_keys)
        self.W_V = nn.Linear(self.d_model, self.n_heads * self.d_values)

        # self.out_proj = nn.Linear(self.n_heads * self.d_values, self.d_model)
        self.out_proj = lambda x: x

        self.attn = Attention(window_size=w_size, mask_flag=mask_flag, scale=scale, dropout=dropout)

    def forward(self, input_data):
        '''
        input : N x L x C(=d_model)
        '''

        N, L, _ = input_data.shape

        # input_data = input_data  + self.pos_embedding(input_data)

        # Q = self.W_Q(input_data).contiguous().view(N, L, self.n_heads, -1)
        # K = self.W_K(input_data).contiguous().view(N, L, self.n_heads, -1)
        # V = self.W_V(input_data).contiguous().view(N, L, self.n_heads, -1)

        Q = input_data.contiguous().view(N, L, self.n_heads, -1)
        K = input_data.contiguous().view(N, L, self.n_heads, -1)
        V = input_data.contiguous().view(N, L, self.n_heads, -1)

        updated_V, attn = self.attn(Q, K, V)  # N x L x Head x d_values
        out = self.out_proj(updated_V.view(N, L, -1))
        # out = self.out_proj(updated_V.view(N, L, -1) + input_data)

        return out, attn


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout=0.2, alpha=0.01, embed_dim=None, use_gatv2=True, use_bias=False):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        # if self.use_bias:
        #     self.bias = nn.Parameter(torch.empty(window_size, window_size))

        # self.leakyrelu = nn.LeakyReLU(alpha)
        self.leakyrelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(window_size)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)                # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, n, n, 1)
            e = self.norm(e)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)

        # if self.use_bias:
        #     e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # h = self.sigmoid(torch.matmul(attention, x))    # (b, n, k)
        h = torch.matmul(attention, x)    # (b, n, k)

        return h, attention

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)
