import torch
import math
import copy
from torch import nn
from torch.nn import functional as F

def _get_activation_fn(activation):
    activation_fn = {"relu": F.relu}
    return activation_fn.get(activation, "relu")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_k, x_v, mask=None):
        n = x_q.size(0)
        d_model, n_heads = self.d_model, self.n_heads
        d_q = d_k = d_v = self.d_model // self.n_heads
        assert x_q.size(-1) == d_model
        assert x_k.size(-1) == d_model
        assert x_v.size(-1) == d_model
        assert x_k.size(0) == n
        assert x_v.size(0) == n

        # Compute key, query, value matrices for each head
        # [N x H x S x d_k]
        q = self.w_q(x_q).view(n, -1, n_heads, d_q).transpose(1, 2)
        k = self.w_k(x_k).view(n, -1, n_heads, d_k).transpose(1, 2)
        v = self.w_v(x_k).view(n, -1, n_heads, d_v).transpose(1, 2)

        # Scaled dot-product attention score [N x H x S x S]
        score = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_model)
        # Mask attention
        if mask is not None:
            score.masked_fill_(mask, float("-inf"))
        # Softmax
        score = torch.softmax(score, dim=-1)
        out = torch.matmul(score, v)
        # Concat
        out = out.transpose(1, 2).contiguous().view(n, -1, d_model)
        out = self.w_o(out)
        return out, score
        

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, d_ff=2048, \
        activation="relu"):
        super().__init__()
        self.multi_att = MultiAttention(d_model, n_heads)
        
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = _get_activation_fn(activation)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, src, src_mask=None):
        """
        Args:
            x (Tensor [N x S x d_model]) - Input tensor
        Returns:
            (Tensor [N x S x d_model]) - Output tensor
        """
        out, score = self.multi_att(src, src, src, src_mask)
        src = src + self.dropout(out)
        src = self.norm_1(src)
        out = self.linear_2(
                self.dropout(
                    self.activation(
                        self.linear_1(src))))
        src = src + self.dropout(out)
        src = self.norm_2(src)
        return src


class Encoder(nn.Module):
    def __init__(self, n_layers=6, d_model=512, n_heads=8, dropout=0.1, \
        activation="relu", d_ff=2048):
        super().__init__()
        layer = EncoderLayer(d_model, n_heads, dropout, d_ff, activation)
        self.layers = _get_clones(layer, n_layers)
        self.n_layers = n_layers

    def forward(self, src, mask=None):
        out = src 
        for mod in self.layers:
            out = mod(out, mask)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, d_ff=2048, \
        activation="relu"):
        super().__init__()
        self.multi_att_1 = MultiAttention(d_model, n_heads)
        self.multi_att_2 = MultiAttention(d_model, n_heads)
        
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = _get_activation_fn(activation)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, trg, memory, memory_mask=None, trg_mask=None):
        """
        Args:
            x (Tensor [N x T x d_model]) - Input tensor
        Returns:
            (Tensor [N x T x d_model]) - Output tensor
        """
        out, self_score = self.multi_att_1(trg, trg, trg, trg_mask)
        trg = trg + self.dropout(out)
        trg = self.norm_1(trg)

        out, score = self.multi_att_2(trg, memory, memory, memory_mask)
        trg = trg + self.dropout(out)
        trg = self.norm_2(trg)

        out = self.linear_2(
                self.dropout(
                    self.activation(
                        self.linear_1(trg))))
        trg = trg + self.dropout(out)
        trg = self.norm_3(trg)
        return trg    


class Decoder(nn.Module):
    def __init__(self, n_layers=6, d_model=512, n_heads=8, dropout=0.1, \
        activation="relu", d_ff=2048):
        super().__init__()
        layer = DecoderLayer(d_model, n_heads, dropout, d_ff, activation)
        self.layers = _get_clones(layer, n_layers)
        self.n_layers = n_layers
    
    def forward(self, trg, memory, memory_mask=None, trg_mask=None):
        out = trg
        for mod in self.layers:
            out = mod(out, memory, memory_mask, trg_mask)
        return out
        
