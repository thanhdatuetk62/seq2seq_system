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

# Default value replaced for None
DEFAULT_MASK = torch.tensor(-1)

class MultiAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_k, x_v, mask=DEFAULT_MASK):
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
        v = self.w_v(x_v).view(n, -1, n_heads, d_v).transpose(1, 2)

        # Scaled dot-product attention score [N x H x S x T]
        score = torch.matmul(q, k.transpose(-1, -2)) * (d_k ** -0.5)
        # Mask attention
        if mask.dim() > 1:
            score.masked_fill_(mask, float("-inf"))
        # Softmax
        score = F.softmax(score, dim=-1)
        score = self.dropout(score)
        out = torch.matmul(score, v)
        # Concat
        out = out.transpose(1, 2).contiguous().view(n, -1, d_model)
        out = self.w_o(out)
        return out, score
        

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, d_ff=2048, \
        activation="relu"):
        super().__init__()
        # Multihead Attention module
        self.multi_att = MultiAttention(d_model, n_heads, dropout)
        
        # Feed forward modules
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

        # Training modules
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=DEFAULT_MASK):
        """
        Args:
            x (Tensor [N x S x d_model]) - Input tensor
            mask ([Optional] Tensor [S x S]) - Input mask tensor
        Returns:
            (Tensor [N x S x d_model]) - Output tensor
        """
        src_2 = self.norm_1(src)
        out, score = self.multi_att(src_2, src_2, src_2, src_mask)
        src = src + self.dropout_1(out)
        src_2 = self.norm_2(src)
        out = self.linear_2(
                self.dropout(
                    self.activation(
                        self.linear_1(src_2))))
        src = src + self.dropout_2(out)

        # ================================================

        # src_2, score = self.multi_att(src, src, src, src_mask)
        # src = src + self.dropout_1(src_2)
        # src = self.norm_1(src)
        # src_2 = self.linear_2(
        #             self.dropout(
        #                 self.activation(
        #                     self.linear_1(src))))
        # src = src + self.dropout_2(src_2)
        # src = self.norm_2(src)

        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, d_ff=2048, \
        activation="relu"):
        super().__init__()
        # Multihead Attention module
        self.multi_att_1 = MultiAttention(d_model, n_heads, dropout)
        self.multi_att_2 = MultiAttention(d_model, n_heads, dropout)
        
        # Feed forward modules
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = _get_activation_fn(activation)

        # Training modules
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
    
    def forward(self, trg, memory, memory_mask=DEFAULT_MASK, trg_mask=DEFAULT_MASK):
        """
        Args:
            x (Tensor [N x T x d_model]) - Input tensor
        Returns:
            (Tensor [N x T x d_model]) - Output tensor
        """
        trg_2 = self.norm_1(trg)
        out, _ = self.multi_att_1(trg_2, trg_2, trg_2, trg_mask)
        trg = trg + self.dropout_1(out)
        
        trg_2 = self.norm_2(trg)
        out, score = self.multi_att_2(trg_2, memory, memory, memory_mask)
        trg = trg + self.dropout_2(out)
        
        trg_2 = self.norm_3(trg)
        out = self.linear_2(
                self.dropout(
                    self.activation(
                        self.linear_1(trg_2))))
        trg = trg + self.dropout_3(out)

        #====================================================================

        # trg_2, self_score = self.multi_att_1(trg, trg, trg, trg_mask)
        # trg = trg + self.dropout_1(trg_2)
        # trg = self.norm_1(trg)

        # trg_2, score = self.multi_att_2(trg, memory, memory, memory_mask)
        # trg = trg + self.dropout_2(trg_2)
        # trg = self.norm_2(trg)

        # trg_2 = self.linear_2(
        #             self.dropout(
        #                 self.activation(
        #                     self.linear_1(trg))))
        # trg = trg + self.dropout_3(trg_2)
        # trg = self.norm_3(trg)

        return trg, score


class Encoder(nn.Module):
    def __init__(self, n_layers=6, d_model=512, n_heads=8, dropout=0.1, \
        activation="relu", d_ff=2048):
        super().__init__()
        layer = EncoderLayer(d_model, n_heads, dropout, d_ff, activation)
        self.layers = _get_clones(layer, n_layers)
        self.n_layers = n_layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=DEFAULT_MASK):
        out = src 
        for mod in self.layers:
            out = mod(out, src_mask)
        out = self.norm(out)
        return out


class Decoder(nn.Module):
    def __init__(self, n_layers=6, d_model=512, n_heads=8, dropout=0.1, \
        activation="relu", d_ff=2048):
        super().__init__()
        layer = DecoderLayer(d_model, n_heads, dropout, d_ff, activation)
        self.layers = _get_clones(layer, n_layers)
        self.n_layers = n_layers
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, trg, memory, memory_mask=DEFAULT_MASK, trg_mask=DEFAULT_MASK):
        out = trg
        score = None
        for mod in self.layers:
            out, score = mod(out, memory, memory_mask, trg_mask)
        out = self.norm(out)
        # return with head-averaged encoder-decoder attention from the top decoder layer
        # score = torch.mean(score, dim=1)
        return out, score[:, 0, :, :].squeeze(1)
        
