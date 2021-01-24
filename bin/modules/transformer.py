import torch
import math
import copy
from .linear import Linear
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

def _get_activation_fn(activation):
    activation_fn = {"relu": F.relu}
    return activation_fn.get(activation, "relu")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _concate(x, y, dim=0):
    return torch.cat((x, y), dim=dim)


class MultiAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads

        # self.w_q = Linear(d_model, d_model)
        # self.w_k = Linear(d_model, d_model)
        # self.w_v = Linear(d_model, d_model)

        self.in_proj_weights = Parameter(torch.Tensor(3 * d_model, d_model))
        self.in_proj_bias = Parameter(torch.Tensor(3 * d_model))

        self.w_o = Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_params()
    
    def _reset_params(self):
        nn.init.xavier_uniform_(self.in_proj_weights)
        nn.init.constant_(self.in_proj_bias, 0.0)

    def forward(self, x_q, x_k, x_v, key_padding_mask=None, \
        attn_mask=None, need_weights=False, cache=None):
        """
        Multi-head attention Module
        Arguments: 
            x_q: (Tensor [T x N x d_model])
            x_k: (Tensor [S x N x d_model])
            x_v: (Tensor [S x N x d_model])
        """
        T, n, d_model = x_q.size()
        S = x_k.size(0)
        n_heads = self.n_heads
        d_q = d_k = d_v = d_model // n_heads
        assert x_q.size(-1) == d_model
        assert x_k.size(-1) == d_model
        assert x_v.size(-1) == d_model
        assert x_k.size(1) == n
        assert x_v.size(1) == n
        assert x_v.size(0) == S

        is_self_attn = False

        # Compute key, query, value matrices for each head
        # [(N * H) x L x d]
        # q = self.w_q(x_q)
        # k = self.w_k(x_k)
        # v = self.w_v(x_v)

        if torch.equal(x_q, x_k) and torch.equal(x_q, x_v):
            # Self-Attention flow
            is_self_attn = True
            q, k, v = F.linear(x_q, self.in_proj_weights, \
                self.in_proj_bias).chunk(3, dim=-1)
        else:
            assert torch.equal(x_k, x_v)
            # Encoder-Decoder Attention flow
            _w = self.in_proj_weights[:d_model]
            _b = self.in_proj_bias[:d_model]
            q = F.linear(x_q, _w, _b)

            _w = self.in_proj_weights[d_model:]
            _b = self.in_proj_bias[d_model:]
            k, v = F.linear(x_k, _w, _b).chunk(2, dim=-1)
        
        if cache is not None:
            m = cache["q"].size(1)
            actives = cache["actives"]
            zeros = torch.zeros((1, m, d_model), device=q.device)
            cache["q"] = _concate(cache["q"], zeros, dim=0)
            cache["q"][-1, actives] = q
            if is_self_attn:
                cache["k"] = _concate(cache["k"], zeros, dim=0)
                cache["v"] = _concate(cache["v"], zeros, dim=0)
                cache["k"][-1, actives] = k
                cache["v"][-1, actives] = v
                # Replace with full cache [T x N x d_model]
                k, v = cache["k"][:, actives], cache["v"][:, actives]

        q = q.contiguous().view(-1, n * n_heads, d_q).transpose(0, 1)
        k = k.contiguous().view(-1, n * n_heads, d_k).transpose(0, 1)
        v = v.contiguous().view(-1, n * n_heads, d_v).transpose(0, 1)
        
        # if using cache: [(N * n_heads) x 1 x T]
        # else: [(N * n_heads) x T x T]
        score = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d_k)

        if (cache is None) and (attn_mask is not None):
            # Mask attention, turn off when using cache
            attn_mask = attn_mask.unsqueeze(0)
            score = score.masked_fill(attn_mask, float("-inf"))
        
        if key_padding_mask is not None:
            # Key padding mask, useful when facing encoder-decoder attention
            score = score.view(n, n_heads, -1, score.size(-1))
            score = score.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(1), 
                float("-inf"))
            score = score.view(n * n_heads, -1, score.size(-1))
        
        # Softmax
        score = F.softmax(score, dim=-1)

        if cache is not None:
            m = cache["weights"].size(0)
            actives = cache["actives"]
            score = score.view(n, n_heads, -1, score.size(-1))
            
            if is_self_attn:
                t = cache["weights"].size(-1)
                zeros = torch.zeros((m, n_heads, t, 1), device=score.device)
                cache["weights"] = _concate(cache["weights"], zeros, dim=-1)

            t = cache["weights"].size(-1)
            zeros = torch.zeros((m, n_heads, 1, t), device=score.device)
            cache["weights"] = _concate(cache["weights"], zeros, dim=2)
            cache["weights"][actives, :, -1] = score[:, :, -1]
            score = score.view(n * n_heads, -1, score.size(-1))

        score = self.dropout(score)
        out = torch.bmm(score, v)
        # Concat
        out = out.transpose(0, 1).contiguous().view(-1, n, d_model)
        out = self.w_o(out)
        
        if need_weights:
            if cache is not None:
                raise ValueError("Currently not support return attention when using cache")
            # Average attention over heads
            return out, score.view(n, n_heads, T, S).mean(dim=1)
        else:
            return out, None
        

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, d_ff=2048, \
        activation="relu"):
        super().__init__()
        # Multihead Attention module
        # self.multi_att = nn.MultiheadAttention(d_model, n_heads, dropout)
        self.multi_att = MultiAttention(d_model, n_heads, dropout)
        
        # Feed forward modules
        self.linear_1 = Linear(d_model, d_ff)
        self.linear_2 = Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask, src_key_padding_mask):
        """
        Args:
            x (Tensor [S x N x d_model]) - Input tensor
            mask (Tensor [S x S]) - Input mask tensor
            src_key_padding_mask (Tensor[N x S]) - Source padding mask
        Returns:
            (Tensor [N x S x d_model]) - Output tensor
        """
        src_2, _ = self.multi_att(src, src, src, \
            key_padding_mask=src_key_padding_mask, \
            need_weights=False, attn_mask=src_mask)
        src = src + self.dropout_1(src_2)
        src = self.norm_1(src)

        src_2 = self.linear_2(self.dropout(self.activation(self.linear_1(src))))
        src = src + self.dropout_2(src_2)
        src = self.norm_2(src)
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1, d_ff=2048, \
        activation="relu"):
        super().__init__()
        # Multihead Attention module
        # self.multi_att_1 = nn.MultiheadAttention(d_model, n_heads, dropout)
        # self.multi_att_2 = nn.MultiheadAttention(d_model, n_heads, dropout)
        self.multi_att_1 = MultiAttention(d_model, n_heads, dropout)
        self.multi_att_2 = MultiAttention(d_model, n_heads, dropout)
        
        # Feed forward modules
        self.linear_1 = Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = Linear(d_ff, d_model)
        self.activation = _get_activation_fn(activation)

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
    
    def forward(self, trg, memory, memory_mask, trg_mask, \
        memory_key_padding_mask, trg_key_padding_mask, \
        attn_cache=None, need_weights=False):
        """
        Args:
            x (Tensor [T x N x d_model]) - Input tensor
            memory (Tensor [S x N x d_model]) - Encoder's output
            memory_mask (Tensor [T x S]) - Memory mask
            trg_mask (Tensor [T x T]) - Target mask
            memory_key_padding_mask (Tensor [N x S]) - Memory key padding mask
            trg_key_padding_mask (Tensor [N x T])
            attn_cache (Dict[str, Tensor]) - Used for ultra-fast decoding (infer only)
        Returns:
            (Tensor [T x N x d_model], Tensor [N x T x S]) 
                - Tuple contains output tensor and attention score
        """
        self_attn, attn = None, None
        if attn_cache is not None:
            self_attn = attn_cache["self_attn"]
            attn = attn_cache["attn"]

        # Self-Attention layer
        trg_2 , _= self.multi_att_1(trg, trg, trg, \
            key_padding_mask=trg_key_padding_mask, \
            need_weights=False, attn_mask=trg_mask, cache=self_attn)
        trg = trg + self.dropout_1(trg_2)
        trg = self.norm_1(trg)

        # Encoder-Decoder Attention layer
        trg_2, score = self.multi_att_2(trg, memory, memory, \
            key_padding_mask=memory_key_padding_mask, \
            need_weights=need_weights, attn_mask=memory_mask, cache=attn)
        trg = trg + self.dropout_2(trg_2)
        trg = self.norm_2(trg)

        # Feed Forward layer
        trg_2 = self.linear_2(self.dropout(self.activation(self.linear_1(trg))))
        trg = trg + self.dropout_3(trg_2)
        trg = self.norm_3(trg)

        return trg, score


class Encoder(nn.Module):
    def __init__(self, n_layers=6, d_model=512, n_heads=8, dropout=0.1, \
        activation="relu", d_ff=2048):
        super().__init__()
        layer = EncoderLayer(d_model, n_heads, dropout, d_ff, activation)
        self.layers = _get_clones(layer, n_layers)
        self.n_layers = n_layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            x (Tensor [S x N x d_model]) - Input tensor
            mask (Tensor [S x S]) - Input mask tensor
            src_key_padding_mask (Tensor[N x S]) - Source padding mask
        Returns:
            (Tensor [S x N x d_model]) - Output tensor
        """
        out = src 
        for mod in self.layers:
            out = mod(out, src_mask, src_key_padding_mask)
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
    
    def forward(self, trg, memory, memory_mask=None, trg_mask=None, \
        memory_key_padding_mask=None, trg_key_padding_mask=None, \
        cache=None, need_weights=False):
        """
        Args:
            x (Tensor [T x N x d_model]) - Input tensor
            memory (Tensor [S x N x d_model]) - Encoder's output
            memory_mask (Tensor [T x S]) - Memory mask
            trg_mask (Tensor [T x T]) - Target mask
            memory_key_padding_mask (Tensor [N x S]) - Memory key padding mask
            trg_key_padding_mask (Tensor [N x T])
            cache (List[Dict[str, Tensor]]) - List of attn_cache of decoder layers
        Returns:
            (Tensor [T x N x d_model], list(Tensor [N x T x S])) 
                - Tuple contains output tensor and attention scores of layers
        """
        if cache is not None:
            assert len(cache) == self.n_layers
        else:
            cache = [None] * self.n_layers
        out = trg
        scores = []
        for mod, attn_cache in zip(self.layers, cache):
            out, score = mod(out, memory, memory_mask, trg_mask, \
                memory_key_padding_mask, trg_key_padding_mask, \
                need_weights=need_weights, attn_cache=attn_cache)
            scores.append(score)
        out = self.norm(out)
        if need_weights:
            return out, scores[-1]
        else:
            return out, None



