import math
from torch import nn
from torch.nn import functional as F
import torch

from .model import _Model

from ..modules import PositionalEncoding, Encoder, Decoder, Linear
from ..utils import generate_subsequent_mask


class TransformerNMT(_Model):
    def __init__(self, d_model=512, nhead=8, activation="relu",
                 num_encoder_layers=6, num_decoder_layers=6,  dim_ff=2048,
                 dropout=0.1, max_input_length=100, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_input_length
        self.d_model = d_model
        self.n_heads = nhead

        # Create embedding layer for both Encoder and Decoder
        self.src_embed = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embed = nn.Embedding(self.trg_vocab_size, d_model)

        # Positional Encoding module
        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Transformer module
        self.encoder = Encoder(n_layers=num_encoder_layers, d_model=d_model,
                               n_heads=nhead, dropout=dropout,
                               activation=activation, d_ff=dim_ff)
        self.decoder = Decoder(n_layers=num_decoder_layers, d_model=d_model,
                               n_heads=nhead, dropout=dropout,
                               activation=activation, d_ff=dim_ff)
                                    
        # Create ff network for output probabilities
        self.out = Linear(d_model, self.trg_vocab_size)

    def forward(self, src, trg):
        """
        Forward step for training batch (include source and target batch)
        Arguments:
            src: (Tensor [S x N]) - Source batch tokens
            trg: (Tensor [T x N]) - Target batch tokens
        Returns:
            (Tensor [T x N x trg_vocab_size]) - Score distributions
        """
        pad_token = self.data.src_vocab.stoi["<pad>"]
        # Masks
        trg_mask = generate_subsequent_mask(trg.size(0), self.device)
        src_key_padding_mask = memory_key_padding_mask = (src == pad_token).t()

        # Embedding source tokens
        src = self.src_embed(src)
        src = self.pe(src)

        # Embedding target tokens
        trg = self.trg_embed(trg)
        trg = self.pe(trg)

        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        output, scores = self.decoder(trg, memory, trg_mask=trg_mask, \
            memory_key_padding_mask=memory_key_padding_mask)
        
        output = self.out(output)
        return output

    def encode(self, src):
        """
        Get info of encoder output, which will be fed into decoder. Guarantte \
        that the first dimension of all vars are {batch_size} dimension
        Arguments:
            src: (Tensor [S x N])
        """
        pad_token = self.data.src_vocab.stoi["<pad>"]
        src_key_padding_mask = memory_key_padding_mask = (src == pad_token).t()

        # Embedding source tokens
        src = self.src_embed(src)
        src = self.pe(src)
 
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        return {
            "memory": memory, 
            "memory_key_padding_mask": memory_key_padding_mask.t()
        }
    
    def decode(self, trg, memory, padding_mask=None, cache=None):
        t = trg.size(0)
        if cache is not None:
            # Only deal with the last token when using cache
            trg = trg[-1].unsqueeze(0)
        # Embedding target tokens
        trg = self.trg_embed(trg)
        trg = self.pe(trg, pos=t-1)

        # Transformer output
        trg_mask = generate_subsequent_mask(trg.size(0), self.device)
        output, score = self.decoder(trg, memory, trg_mask=trg_mask, \
            memory_key_padding_mask=padding_mask.t(), cache=cache)

        output = self.out(output)
        output = F.softmax(output, dim=-1)

        # only care prob from the last token [N x trg_vocab_size]
        return output[-1]
    
    def infer_step(self, trg, memory_info):
        """
        Compute output for each decode step
        """
        # Extract memory info (cause of TorchScript does not support varargs)
        memory = memory_info["memory"]
        memory_key_padding_mask = memory_info["memory_key_padding_mask"].t()

        # Embedding target tokens
        trg = self.trg_embed(trg)
        trg = self.pe(trg)

        # Transformer output
        trg_mask = generate_subsequent_mask(trg.size(0), self.device)
        output, score = self.decoder(trg, memory, trg_mask=trg_mask, \
            memory_key_padding_mask=memory_key_padding_mask, need_weights=True)

        output = self.out(output)
        output = F.softmax(output, dim=-1)

        # only care prob from the last token [N x trg_vocab_size]
        output = output[-1]
        assert output.size() == (trg.size(1), self.trg_vocab_size)

        return output, score


class MemorizedDecoder(nn.Module):
    """
    Ultra fast decoder for BeamSearch powered by Dynamic Programming
    """
    def __init__(self, model, k, eos_token, sos_token, memory_info, \
        batch_size, device="cpu"):
        super().__init__()
        self.model = model
        self.k = k
        self.n = batch_size
        self.eos_token = eos_token
        self.sos_token = sos_token
        self.device = device

        self.cache = [{
            "self_attn": {
                "q": torch.zeros(0, self.n, model.d_model, device=device), 
                "k": torch.zeros(0, self.n, model.d_model, device=device), 
                "v": torch.zeros(0, self.n, model.d_model, device=device), 
                # "actives": torch.arange(self.n)
            },
            "attn": {
                "q": torch.zeros(0, self.n, model.d_model, device=device), 
                # "actives": torch.arange(self.n)
            }
        } for _ in range(model.decoder.n_layers)]

        self.memory = memory_info["memory"]
        self.memories = self.memory.repeat_interleave(k, dim=1)

        self.padding_mask = memory_info["memory_key_padding_mask"]
        self.padding_masks = self.padding_mask.repeat_interleave(k, dim=1)
    
    def reorder_beams(self, i):
        """
        Eliminate and reorder beams for each batch after each timestep in CACHE
        Arguments:
            i: (Tensor) - Indices of beams composed by batches
        """
        for attn_cache in self.cache:
            attn_cache["self_attn"]["q"] = attn_cache["self_attn"]["q"][:, i]
            attn_cache["self_attn"]["k"] = attn_cache["self_attn"]["k"][:, i]
            attn_cache["self_attn"]["v"] = attn_cache["self_attn"]["v"][:, i]
            attn_cache["attn"]["q"] = attn_cache["attn"]["q"][:, i]

    def burn(self):
        """Initialization for the first timestep"""
        trg = torch.tensor([self.sos_token] * self.n, \
            device=self.device).unsqueeze(0)
        prob = self.model.decode(trg, self.memory, self.padding_mask, self.cache)
        it = torch.arange(self.n).repeat_interleave(self.k, dim=-1)
        self.reorder_beams(it)
        return prob

    def forward(self, trg):
        """
        Feed batch of target beams into Transformer decoder
        Arguments:
            trg: (Tensor [T x N x k]) Target input with it beams
        Returns:
            Tensor [N x k x trg_vocab_size]
            - Probabilities distribution of the next generated target-side tokens
        """
        t, n, k = trg.size()
        assert n == self.n and k == self.k

        # active_mask = (trg == self.eos_token).any(0).view(-1)
        # actives = torch.nonzero(active_mask==0).view(-1)

        # for attn_cache in self.cache:
        #     attn_cache["self_attn"]["actives"] = actives
        #     attn_cache["attn"]["actives"] = actives
        
        probs = torch.zeros((n*k, self.model.trg_vocab_size), \
            device=self.device, dtype=torch.float)
        
        flatten_trg = trg.view(t, -1)
        # probs[actives] = self.model.decode(flatten_trg[:, actives], \
        #     self.memories[:, actives], self.padding_masks[:, actives], self.cache)
        probs = self.model.decode(flatten_trg, \
            self.memories, self.padding_masks, self.cache)
        return probs.view(n, k, -1)