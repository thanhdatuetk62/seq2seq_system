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
            memory_key_padding_mask=memory_key_padding_mask)

        output = self.out(output)
        output = F.softmax(output, dim=-1)

        # only care prob from the last token [N x trg_vocab_size]
        output = output[-1]
        assert output.size() == (trg.size(1), self.trg_vocab_size)

        return output, score
