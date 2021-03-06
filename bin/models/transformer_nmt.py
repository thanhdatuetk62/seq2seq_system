from torch import nn
from torch.nn import functional as F
import torch

from .model import _Model

from ..modules import PositionalEncoding
from ..utils import generate_subsequent_mask


class TransformerNMT(_Model):
    def __init__(self, d_model=512, nhead=8, activation="relu",
                 num_encoder_layers=6, num_decoder_layers=6,  dim_ff=2048,
                 dropout=0.1, max_input_length=300, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model

        # Create embedding layer for both Encoder and Decoder
        self.src_embed = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embed = nn.Embedding(self.trg_vocab_size, d_model)

        # Positional Encoding module
        self.pe = PositionalEncoding(max_len=max_input_length,
                                     d_model=d_model, dropout=dropout)

        # Transformer module
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_ff,
                                          dropout=dropout,
                                          activation=activation)
                                    
        # Create ff network for output probabilities
        self.out = nn.Linear(d_model, self.trg_vocab_size)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        """
        Forward step for training batch (include source and target batch)
        Arguments:
            src: (Tensor [N x S]) - Source batch tokens
            trg: (Tensor [N x T]) - Target batch tokens
            src_mask: (Tensor [N x S]) - Source mask
            trg_mask: (Tensor [T X T]) - Target mask (default: Causal mask)
        Returns:
            (Tensor [N x T x trg_vocab_size]) - Score distributions
        """
        # Embedding source tokens
        src = self.src_embed(src)
        src = self.pe(src)

        # Embedding target tokens
        trg = self.trg_embed(trg)
        trg = self.pe(trg)

        # transpose batch <-> len dim for using built-in transformer module
        output = self.transformer(src.transpose(0, 1),
                                  trg.transpose(0, 1),
                                  src_mask=src_mask,
                                  tgt_mask=trg_mask).transpose(0, 1)
        output = self.out(output)

        return output

    def e_out(self, src, src_mask=None):
        """
        Get info of encoder output, which will be fed into decoder. Guarantte \
        that the first dimension of all vars are {batch_size} dimension
        Arguments:
            src: (Tensor [N x S]) 
            src_mask: (Tensor [N x S])
        Returns:
            (Dict[str, Tensor]) - A dictionary contains all information.
        """
        # Embedding source tokens
        src = self.src_embed(src)
        src = self.pe(src)

        # transpose batch <-> len dim for using built-in transformer module
        memory = self.transformer.encoder(src.transpose(0, 1), src_mask).\
            transpose(0, 1)

        return {"memory": memory}

    def train_step(self, src, trg):
        """
        Compute loss for each train step
        Arguments:
            src: (Tensor [N x S]) - Input to Encoder
            trg: (Tensor [N x T]) - Input to Decoder
        Return:
            Scalar (float) - Batch's loss value 
        """
        self.train()

        # Compute model output
        trg_mask = generate_subsequent_mask(trg.size(1)-1, self.device)
        output = self(src, trg[:, :-1], trg_mask=trg_mask)

        # Flatten tensors for computing loss
        preds = output.view(-1, self.trg_vocab_size)
        ys = trg[:, 1:].contiguous().view(-1)

        # Feed inputs into loss metric
        loss = self.loss_metric(preds, ys)
        loss_val = loss.item()

        # Back-propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_val

    @torch.no_grad()
    def validate_step(self, src, trg):
        """
        Compute loss for each validate step
        Arguments:
            src: (Tensor [N x S]) - Input to Encoder
            trg: (Tensor [N x T]) - Input to Decoder
        Return:
            Scalar (float) - Batch's loss value 
        """
        self.eval()

        # Compute model output
        trg_mask = generate_subsequent_mask(trg.size(1)-1, self.device)
        output = self(src, trg[:, :-1], trg_mask=trg_mask)

        # Flatten tensors for computing loss
        preds = output.view(-1, self.trg_vocab_size)
        ys = trg[:, 1:].contiguous().view(-1)

        # Feed inputs into loss metric
        loss = self.loss_metric(preds, ys)
        loss_val = loss.item()

        return loss_val

    @torch.no_grad()
    def infer_step(self, trg, memory, memory_mask=None):
        """
        Compute output for each decode step
        Arguments:
            memory (Tensor [N x S x d_model]) \
                - Output from encoder
            trg (Tensor [N x T]) \
                - Input to decoder which is generated by algorithms.
        Return:
            prob (Tensor [N x trg_vocab_size]) \
                - Prob from the last token
        """
        self.eval()

        N, T = trg.size()

        # Embedding target tokens
        trg = self.trg_embed(trg)
        trg = self.pe(trg)

        # Transformer output
        trg_mask = generate_subsequent_mask(T, self.device)
        output = self.transformer.decoder(trg.transpose(0, 1),
                                          memory.transpose(0, 1),
                                          tgt_mask=trg_mask,
                                          memory_mask=memory_mask).transpose(0, 1)
        output = self.out(output)

        # prob distributions [N x cur_len x trg_vocab_size]
        output = F.softmax(output, dim=-1)

        # only care prob from the last token [N x trg_vocab_size]
        output = output[:, -1, :]
        assert output.size() == (N, self.trg_vocab_size)

        return output
