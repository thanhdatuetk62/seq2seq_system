from torch import nn
import torch

class _Model(nn.Module):
    def __init__(self, portal, device="cpu"):
        super().__init__()
        self.portal = portal
        self.device = device

        # Reveal some data info
        self.src_vocab_size = len(portal.src_vocab)
        self.trg_vocab_size = len(portal.trg_vocab)

        print("Source vocab size: ", self.src_vocab_size)
        print("Target vocab size: ", self.trg_vocab_size)

    def init_params(self):
        print("Initialize model parameters ...")
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print("Done!")

    def train_step(self):
        raise NotImplementedError
    
    def e_out(self):
        raise NotImplementedError

    def validate_step(self):
        raise NotImplementedError
    
    def infer_step(self):
        raise NotImplementedError
