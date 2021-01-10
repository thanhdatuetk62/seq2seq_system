import torch 
from torch import nn
from torch.nn import functional as F

activation_fn = {"relu": F.relu}

class FeedForward(nn.Module):
    def __init__(self, *d_ff, activation="relu"):
        super().__init__()
        n = len(d_ff)
        assert n > 1

        self.d_in = d_ff[0]
        self.activation_fn = activation_fn[activation]
        self.layers = [nn.Linear(d_ff[i-1], d_ff[i]) for i in range(1, n)]
    
    def foward(self, x):
        assert x.size(-1) == self.d_in
        for layer in self.layers:
            x = self.activation_fn(layer(x))
        return x