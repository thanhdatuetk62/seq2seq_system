import torch 
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter

activation_fn = {"relu": F.relu}

class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.w = Parameter(torch.Tensor(d_out, d_in))
        self.b = Parameter(torch.Tensor(d_out))

        self._reset_params()
    
    def _reset_params(self):
        init.xavier_uniform_(self.w)
        init.constant_(self.b, 0.0)

    def forward(self, x):
        return F.linear(x, self.w, self.b)
