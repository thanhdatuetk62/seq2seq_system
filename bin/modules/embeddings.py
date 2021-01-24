import torch, math
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, fix_max_len=200, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
    
        pe = torch.zeros(fix_max_len, d_model)
        rows = torch.arange(fix_max_len).unsqueeze(1)
        cols = torch.exp(-math.log(10000.0) / d_model * \
            torch.arange(0, d_model, 2))
        
        pe[:, 0::2] = torch.sin(rows * cols)
        pe[:, 1::2] = torch.cos(rows * cols)

        pe.unsqueeze_(1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x, pos=None):
        """
        Arguments:
            x (Tensor): [len x N x d_model]
        """
        # Add pe to the input and dropout
        x = x * (self.d_model ** 0.5)
        if pos is not None:
            x = x + self.pe[pos]
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)
