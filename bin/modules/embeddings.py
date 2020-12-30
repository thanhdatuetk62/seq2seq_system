import torch, math
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=300, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        rows = torch.arange(max_len).unsqueeze(1)
        cols = torch.exp(-math.log(10000.0) / d_model * \
            torch.arange(0, d_model, 2))
        
        pe[:, 0::2] = torch.sin(rows * cols)
        pe[:, 1::2] = torch.cos(rows * cols)

        pe.unsqueeze_(0)
        self.register_buffer("pe", pe)
        

    def forward(self, x):
        """
        Arguments:
            x (Tensor): [batch_size x seq_len x d_model]
        """
        # Add pe to the input and dropout
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
