from torch import nn


class _Model(nn.Module):
    def __init__(self, data, device="cpu"):
        super().__init__()
        self.data = data
        self.device = device

        # Reveal some data info
        self.src_vocab_size = len(data.src_vocab)
        self.trg_vocab_size = len(data.trg_vocab)

        print("Source vocab size: ", self.src_vocab_size)
        print("Target vocab size: ", self.trg_vocab_size)

    def init_params(self):
        print("Initialize model parameters ...", end="")
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
