import torch
from torch import nn


class _Strategy(nn.Module):
    def __init__(self, data, model, sos_token="<sos>", device="cpu"):
        super().__init__()
        self.data = data
        self.model = model
        self.device = device
        self.sos_token = self.data.trg_vocab.stoi[sos_token]
        self.eos_token = self.data.trg_vocab.stoi["<eos>"]
      
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
    
       
    