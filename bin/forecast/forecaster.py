from torch import nn
from .strategies import find_forecast_strategy

class Forecaster(nn.Module):
    def __init__(self, controller, strategy="beam_search", strategy_kwargs={}):
        super().__init__()
        self.controller = controller
        self.model = controller.model
        self.data = controller.data
        
        # Define decode strategy
        sos_token = self.data.trg_vocab.stoi["<sos>"]
        eos_token = self.data.trg_vocab.stoi["<eos>"]
        self.strategy = find_forecast_strategy(strategy)(\
            sos_token=sos_token, eos_token=eos_token, **strategy_kwargs)
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])

    def run(self, src):
        # Create forecast instance
        e_out = self.model.e_out(src)
        N = src.size(0)
        for t, (trg, kwargs) in enumerate(self.strategy.queries(e_out, N)):
            probs = self.model.infer_step(trg, **kwargs)
            self.strategy.update(t, probs)
        return self.strategy.top()
    
       
    