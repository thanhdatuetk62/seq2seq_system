import torch
import time
import os

from torch import nn

from ..metrics import find_loss_metric, find_eval_metric
from ..train import find_scheduler, find_optimizer
from ..forecast import find_forecast_strategy
from ..utils import count_params

class _Model(nn.Module):
    def __init__(self, controller, data, device="cpu"):
        super().__init__()
        self.controller = controller
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

    def init_train(self, loss_metric="xent", loss_kwargs={},
              optimizer="adam", optimizer_kwargs={},
              scheduler="noam", scheduler_kwargs={}, 
              eval_metric=None, eval_config={}):
    
        if count_params(self) == 0:
            raise RuntimeError("Model not found. Be sure to define model \
architecture before proceed training !")

        # Create loss function for training and validation (default: xent)
        # self.loss_metric = find_loss_metric(loss_metric)(
        #     ignore_index=data.trg_vocab.stoi["<pad>"],
        #     tgt_vocab_size=len(data.trg_vocab), **loss_kwargs)
        self.loss_metric = nn.CrossEntropyLoss(
            ignore_index=self.data.trg_vocab.stoi["<pad>"])

        # Create optimizer
        optimizer = find_optimizer(optimizer)(self.parameters(),
                                              **optimizer_kwargs)

        # Create learning rate scheduler
        if scheduler is not None:
            self.optimizer = find_scheduler(scheduler)(optimizer,
                                                       **scheduler_kwargs)
        else:
            self.optimizer = optimizer

        if eval_metric is not None:
            # Create eval metric (default: BLEU)
            self.eval_metric = find_eval_metric(eval_metric)
            self.init_infer(**eval_config)
        
        # Cast model to a specifice device type (CPU <-> CUDA)
        self.to(self.device)
    
    def init_infer(self, forecast_strategy="beam_search", strategy_kwargs={}):
        # Define strategy
        self.strategy_cls = find_forecast_strategy(forecast_strategy)
        self.strategy_kwargs = strategy_kwargs
        
        self.to(self.device)

    def train_step(self):
        raise NotImplementedError
    
    def e_out(self):
        raise NotImplementedError

    def validate_step(self):
        raise NotImplementedError

    def forecast(self, src):
        # Create forecast instance
        sos_token = self.data.trg_vocab.stoi["<sos>"]
        eos_token = self.data.trg_vocab.stoi["<eos>"]
        e_out = self.e_out(src)

        strategy = self.strategy_cls(e_out, src.size(0), sos_token, eos_token,
                                     **self.strategy_kwargs).to(self.device)

        # Start infer steps for this batch
        for t, (trg, kwargs) in enumerate(strategy.queries()):
            probs = self.infer_step(trg, **kwargs)
            strategy.update(t, probs)

        return strategy.top()
    
    def infer_step(self):
        raise NotImplementedError

    def state_dict(self):
        return {
            "model": super().state_dict(),
            "optimizer": self.optimizer.state_dict()
            if hasattr(self, "optimizer") else None,
            "epoch": self.epoch
            if hasattr(self, "epoch") else None
        }

    def load_state_dict(self, state):
        super().load_state_dict(state["model"])
        # Try to load training config (for training purpose)
        if hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(state["optimizer"])
