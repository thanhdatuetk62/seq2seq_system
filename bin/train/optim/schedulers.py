from torch.optim import Optimizer
import torch

class NoamScheduler(object):
    def __init__(self, optimizer, scalar=0.2, d_model=512, \
        warmup_steps=4000):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step_count = 0
        self.scalar = scalar

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @torch.no_grad()
    def step(self):
        """
        Update learning rate and then parameters
        """
        self._step_count += 1
        for group in self.optimizer.param_groups:
            group["lr"] = self.scalar * ((self.d_model ** -0.5) * \
                min(self._step_count ** -0.5, self._step_count * \
                    self.warmup_steps ** -1.5))
        self.optimizer.step()
    
    def zero_grad(self, **kwargs):
        self.optimizer.zero_grad(**kwargs)

    def state_dict(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "step_count": self._step_count,
            "scalar": self.scalar,
            "optimizer": self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.d_model = state_dict["d_model"]
        self.warmup_steps = state_dict["warmup_steps"]
        self._step_count = state_dict["step_count"]
        self.scalar = state_dict["scalar"]
        self.optimizer.load_state_dict(state_dict["optimizer"])
    
    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


