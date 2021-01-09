from .optimizers import AdamOptimizer
from .schedulers import NoamScheduler

schedulers = {"noam": NoamScheduler}
optimizers = {"adam": AdamOptimizer}

def find_optimizer(optimizer):
    if optimizer not in optimizers:   
        raise ValueError("Optimizer {} did not exist in our system".\
            format(optimizer))
    return optimizers[optimizer]

def find_scheduler(scheduler):
    if scheduler not in schedulers:
        raise ValueError("Scheduler {} did not exist in our system".\
            format(scheduler))
    return schedulers[scheduler]