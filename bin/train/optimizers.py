from torch.optim import Adam

class AdamOptimizer(Adam):
    def __init__(self, params, beta1, beta2, eps, lr):
        super(AdamOptimizer, self).__init__(params, betas=(beta1, beta2), \
            eps=eps, lr=lr)