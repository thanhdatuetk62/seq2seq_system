import torch
from torch import nn
from .strategy import _Strategy
from ...models import DecodeSupport


class BeamSearch(_Strategy):
    def __init__(self, k=4, max_len=160, replace_unk=False, \
        alpha=0.0, beta=0.0, gamma=0.0, support_kwargs={}, **kwargs):
        """
        Beam Search implementation
        Constructor Args:
            k: (int) - Beam size
            max_len: (int) - Maximum number of time steps for generating output
            replace_unk: (bool) - Whether to replace unk tokens by origin src tokens
            alpha: (float) - Length normalization coefficient
            beta: (float) - Coverage normalization coefficient
            gamma: (float) - End of sentence normalization
        """
        super().__init__(**kwargs)
        # Memorize some useful scalars
        self.k = k
        self.max_len = max_len
        self.support_kwargs = support_kwargs
        self.replace_unk = replace_unk
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        sent = torch.zeros(max_len, 1, k, dtype=torch.long)
        self.register_buffer("sent", sent, persistent=False)

        score = torch.zeros((1, k), dtype=torch.float)
        self.register_buffer("score", score, persistent=False)

        it = torch.arange(self.max_len+1)
        self.register_buffer("it", it, persistent=False)

        # Init tmp vars
        self._reset()
    
    def forward(self, src, sos_tokens=None):
        """
        Infer a source batch sentence
        Arguments:
            src: (Tensor [S x N])
        """
        self._initialize(src, sos_tokens)
        # Algorithm starts here
        for t in range(1, self.max_len):
            # All beams from all batches terminated, early stopping process
            if (self.sents[t-1] == self.eos_token).all():
                break
            if t == 1:
                probs = self.support.burn()
            else:
                probs = self.support(self.sents[:t])
            self._update(t, probs)
        results = self._top()
        # Algorithm ends here
        self._reset()
        return results

    def _initialize(self, src, sos_tokens=None):
        """Create some useful temporary variables for inference"""
        # Encoder output
        n = src.size(1)
        self.n = n

        self.sos_tokens = sos_tokens if sos_tokens is not None \
            else torch.tensor([self.sos_token] * n, device=self.device)

        # Scale up attributes to batch size (N), they change after each batch
        self.sents = self.sent.repeat(1, n, 1)
        self.sents[0] = self.sos_tokens.unsqueeze(-1)
        
        self.scores = self.score.repeat(n, 1)
        
        # Initialize decode class
        self.support = DecodeSupport(self.model, self.k, self.eos_token, \
            self.sos_tokens, src, self.device, **self.support_kwargs)
        
    def _reset(self):
        """Free temporary variables after finishing one batch sentence"""
        self.sents = None
        self.scores = None
        self.support = None
    
    def _update(self, t:int, probs):
        """
        Update scores for the next infer step
        Arguments:
            t (int) - decode timestep (current length)
            probs (Tensor [n x k x vocab_size] or Tensor [n x vocab_size])
        """
        n, k, eos_token, sents = self.n, self.k, self.eos_token, self.sents
        assert probs.size(0) == n

        k_prob, k_index = probs.topk(k, dim=-1)

        # First time step
        if t == 1:
            assert k_prob.size() == (n, k)
            assert k_index.size() == (n, k)

            sents[t] = k_index
            self.pured = k_prob.log()
            self.refined = torch.clone(self.pured)
    
        if t > 1:
            # Follow the first time step
            assert k_prob.size() == (n, k, k)
            assert k_index.size() == (n, k, k)

            # Preserve eos beams
            eos_mask = (sents[t-1] == eos_token).unsqueeze(-1)
            k_prob = k_prob.masked_fill(eos_mask, 1.0)
            k_index = k_index.masked_fill(eos_mask, eos_token)

            self.pured = self.pured.unsqueeze(-1) + k_prob.log()
            self.refined = self._compute_score(sents[:t], self.pured)

            self.pured = self.pured.view(n, -1)
            self.refined = self.refined.view(n, -1)
            
            self.refined, positions = self.refined.topk(k, dim=-1)
            self.pured = self.pured.gather(dim=-1, index=positions)

            # Restore origin beams indices
            rows, cols = positions // k, positions % k

            # Eliminate and order beams in decoder's cache
            indices = (torch.arange(self.n, device=self.device).\
                unsqueeze(-1) * self.k + rows).view(-1)
            self.support.reorder_beams(indices)

            it = torch.arange(n).unsqueeze(-1)
            sents[:t] = sents[:t, it, rows]
            sents[t] = k_index[it, rows, cols]
    
    def _compute_score(self, sents, scores):
        """
        Update scores for each timestep
        Args:
            sents: (Tensor [t x n x k])
            scores:  (Tensor [n x k x k]) Current probabilities.
        Returns:
            (Tensor [n x k x k]) - Combined scores 
        """
        t, n, k = sents.size()
        lp = 1.0

        if self.alpha > 0.0:
            # Length normalization
            fool = self.it[:t].view(t, 1, 1).repeat(1, n, k)
            lp = (sents == self.eos_token) * fool
            lp = lp.masked_fill(lp == 0, t+1)
            lp[-1] = t
            lp, _ = torch.min(lp, dim=0)
            lp = lp.unsqueeze(-1)
            lp = ((5 + lp) / 6) ** self.alpha
        return scores / lp

    def _top(self):
        """
        Get the best beam for each sentence in the batch
        Returns:
            (Tensor [N x T]) - Target sentences generated by the algorithm
        """
        # Get the best beam
        results = [self.sents[:, i, int(j.item())] \
            for i, j in enumerate(self.refined.argmax(dim=-1))]
        return results