import torch
from torch import nn
from .forecaster import Forecaster


class BeamSearch(Forecaster):
    def __init__(self, k=4, max_len=160, replace_unk=False, \
        alpha=0.0, beta=0.0, gamma=0.0, **kwargs):
        """
        Beam Search implementation
        Constructor Args:
            k: (int) - Beam size
            max_len: (int) - Maximum number of time steps for generating output
            replace_unk: (bool) - Whether to replace unk word by origin src tokens
            alpha: (float) - Length normalization coefficient
            beta: (float) - Coverage normalization coefficient
            gamma: (float) - End of sentence normalization
        """
        super().__init__(**kwargs)
        # Memorize some useful scalars
        self.k = k
        self.max_len = max_len
        self.replace_unk = replace_unk
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Hypothesises for searching
        sent = torch.zeros(1, k, max_len).long()
        sent[:, :, 0] = self.sos_token
        score = torch.zeros((1, k), dtype=torch.float)
        is_done = torch.zeros(1, dtype=torch.int)
        sent_eos = torch.tensor([self.eos_token] * k).unsqueeze(0)
        
        # Register these hypo to module's buffer (Useful when casting device)
        vars_register = [("sent", sent), ("score", score),
                         ("is_done", is_done), ("sent_eos", sent_eos)]
        for name, val in vars_register:
            self.register_buffer(name, val, persistent=False)

        # Init tmp vars
        self._disable()
    
    def forward(self, src):
        """
        Infer a source batch sentence
        """
        self._enable(src)
        # Algorithm starts here
        for t in range(1, self.max_len):
            # All beams from all batches terminated, early stopping process
            if len(self.q) == 0:
                break
            if t == 1:
                # if cur_len = 1, run for the first tokens of k beams
                probs, attn_scores = self.model.infer_step(
                    torch.tensor([self.sos_token] * len(self.q), \
                        device=self.sents.device).unsqueeze(1), \
                        self.memory, src)
            else:
                probs, attn_scores = self.model.infer_step(
                    self.sents[self.q, :, :t].view(len(self.q) * self.k, -1), 
                    self.memories, src)
            self._update(t, probs, attn_scores)
        results = self._top()
        # Algorithm ends here
        self._disable()
        return results

    def _enable(self, src):
        """
        Create some useful temporary variables for inference
        """
        # Encoder output
        n = src.size(0)
        self.memory = self.model.memory(src)
        self.q = torch.arange(n).long()

        # Scale-up attributes to batch size (N), they change after each batch
        self.sents = self.sent.repeat(n, 1, 1)
        self.scores = self.score.repeat(n, 1)
        self.are_done = self.is_done.repeat(n)
        self.memories = torch.repeat_interleave(self.memory[self.q], self.k, dim=0)

    def _disable(self):
        """Free temporary variables after finishing one batch sentence"""
        DEFAULT_TENSOR_VALUE = torch.tensor(-1).long()

        self.q = DEFAULT_TENSOR_VALUE
        self.sents = DEFAULT_TENSOR_VALUE
        self.scores = DEFAULT_TENSOR_VALUE.float()
        self.are_done = DEFAULT_TENSOR_VALUE
        self.memory = DEFAULT_TENSOR_VALUE.float()
        self.memories = DEFAULT_TENSOR_VALUE.float()
        self.att_scores = []
    
    def _update(self, t:int, probs, attn_scores):
        """
        Update scores for the next infer step
        Arguments:
            t (int) - decode timestep (current length)
            probs (Tensor [m x vocab_size])
                m = n * k   (if t > 0)
                m = n       (if t = 0) 
            attn_scores (Tensor [m x t x S]) - Encoder-Decoder attention score
        """
        n, k, q, eos_token = len(self.q), self.k, self.q, self.eos_token
        sents, scores, sent_eos = self.sents, self.scores, self.sent_eos

        m = probs.size(0)
        assert m % n == 0

        k_prob, k_index = probs.topk(k, dim=-1)
        assert k_index.size() == (m, k)
        assert k_prob.size()  == (m, k)

        # First time step
        if t == 1:
            sents[:, :, 1] = k_index
            scores += k_prob.log()
    
        # Follow the first time step
        if t > 1:
            assert m == n * k

            # Detach batch size and beam size
            k_prob, k_index = k_prob.view(n, k, k), k_index.view(n, k, k)
            attn_scores = attn_scores.view(n, k, t, -1)

            # Preserve eos beams
            eos_mask = (sents[q, :, t-1] == eos_token).view(n, k, 1)
            k_prob.masked_fill_(eos_mask, 1.0)
            k_index.masked_fill_(eos_mask, eos_token)

            # [n x k x 1] +  [n x k x k] = [n x k x k]
            combine_prob = self._compute_score(scores[q], k_prob, t, \
                attn_scores).view(n, int(k ** 2))
            
            # [n x k], [n x k]
            scores[q], positions = combine_prob.topk(k, dim=-1)

            # The rows selected from top k
            rows = positions // k
            # The indexes in vocab respected to these rows
            cols = positions % k

            id = torch.arange(n).unsqueeze(1)
            sents[q, :, :] = sents[q.unsqueeze(1), rows, :]
            sents[q, :, t] = k_index[id, rows, cols].view(n, k)

        # update which sentences finished all its beams
        mask = (sents[:, :, t] == sent_eos).all(1).view(-1)
        tmp = self.are_done.masked_fill(mask, 1)
        if not (tmp == self.are_done).all():
            # Recreate some tmp vars whenever encounter a change
            self.are_done = tmp
            self.q = torch.nonzero(self.are_done == 0).view(-1)
            self.memories = torch.repeat_interleave(self.memory[self.q], self.k, dim=0)
    
    def _compute_score(self, scores, k_prob, t:int, attn_scores):
        """
        Update scores each timestep
        Args:
            scores:  (Tensor [n x k])
                - Current probabilities.
            k_prob: (Tensor [n x k x k]) 
                - Probabilities of the next tokens for each beam from each 
                    sentence of the batch.
            t: (int) - current length
            attn_scores: (Tensor [n x k x t x S]) - Encoder-Decoder attention score
        Returns:
            (Tensor [n x k x k]) - Combined scores 
        """
        scores = scores.unsqueeze(-1)
        k_prob = torch.log(k_prob)
        
        # Length normalization
        lp = (5 + t) ** self.alpha / (5 + 1) ** self.alpha
        
        # Coverage penalty
        # [n x k x S]
        cp = attn_scores.sum(dim=2)
        cp.masked_fill_(cp > 1.0, 1.0)
        # [n x k x 1]
        cp = self.beta * cp.log().sum(dim=-1).unsqueeze(-1)

        return (scores + k_prob) / lp + cp

    def _top(self):
        """
        Get the best beam for each sentence in the batch
        Returns:
            (Tensor [N x T]) - Target sentences generated by the algorithm
        """
        # Get the best beam
        results = [self.sents[t, int(j.item()), :] \
            for t, j in enumerate(self.scores.argmax(dim=-1))]
        return results
