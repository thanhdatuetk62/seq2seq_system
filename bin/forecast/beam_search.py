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

        sent = torch.zeros(max_len, 1, k, dtype=torch.long)
        sent[0] = self.sos_token
        self.register_buffer("sent", sent, persistent=False)

        score = torch.zeros((1, k), dtype=torch.float)
        self.register_buffer("score", score, persistent=False)

        is_done = torch.zeros(1, dtype=torch.int)
        self.register_buffer("is_done", is_done, persistent=False)

        # Init tmp vars
        self._reset()
    
    def forward(self, src):
        """
        Infer a source batch sentence
        Arguments:
            src: (Tensor [S x N])
        """
        self._initialize(src)
        # Algorithm starts here
        for t in range(1, self.max_len):
            # All beams from all batches terminated, early stopping process
            if len(self.q) == 0:
                break
            if t == 1:
                # if cur_len = 1, run for the first tokens of k beams
                probs, attn_score = self.model.infer_step(
                    torch.tensor([self.sos_token] * len(self.q), \
                        device=self.device).unsqueeze(0), self.memory_info)
            else:
                probs, attn_score = self.model.infer_step(
                    self.sents[:t, self.q].view(t, -1), self.memories_info)
            self._update(t, probs, attn_score)
        results = self._top()
        # Algorithm ends here
        self._reset()
        return results

    def _repeat_interleave_keys(self):
        self.memories_info = {k : v[:, self.q].repeat_interleave(self.k, dim=1) \
            for k, v in self.memory_info.items()}

    def _initialize(self, src):
        """Create some useful temporary variables for inference"""
        # Encoder output
        n = src.size(1)
        self.memory_info = self.model.encode(src)
        self.q = torch.arange(n).long()

        # Scale-up attributes to batch size (N), they change after each batch
        self.sents = self.sent.repeat(1, n, 1)
        self.scores = self.score.repeat(n, 1)
        self.are_done = self.is_done.repeat(n)

        for k, v in self.memory_info.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError("Expected Tensor type but got {}".format(type(v)))
            assert v.size(1) == n
        self._repeat_interleave_keys()
        
    def _reset(self):
        """Free temporary variables after finishing one batch sentence"""
        self.q = None
        self.sents = None
        self.scores = None
        self.are_done = None
        self.memory_info = None
        self.memories_info = None
    
    def _update(self, t:int, probs, attn_score):
        """
        Update scores for the next infer step
        Arguments:
            t (int) - decode timestep (current length)
            probs (Tensor [m x vocab_size])
                m = n * k   (if t > 0)
                m = n       (if t = 0)
        """
        n, k, q, eos_token = len(self.q), self.k, self.q, self.eos_token
        sents, scores = self.sents, self.scores

        m = probs.size(0)
        assert m % n == 0

        k_prob, k_index = probs.topk(k, dim=-1)
        assert k_index.size() == (m, k)
        assert k_prob.size()  == (m, k)

        # First time step
        if t == 1:
            sents[t] = k_index
            scores = scores + k_prob.log()
    
        # Follow the first time step
        if t > 1:
            assert m == n * k

            # Detach batch size and beam size
            k_prob, k_index = k_prob.view(n, k, k), k_index.view(n, k, k)

            # Preserve eos beams
            eos_mask = (sents[t-1, q] == eos_token).unsqueeze(-1)
            assert eos_mask.size() == (n, k, 1)

            k_prob = k_prob.masked_fill(eos_mask, 1.0)
            k_index = k_index.masked_fill(eos_mask, eos_token)

            combine_prob = scores[q].unsqueeze(-1) + k_prob.log()
            combine_prob = self._compute_score(combine_prob, t).view(n, -1)
            assert combine_prob.size() == (n, k * k)
            
            scores[q], positions = combine_prob.topk(k, dim=-1)
            assert positions.size() == (n, k)

            # Restore origin beams indices
            rows, cols = positions // k, positions % k
            sents[:t, q] = sents[:t, q.unsqueeze(1), rows]
            sents[t, q] = k_index[torch.arange(n).unsqueeze(1), rows, cols]

        # update which sentences finished all its beams
        mask = (sents[t] == eos_token).all(-1)
        tmp = self.are_done.masked_fill(mask, 1)
        if (tmp == self.are_done).any():
            # Recreate some tmp vars whenever encounter a change
            self.are_done = tmp
            self.q = torch.nonzero(self.are_done == 0).view(-1)
            self._repeat_interleave_keys()
    
    def _compute_score(self, scores, t:int):
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
        return scores

    def _top(self):
        """
        Get the best beam for each sentence in the batch
        Returns:
            (Tensor [N x T]) - Target sentences generated by the algorithm
        """
        # Get the best beam\
        results = [self.sents[:, i, int(j.item())] \
            for i, j in enumerate(self.scores.argmax(dim=-1))]
        return results
