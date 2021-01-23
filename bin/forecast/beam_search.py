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
            replace_unk: (bool) - Whether to replace unk tokens by origin src tokens
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

        indices = torch.arange(1, max_len+1, dtype=torch.long)
        self.register_buffer("indices", indices, persistent=False)

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

        # Scale up attributes to batch size (N), they change after each batch
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
            attn_score (Tensor [m x t x S]) - Encoder-Decoder Attention score
        """
        n, k, q, eos_token = len(self.q), self.k, self.q, self.eos_token
        sents, scores = self.sents, self.scores

        m = probs.size(0)
        assert m % n == 0
        assert attn_score.size(0) == m and attn_score.size(1) == t

        k_prob, k_index = probs.topk(k, dim=-1)
        assert k_index.size() == (m, k) and k_prob.size() == (m, k)

        # First time step
        if t == 1:
            sents[t] = k_index
            self.scores = scores + k_prob.log()
    
        # Follow the first time step
        if t > 1:
            assert m == n * k

            # Detach batch size and beam size
            k_prob, k_index = k_prob.view(n, k, k), k_index.view(n, k, k)
            attn_score = attn_score.view(n, k, t, -1)

            # Preserve eos beams
            eos_mask = (sents[t-1, q] == eos_token).unsqueeze(-1)
            assert eos_mask.size() == (n, k, 1)

            k_prob = k_prob.masked_fill(eos_mask, 1.0)
            k_index = k_index.masked_fill(eos_mask, eos_token)

            pured = scores[q].unsqueeze(-1) + k_prob.log()
            combined = self._compute_score(sents[:t, q], pured, attn_score)

            pured = pured.view(n, -1)
            combined = combined.view(n, -1)
            assert combined.size() == (n, k * k) and pured.size() == (n, k * k)
            
            combined, positions = combined.topk(k, dim=-1)
            assert positions.size() == (n, k)
            scores[q] = pured.gather(dim=-1, index=positions)

            # Restore origin beams indices
            rows, cols = positions // k, positions % k
            sents[:t, q] = sents[:t, q.unsqueeze(1), rows]
            sents[t, q] = k_index[torch.arange(n).unsqueeze(1), rows, cols]

            if t + 1 == self.max_len:
                scores[q] = combined
            else: 
                eos_mask = (sents[t, q] == eos_token)
                scores[q] = scores[q].masked_fill(eos_mask, 0.0)
                scores[q] = scores[q] + combined.masked_fill(eos_mask==0, 0.0)

        # update which sentences finished all its beams
        mask = (sents[t] == eos_token).all(-1)
        tmp = self.are_done ^ mask
        if (tmp != self.are_done).any():
            # Recreate some tmp vars whenever encounter a change
            self.are_done = tmp
            self.q = torch.nonzero(self.are_done == 0).view(-1)
            self._repeat_interleave_keys()
        
    def _compute_score(self, sents, scores, attn_scores):
        """
        Update scores for each timestep
        Args:
            sents: (Tensor [t x n x k])
            scores:  (Tensor [n x k x k]) Current probabilities.
            attn_scores: (Tensor [n x k x t x S]) - Encoder-Decoder attention score
        Returns:
            (Tensor [n x k x k]) - Combined scores 
        """
        t, n, k = sents.size()
        lp, cp = 1.0, 0.0
        eos_mask = (sents[t-1] == self.eos_token)

        if self.alpha > 0.0:
            # Length normalization
            lp = torch.ones((n, k), device=self.device)
            scalar = ((5 + t) / 6) ** self.alpha
            lp = lp * scalar
            lp = lp.masked_fill(eos_mask, 1.0).unsqueeze(-1)
        
        if self.beta > 0.0:
            # Coverage penalty, currently not working properly with Transformer 
            #   without using guided alignment
            cp = attn_scores.sum(dim=2)
            cp = cp.masked_fill(cp > 1.0, 1.0).log().sum(dim=2)
            cp = self.beta * cp
            cp = cp.masked_fill(eos_mask, 0.0).unsqueeze(-1)

        return scores / lp + cp

    def _top(self):
        """
        Get the best beam for each sentence in the batch
        Returns:
            (Tensor [N x T]) - Target sentences generated by the algorithm
        """
        # Get the best beam
        results = [self.sents[:, i, int(j.item())] \
            for i, j in enumerate(self.scores.argmax(dim=-1))]
        return results


class BeamSearch2(Forecaster):
    def __init__(self, k=4, max_len=160, replace_unk=False, \
        alpha=0.0, beta=0.0, gamma=0.0, **kwargs):
        """
        Beam Search simplicity (consume more RAM)
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
        self.replace_unk = replace_unk
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        sent = torch.zeros(max_len, 1, k, dtype=torch.long)
        sent[0] = self.sos_token
        self.register_buffer("sent", sent, persistent=False)

        score = torch.zeros((1, k), dtype=torch.float)
        self.register_buffer("score", score, persistent=False)

        it = torch.arange(self.max_len+1)
        self.register_buffer("it", it, persistent=False)
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
            # All beams from all batches terminated, early stopping
            if (self.sents[t-1] == self.eos_token).all():
                break
            if t == 1:
                # if cur_len = 1, run for the first tokens of k beams
                probs, attn_scores = self.model.infer_step(
                    torch.tensor([self.sos_token] * self.n, \
                        device=self.device).unsqueeze(0), self.memory_info)
            else:
                probs, attn_scores = self.model.infer_step(
                    self.sents[:t].view(t, -1), self.memories_info)
            self._update(t, probs, attn_scores)
        results = self._top()
        # Algorithm ends here
        self._reset()
        return results

    def _initialize(self, src):
        """Create some useful temporary variables for inference"""
        # Encoder output
        n = src.size(1)
        self.n = n
        
        self.memory_info = self.model.encode(src)
        self.memories_info = {}
        # Scale up attributes to batch size (N), they change after each batch
        self.sents = self.sent.repeat(1, n, 1)
        self.scores = self.score.repeat(n, 1)

        for k, v in self.memory_info.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError("Expected Tensor type but got {}".format(type(v)))
            assert v.size(1) == n
            self.memories_info[k] = v.repeat_interleave(self.k, dim=1)
        
    def _reset(self):
        """Free temporary variables after finishing one batch sentence"""
        self.sents = None
        self.pured = None
        self.refined = None
        self.memory_info = None
        self.memories_info = None
    
    def _update(self, t:int, probs, attn_scores):
        """
        Update scores for the next infer step
        Arguments:
            t (int) - decode timestep (current length)
            probs (Tensor [m x vocab_size])
                m = n * k   (if t > 0)
                m = n       (if t = 0)
            attn_score (Tensor [m x t x S]) - Encoder-Decoder Attention score
        """
        n, k, eos_token, sents = self.n, self.k, self.eos_token, self.sents

        m = probs.size(0)
        assert m % n == 0
        assert attn_scores.size(0) == m and attn_scores.size(1) == t

        k_prob, k_index = probs.topk(k, dim=-1)
        assert k_index.size() == (m, k) and k_prob.size() == (m, k)

        # First time step
        if t == 1:
            sents[t] = k_index
            self.pured = k_prob.log()
            self.refined = torch.clone(self.pured)
    
        # Follow the first time step
        if t > 1:
            assert m == n * k

            # Detach batch size and beam size
            k_prob, k_index = k_prob.view(n, k, k), k_index.view(n, k, k)
            attn_scores = attn_scores.view(n, k, t, -1)

            # Preserve eos beams
            eos_mask = (sents[t-1] == eos_token).unsqueeze(-1)
            k_prob = k_prob.masked_fill(eos_mask, 1.0)
            k_index = k_index.masked_fill(eos_mask, eos_token)

            self.pured = self.pured.unsqueeze(-1) + k_prob.log()
            self.refined = self._compute_score(sents[:t], self.pured, attn_scores)

            self.pured = self.pured.view(n, -1)
            self.refined = self.refined.view(n, -1)
            
            self.refined, positions = self.refined.topk(k, dim=-1)
            self.pured = self.pured.gather(dim=-1, index=positions)

            # Restore origin beams indices
            rows, cols = positions // k, positions % k
            it = torch.arange(n).unsqueeze(-1)
            sents[:t] = sents[:t, it, rows]
            sents[t] = k_index[it, rows, cols]
    
    def _compute_score(self, sents, scores, attn_scores):
        """
        Update scores for each timestep
        Args:
            sents: (Tensor [t x n x k])
            scores:  (Tensor [n x k x k]) Current probabilities.
            attn_scores: (Tensor [n x k x t x S]) - Encoder-Decoder attention score
        Returns:
            (Tensor [n x k x k]) - Combined scores 
        """
        t, n, k = sents.size()
        lp, cp = 1.0, 0.0

        if self.alpha > 0.0:
            # Length normalization
            fool = self.it[:t].view(t, 1, 1).repeat(1, n, k)
            lp = (sents == self.eos_token) * fool
            lp = lp.masked_fill(lp == 0, t+1)
            lp[-1] = t
            lp, _ = torch.min(lp, dim=0)
            lp = lp.unsqueeze(-1)
            lp = ((5 + lp) / 6) ** self.alpha
        
        if self.beta > 0.0:
            # Coverage penalty
            cp = attn_scores.sum(dim=2)
            cp = cp.masked_fill(cp > 1.0, 1.0).log().sum(dim=2)
            cp = self.beta * cp
            cp = cp.unsqueeze(-1)

        return scores / lp + cp

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