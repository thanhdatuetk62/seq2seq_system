import torch
from torch import nn
from .forecaster import Forecaster


class BeamSearch(Forecaster):
    def __init__(self, k=4, max_len=160, **kwargs):
        """
        A Simple Beam Search implementation
        Constructor Args:
            k (int) - Beam size
            max_len (int) - Maximum number of time steps for generating output
        """
        super().__init__(**kwargs)
        # Memorize some useful scalars
        self.k = k
        self.max_len = max_len

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
                probs = self.model.infer_step(
                    torch.tensor([self.sos_token] * len(self.q), \
                        device=self.sents.device).unsqueeze(1), \
                        self.memory, src)
            else:
                probs = self.model.infer_step(
                    self.sents[self.q, :, :t].view(len(self.q) * self.k, -1), 
                    self.memories, src)
            self._update(t, probs)
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
    
    def _update(self, t:int, probs):
        """
        Update scores for the next infer step
        Arguments:
            t (int) - decode timestep
            probs (Tensor [m x vocab_size])
                m = n * k   (if t > 0)
                m = n       (if t = 0) 
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
            # Deflatten k_prob & k_index
            k_prob, k_index = k_prob.view(n, k, k), k_index.view(n, k, k)

            # Preserve eos beams
            eos_mask = (sents[q, :, t-1] == eos_token).view(n, k, 1)
            k_prob.masked_fill_(eos_mask, 1.0)
            k_index.masked_fill_(eos_mask, eos_token)

            # [n x k x 1] +  [n x k x k] = [n x k x k]
            combine_prob = (scores[q].unsqueeze(-1) + k_prob.log()).\
                view(n, int(k ** 2))
            
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
