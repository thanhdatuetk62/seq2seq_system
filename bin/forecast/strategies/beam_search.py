import torch
from torch import nn

class BeamSearch(nn.Module):
    def __init__(self, sos_token, eos_token, k=4, max_len=160):
        """
        A Simple Beam Search implementation
        Constructor Args:
            [sos/eos]_token (int) - Index of "<sos>" and "<eos>" token
            k (int) - Beam size
            max_len (int) - Maximum number of time steps for generating output
        """
        super().__init__()
        # Memorize some useful scalars
        self.k = k
        self.max_len = max_len
        self.sos_token = sos_token
        self.eos_token = eos_token

        # Hypothesises for searching
        sent = torch.zeros(1, k, max_len).long()
        sent[:, :, 0] = sos_token

        # Log scores of beams
        score = torch.zeros((1, k), dtype=torch.float)

        # Mask of finished batches
        is_done = torch.zeros(1, dtype=torch.int)

        # Create mask for checking eos
        sent_eos = torch.tensor([eos_token] * k).unsqueeze(0)
        
        # Register to buffer
        vars_register = [("sent", sent), ("score", score),
            ("is_done", is_done), ("sent_eos", sent_eos)]
        for name, val in vars_register:
            self.register_buffer(name, val, persistent=False)

    def repeat_interleave_keys(self):
        e_outs = {}
        for key, value in self.e_out.items():
            # Check value type and format
            if not isinstance(value, torch.Tensor):
                raise ValueError("Expected Tensor type only, but key `{}` got \
{} instead!".format(key, type(value)))
            # Be sure batch size if consistent with initial batch
            assert value.size(0) == self.N
            e_outs[key] = torch.repeat_interleave(value[self.q], self.k, dim=0)
        return e_outs

    def queries(self, e_out, N):
        """
        Create sort of queries which will be sent to the model for getting
        score distribution.
        Arguments:
            e_out: (Dict[str, Tensor]) - Encoder info from the model
            N: (int) - current batch size
        return:
            An Iterator object contains queries
        """
        # Scale-up attributes to batch size (N), they change after each batch
        self.q = torch.arange(N).long()
        self.N = N
        self.sents = self.sent.repeat(N, 1, 1)
        self.scores = self.score.repeat(N, 1)
        self.are_done = self.is_done.repeat(N)
        self.e_out = e_out
        self.e_outs = self.repeat_interleave_keys()
        # Iterable object start here
        for cur_len in range(1, self.max_len):
            # All beams from all batches terminated, early stopping process
            if len(self.q) == 0:
                break
            if cur_len > 1:
                yield self.sents[self.q, :, :cur_len].view(\
                    len(self.q) * self.k, -1), self.e_outs
            else:
                # if cur_len = 1, run for the first tokens of k beams
                yield torch.tensor([self.sos_token] * len(self.q), \
                    device=self.sents.device).unsqueeze(1), self.e_out
    
    def update(self, t, probs):
        """
        Update scores for the next infer step
        Arguments:
            t (int) - decode timestep
            probs (Tensor [m x vocab_size])
                m = n * k   (if t > 0)
                m = N       (if t = 0) 
        """
        n, k, q, eos_token = len(self.q), self.k, self.q, self.eos_token
        sents, scores, sent_eos = self.sents, self.scores, self.sent_eos

        m, vocab_size = probs.size()
        assert m % n == 0

        k_prob, k_index = probs.topk(k, dim=-1)
        assert k_index.size() == (m, k)
        assert k_prob.size()  == (m, k)

        # First time step
        if t == 0:
            sents[:, :, 1] = k_index
            scores += k_prob.log()
    
        # Follow the first time step
        if t > 0:
            assert m == n * k
            # Deflatten k_prob & k_index
            k_prob, k_index = k_prob.view(n, k, k), k_index.view(n, k, k)

            # Preserve eos beams
            eos_mask = (sents[q, :, t] == eos_token).view(n, k, 1)
            k_prob.masked_fill_(eos_mask, 1.0)
            k_index.masked_fill_(eos_mask, eos_token)

            # [n x k x 1] +  [n x k x k] = [n x k x k]
            combine_prob = (scores[q].unsqueeze(-1) + k_prob.log()).\
                view(n, k ** 2)
            
            # [n x k], [n x k]
            scores[q], positions = combine_prob.topk(k, dim=-1)

            # The rows selected from top k
            rows = positions // k
            # The indexes in vocab respected to these rows
            cols = positions % k

            id = torch.arange(n).unsqueeze(1)
            sents[q, :, :] = sents[q.unsqueeze(1), rows, :]
            sents[q, :, t+1] = k_index[id, rows, cols].view(n, k)

        # update which sentences finished all its beams
        mask = (sents[:, :, t+1] == sent_eos).all(1).view(-1)
        self.are_done.masked_fill_(mask, 1)
        self.q = torch.nonzero(self.are_done == 0).view(-1)
        self.e_outs = self.repeat_interleave_keys()
        
    def top(self):
        """
        Get the best beam for each sentence in the batch
        Returns:
            (Tensor [N x T]) - Target sentences generated by the algorithm
        """
        # Get the best beam
        results = [self.sents[t, j.item(), :] \
            for t, j in enumerate(self.scores.argmax(dim=-1))]
        return results
