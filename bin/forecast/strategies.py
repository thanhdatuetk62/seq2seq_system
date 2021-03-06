import torch
from torch import nn

class BeamSearch(nn.Module):
    def __init__(self, e_out, N, sos_token, eos_token, k=4, max_len=160):
        """
        A Simple Beam Search implementation
        Constructor Args:
            e_out (Dict[str, Tensor]) - A dictionary contains some useful \
                tensors for feeding into the Decoder
            N (int): Batch size
            [sos, eos]_token (int) - Index of "<sos>" and "<eos>" token
            k (int) - Beam size
            max_len (int) - Maximum number of time steps for generating output
        """
        super().__init__()

        # Memorize some useful scalars
        self.N = N
        self.k = k
        self.max_len = max_len
        self.sos_token = sos_token
        self.eos_token = eos_token

        # Hypothesises for searching
        sents = torch.zeros(self.N, k, max_len).long()
        sents[:, :, 0] = sos_token

        # Log scores of beams
        scores = torch.zeros((self.N, k), dtype=torch.float)

        # Indices of un-terminated batches
        q = torch.arange(self.N).long()

        # Mask of finished batches
        done = torch.zeros(self.N, dtype=torch.int)

        # Create mask for checking eos
        sent_eos = torch.tensor([eos_token] * k).unsqueeze(0)

        # Register to buffer
        vars_register = [("sents", sents), ("scores", scores), ("q", q),
            ("done", done), ("sent_eos", sent_eos)]
        for name, val in vars_register:
            self.register_buffer(name, val)
        
        # Prepare encoder outputs
        self.e_out = e_out
        self.repeat_interleave_keys()

    def repeat_interleave_keys(self):
        self.e_outs = {}
        for key, value in self.e_out.items():
            # Check value type and format
            if not isinstance(value, torch.Tensor):
                raise ValueError("Expected Tensor type only, but key `{}` got \
{} instead!".format(key, type(value)))
            # Be sure batch size if consistent with initial batch
            assert value.size(0) == self.N
            self.e_outs[key] = torch.repeat_interleave(\
                value[self.q], self.k, dim=0)

    def queries(self):
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
        self.done.masked_fill_(mask, 1)
        self.q = torch.nonzero(self.done == 0).view(-1)
        self.repeat_interleave_keys()
        
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
