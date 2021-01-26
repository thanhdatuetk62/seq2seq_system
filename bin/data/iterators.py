from torchtext.data import Batch
from random import shuffle

def exbatch(batch_size, examples):
    batches = []
    for i in range(0, len(examples), batch_size):
        batches.append(examples[i:i+batch_size])
    return batches
    
def dynbatch(n_tokens, examples, fields):
    cnt = {k: 0 for k in fields.keys()}
    batches, batch = [], []
    for ex in examples:
        batch.append(ex)
        for k in fields.keys():
            if hasattr(ex, k):
                val = getattr(ex, k)
                assert type(val) == list
                cnt[k] += len(val)
        if max(cnt.values()) >= n_tokens:
            batches.append(batch)
            batch = []
            cnt = {k: 0 for k in fields.keys()}
    if len(batch) > 0:
        batches.append(batch)
    return batches


class Iterator(object):
    def __init__(self, ds, batch_size=64, n_tokens=None, train=False, device="cpu"):
        self.ds = ds
        self.batch_size = batch_size
        self.n_tokens = n_tokens
        self.train = train
        self.device = device
        
        self.init_epoch()

    def __iter__(self):
        self.init_epoch()
    
        examples = self.ds.examples
        fields = self.ds.fields
        
        if self.n_tokens is not None:
            batches = dynbatch(self.n_tokens, examples, fields)
        else:
            batches = exbatch(self.batch_size, examples)
        
        for batch in batches:
            yield Batch(batch, dataset=self.ds, device=self.device)

    def init_epoch(self):
        if self.train:
            shuffle(self.ds.examples)

