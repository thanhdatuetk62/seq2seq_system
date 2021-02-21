import io
import linecache
from os import stat
import random
from torchtext.data import Batch, interleave_keys, Example
from torchtext.data.iterator import batch
from torchtext.vocab import Vocab
from random import shuffle
from collections import Counter

from .io_handlers import load_bitext, load_monotext
from .utils import standard_batch, dynamic_batch, statistic


class Batch(object):
    def __init__(self, data, fields, device="cpu"):
        self.batch_size = len(data)
        for name, field in fields:
            if field is not None:
                batch = [getattr(x, name) for x in data]
                setattr(self, name, field.process(batch, device=device))


class EagerLoader(object):
    def __init__(self, fields, batch_size=32, n_tokens=None, train=False, \
        sampling=False, device="cpu", bitext=True, **corpora):
        self.fields = fields
        self.device = device
        self.sampling = sampling
        self.corpora = corpora
        self.train = train
        self.n_tokens = n_tokens
        self.batch_size = batch_size
        self.bitext = bitext

        # Gather and compose corpora together (for further actions)
        self.n_sents = 0
        self.owner = []
        self.lengths = []
        self.probs = []
        self.line_id = []
        for name, corpus in corpora.items():
            stats = statistic(name, corpus, bitext)
            self.lengths += stats["lengths"]
            self.n_sents += stats["n_sents"]
            self.probs += [stats["weight"] / stats["n_sents"]] * stats["n_sents"]
            self.owner += [name] * stats["n_sents"]
            self.line_id += stats["line_id"]

        # this method is implemented differently depend on loader type
        self.prepare()
    
    def prepare(self):
        self.sents = []
        for name, corpus in self.corpora.items():
            if self.bitext:
                for s, t, i in load_bitext(**corpus):
                    self.sents.append((s, t))
            else:
                for s, i in load_monotext(**corpus):
                    self.sents.append(s)
    
    def get_line(self, i):
        return self.sents[i]

    def __len__(self):
        return self.n_sents

    def __iter__(self):
        print("Iterate over corpora in Eager mode")
        id = [i for i in range(self.n_sents)]

        # Shuffle if loading for training and not using sampling
        if self.train and not self.sampling:
            shuffle(id)
        
        batch_method = standard_batch
        batch_size = self.batch_size
        use_sampling = (self.train and self.sampling)
        use_sort = (not self.train)

        if self.n_tokens is not None:
            batch_method = dynamic_batch
            batch_size = self.n_tokens
        
        for batch_id in batch_method(batch_size, id, self.lengths, \
            self.probs, sort=use_sort, sampling=use_sampling):
            batch, pos = [], []
            for i in batch_id:
                corpus = self.corpora[self.owner[i]]
                pos.append((self.line_id[i], self.owner[i]))

                if self.bitext:
                    # Add prefix in sentence if needed
                    src_prefix = corpus.get("src_prefix", "")
                    trg_prefix = corpus.get("trg_prefix", "<sos>")

                    # Get sentence based on this id
                    src, trg = self.get_line(i)
                    src = (src_prefix + " " + src).strip()
                    trg = (trg_prefix + " " + trg).strip()

                    # Add it to the batch list
                    batch.append(Example.fromlist([src, trg], self.fields))
                else:
                    # Add prefix in sentence if needed
                    src_prefix = corpus.get("src_prefix", "")

                    # Get sentence based on this id
                    src = self.get_line(i)
                    src = (src_prefix + " " + src).strip()

                    # Add it to the batch list
                    batch.append(Example.fromlist([src], self.fields))
            
            # Convert to Tensor
            yield Batch(batch, self.fields, device=self.device), pos


class LazyLoader(EagerLoader):

    def prepare(self):
        "No need to prepare in lazy mode"
        return

    def getline(self, i):
        corpus = self.corpora[self.owner[i]]
        src_path = corpus["path"] + '.' + corpus["src_lang"]
        trg_path = corpus["path"] + '.' + corpus["trg_lang"]
        src = linecache.getline(src_path, self.line_id[i])
        trg = linecache.getline(trg_path, self.line_id[i])
        return src, trg
