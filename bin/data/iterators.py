import io
from torchtext.data import Batch, interleave_keys
from torchtext.vocab import Vocab
from random import shuffle
from collections import Counter

def _filter(src, trg, src_max_len=None, trg_max_len=None):
    """
    Create filter_pred function for filtering out sentences \
        whose length are greater than {max_len}. 
    
    Returns a callable max_len specified filter function
    """
    src_len = len(src.split())
    trg_len = len(trg.split())
    return (src_max_len is None or (src_len <= src_max_len)) \
        and (trg_max_len is None or (trg_len <= trg_max_len))

def _statistic(corpus):
    """Get stats from parallel corpus"""
    n_sents = 0
    lengths = []
    with io.open(corpus["src_path"], "r", encoding="utf-8") as sf, \
         io.open(corpus["trg_path"], "r", encoding="utf-8") as tf:
        for s, t in zip(sf, tf):
            s = s.strip()
            t = t.strip()
            src_max_len = corpus.get("src_max_len", None)
            trg_max_len = corpus.get("trg_max_len", None)
            if s != "" and t != "" and _filter(s, t, src_max_len, trg_max_len):
                lengths.append((len(s.split()), len(t.split())))
                n_sents += 1
    return {"n_sents": n_sents, "lengths": lengths}


def dynamic_batch(n_tokens, id, lengths, keep_order=False):
    batch, cnt = [], 0
    max_src, max_trg = 0, 0
    if not keep_order:
        chunk = 10000
        for i in range(0, len(id), chunk):
            p_id = sorted(id[i:i+chunk], \
                key=lambda x: interleave_keys(*lengths[x]))
            for i in p_id:
                src_len, trg_len = lengths[i]
                sz = max(max_src, max_trg, src_len, trg_len) * (cnt + 1)
                if sz > n_tokens:
                    yield batch
                    batch, cnt = [], 0
                    max_src, max_trg = 0, 0
                cnt += 1
                max_src = max(max_src, src_len)
                max_trg = max(max_trg, trg_len)
                batch.append(i)
            if len(batch) > 0:
                yield batch
    else:
        for i in id:
            src_len, trg_len = lengths[i]
            sz = max(max_src, max_trg, src_len, trg_len) * (cnt + 1)
            if sz > n_tokens:
                yield batch
                batch, cnt = [], 0
                max_src, max_trg = 0, 0
            cnt += 1
            max_src = max(max_src, src_len)
            max_trg = max(max_trg, trg_len)
            batch.append(i)
        if len(batch) > 0:
            yield batch


def standard_batch(batch_size, id, lengths, keep_order=False):
    if not keep_order:
        chunk = batch_size * 100
        for i in range(0, len(id), chunk):
            p_id = sorted(id[i:i+chunk], \
                key=lambda x: interleave_keys(*lengths[x]))
            for j in range(0, len(p_id), batch_size):
                yield id[j:j+batch_size]
    else:
        for i in range(0, len(id), batch_size):
            yield id[i:i+batch_size]


class EagerLoader(object):
    def __init__(self, corpora, n_tokens=None, batch_size=32, train=False):
        self.corpora = corpora
        self.train = train
        self.n_tokens = n_tokens
        self.batch_size = batch_size
        self.stat = {k: _statistic(v) for k, v in corpora.items()}
    
    def build_vocab(self, src_min_freq=1, trg_min_freq=1, src_max_size=None, \
        trg_max_size=None, src_specials=("<unk>", "<pad>"), \
        trg_specials=("<unk>", "<pad>")):

        print("Building vocab ...")
        src_cnt, trg_cnt = Counter(), Counter()
        for name, corpus in self.corpora.items():
            print("Loading corpus {} ...".format(name), end=' ', flush=True)
            with io.open(corpus["src_path"], "r", encoding="utf-8") as sf, \
                 io.open(corpus["trg_path"], "r", encoding="utf-8") as tf:
                for s, t in zip(sf, tf):
                    s = s.strip()
                    t = t.strip()
                    if s != "" and t != "":
                        src_cnt.update(s.split())
                        trg_cnt.update(t.split())
            print("Done!")
        src_vocab = Vocab(src_cnt, src_max_size, src_min_freq, src_specials)
        trg_vocab = Vocab(trg_cnt, trg_max_size, trg_min_freq, trg_specials)
        return src_vocab, trg_vocab

    def __len__(self):
        return sum([stat["n_sents"] for _, stat in self.stat.items()])

    def __iter__(self):
        n_sents = 0
        lengths, corpora, sents = [], [], []
        for corpus, stat in self.stat.items():
            lengths += [(k, v) for k, v in stat["lengths"]]
            corpora.append(corpus)
            n_sents += stat["n_sents"]
        
        print("\x1b[1K\rIterate over corpora in Eager mode", flush=True)
        for name, corpus in self.corpora.items():
            with io.open(corpus["src_path"], "r", encoding="utf-8") as sf, \
                 io.open(corpus["trg_path"], "r", encoding="utf-8") as tf:
                for i, (s, t) in enumerate(zip(sf, tf)):
                    s = s.strip()
                    t = t.strip()
                    src_max_len = corpus.get("src_max_len", None)
                    trg_max_len = corpus.get("trg_max_len", None)
                    if s != "" and t != "" and _filter(s, t, src_max_len, trg_max_len):
                        sents.append((s, t, corpus))
        
        id = [i for i in range(n_sents)]
        if self.train:
            shuffle(id)

        if self.n_tokens is not None:
            for batch_id in dynamic_batch(self.n_tokens, id, lengths):
                batch = [sents[i] for i in batch_id]
                yield batch
        else:
            for batch_id in standard_batch(self.batch_size, id, lengths):
                batch = [sents[i] for i in batch_id]
                yield batch


class LazyLoader(object):
    def __init__(self, corpora):
        self.corpora = corpora
        self.train_stat = {k: _statistic(v, get_length=True) \
            for k, v in corpora.items()}
        
    def _load_bilingual_examples(corpora):
        examples, specials = [], {}
        for name, corpus in corpora:
            print("Loading corpus {}".format(corpus))
            with io.open(corpus["src_path"], "r", encoding="utf-8") as f:
                print("?")

    def _load_multilingual_examples(corpora):
        specials = {}