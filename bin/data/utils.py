import io
import random
from torchtext.data import interleave_keys
from torchtext.vocab import Vocab
from collections import Counter

from .io_handlers import load_bitext, load_monotext

def statistic(name, corpus, bitext=True):
    """Get stats from parallel corpus"""
    print("Gathering corpora information from corpus {} ...".format(name), end=" ",)
    n_sents = 0
    lengths = []
    line_id = []
    if bitext:
        for s, t, i in load_bitext(**corpus):
            s, t = s.split(), t.split()
            lengths.append((len(s), len(t)))
            line_id.append(i)
            n_sents += 1
    else:
        for s, i in load_monotext(**corpus):
            s = s.split()
            lengths.append((len(s), 0))
            line_id.append(i)
            n_sents += 1
    print("Done!")
    return {
        "n_sents": n_sents, \
        "lengths": lengths, \
        "line_id": line_id, \
        "weight": corpus.get("weight", 1.0)
    }

def build_vocab(src_paths, trg_paths, src_min_freq=1, trg_min_freq=1, \
    src_max_size=None, trg_max_size=None, src_specials=("<unk>", "<pad>"), \
    trg_specials=("<unk>", "<pad>")):
    """
    Build source vocab and target vocab given src_paths and trg_paths with some
    additional configurations.
    """
    print("Building vocab ...")
    src_cnt, trg_cnt = Counter(), Counter()
    for src_path in src_paths:
        print("Tracing path {} ...".format(src_path))
        with io.open(src_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    src_cnt.update(line.split())
    for trg_path in trg_paths:
        print("Tracing path {} ...".format(trg_path))
        with io.open(trg_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line != "":
                    trg_cnt.update(line.split())
    print("Done!")
    src_vocab = Vocab(src_cnt, src_max_size, src_min_freq, src_specials)
    trg_vocab = Vocab(trg_cnt, trg_max_size, trg_min_freq, trg_specials)
    return src_vocab, trg_vocab

def dynamic_batch(n_tokens, id, lengths, probs=None, sort=False, sampling=False):
    batch, cnt, batches = [], 0, []
    max_src, max_trg = 0, 0
    if sampling:
        chunk = 10000
        assert probs is not None
        probs_id = [probs[i] for i in id]
        while True:
            q = random.choices(id, probs_id, k=chunk)
            for x in q:
                src_len, trg_len = lengths[x]
                sz = max(max_src, max_trg, src_len, trg_len) * (cnt + 1)
                if sz > n_tokens:
                    yield batch
                    batch, cnt = [], 0
                    max_src, max_trg = 0, 0
                cnt += 1
                max_src = max(max_src, src_len)
                max_trg = max(max_trg, trg_len)
                batch.append(x)
    elif sort:
        chunk = 10000
        for i in range(0, len(id), chunk):
            p_id = sorted(id[i:i+chunk], key=lambda x: interleave_keys(*lengths[x]))
            for j in p_id:
                src_len, trg_len = lengths[j]
                sz = max(max_src, max_trg, src_len, trg_len) * (cnt + 1)
                if sz > n_tokens:
                    batches.append(batch)
                    batch, cnt = [], 0
                    max_src, max_trg = 0, 0
                cnt += 1
                max_src = max(max_src, src_len)
                max_trg = max(max_trg, trg_len)
                batch.append(j)
            if len(batch) > 0:
                batches.append(batch)
                batch, cnt = [], 0
                max_src, max_trg = 0, 0
        for b in batches:
            yield b
    else:
        for i in id:
            src_len, trg_len = lengths[i]
            sz = max(max_src, max_trg, src_len, trg_len) * (cnt + 1)
            if sz > n_tokens:
                batches.append(batch)
                batch, cnt = [], 0
                max_src, max_trg = 0, 0
            cnt += 1
            max_src = max(max_src, src_len)
            max_trg = max(max_trg, trg_len)
            batch.append(i)
        if len(batch) > 0:
            batches.append(batch)
            # yield batch
            batch, cnt = [], 0
            max_src, max_trg = 0, 0
        for b in batches:
            yield b

def standard_batch(batch_size, id, lengths, probs=None, sort=False, sampling=False):
    batches = []
    if sampling:
        assert probs is not None
        probs_id = [probs[i] for i in id]
        while True:
            yield random.choices(id, probs_id, k=batch_size)      
    elif sort:
        chunk = batch_size * 100
        for i in range(0, len(id), chunk):
            p_id = sorted(id[i:i+chunk], key=lambda x: interleave_keys(*lengths[x]))
            for j in range(0, len(p_id), batch_size):
                batch = p_id[j:j+batch_size]
                # assert len(batch) > 0
                batches.append(batch)
        for b in batches:
            yield b
    else:
        for i in range(0, len(id), batch_size):
            batches.append(id[i:i+batch_size])
        for b in batches:
            yield b