import io

def _filter(src_len, trg_len, src_max_len=None, trg_max_len=None):
    """
    Create filter_pred function for filtering out sentences \
        whose length are greater than {max_len}. 
    
    Returns a callable max_len specified filter function
    """
    return (src_max_len is None or (src_len <= src_max_len)) \
        and (trg_max_len is None or (trg_len <= trg_max_len))

def load_monotext(path, **kwargs):
    with io.open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line != "":
                yield line, i

def load_bitext(src_lang, trg_lang, path, src_max_len=None, trg_max_len=None, **kwargs):
    with io.open(path + '.' + src_lang, "r", encoding="utf-8") as sf, \
         io.open(path + '.' + trg_lang, "r", encoding="utf-8") as tf:
        for i, (s, t) in enumerate(zip(sf, tf), 1):
            s = s.strip()
            t = t.strip()
            if s != "" and t != "" and \
                _filter(len(s.split()), len(t.split()), src_max_len, trg_max_len):
                yield s, t, i

def count_lines(path):
    n_lines = 0
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
    return n_lines