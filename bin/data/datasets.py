from torchtext.data import Dataset, Example, interleave_keys

import io

def _create_filter_fn(src_max_len=None, trg_max_len=None):
    """
    Create filter_pred function for filtering out sentences \
        whose length are greater than {max_len}. 
    
    Returns a callable max_len specified filter function
    """
    return lambda ex: (src_max_len is None or (len(ex.src) <= src_max_len)) \
        and (trg_max_len is None or (len(ex.trg) <= trg_max_len))

class TranslationDataset(Dataset):
    @staticmethod
    def sort_key(ex):
        return interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, src_field, trg_field, src_path, trg_path, \
        src_max_len=None, trg_max_len=None, **kwargs):
        """
        Arguments:
            src_field (Field): source field
            trg_field (Field): target field
            train_path (str): path to training data
            valid_path ([, str]): path to valid data (if exists)
            ===================================================================
            **kwargs: Keyword params which are passed to Dataset instance
        """
        fields = [("src", src_field), ("trg", trg_field)]
        examples = []
        
        print("Loading data from src: `{}` and trg: `{}`".\
            format(src_path, trg_path))

        with io.open(src_path, 'r', encoding="utf-8") as src_file, \
             io.open(trg_path, 'r', encoding="utf-8") as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                examples.append(Example.fromlist([src_line, trg_line], fields))
        
        filter_fn = _create_filter_fn(src_max_len=src_max_len, \
            trg_max_len=trg_max_len)
      
        super(TranslationDataset, self).__init__(examples, fields, \
            filter_pred=filter_fn, **kwargs)

    def __len__(self):
        return len(self.examples)
