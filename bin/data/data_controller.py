from torchtext.data import Field, BucketIterator, Example
from .iterators import EagerLoader, LazyLoader, dynamic_batch, standard_batch
    
import warnings
import os, io, torch

class Batch(object):
    def __init__(self, data, fields, device="cpu"):
        self.batch_size = len(data)
        for name, field in fields:
            if field is not None:
                batch = [getattr(x, name) for x in data]
                setattr(self, name, field.process(batch, device=device))


class DataController(object):
    def __init__(self, dataset="bilingual", build_vocab_kwargs={}, \
        save_dir=None, device="cpu", **kwargs):
        
        self.dataset = dataset
        self.save_dir = save_dir
        self.device = device
        self.build_vocab_kwargs = build_vocab_kwargs

        self.src_field = Field()
        self.trg_field = Field(eos_token="<eos>")
    
    def create_iter(self, loader):
        fields = [("src", self.src_field), ("trg", self.trg_field)]
        examples = []
        for batch in loader:
            ex = []
            for (src, trg, corpus) in batch:
                init_token = ""
                if self.dataset == "bilingual":
                    init_token = "<sos>"
                if self.dataset == "multilingual":
                    trg_lang = corpus["trg_lang"]
                    init_token = "<{}>".format(trg_lang)
                trg = (init_token + " " + trg).strip()
                ex.append(Example.fromlist([src, trg], fields))
            examples += ex
            yield Batch(ex, fields, device=self.device)
    
    def create_infer_iter(self, src_sents, batch_size=32, n_tokens=None):
        """
        Create iterator for inference
        """
        fields = [("src", self.src_field)]
        examples = []
        for sent in src_sents:
            examples.append(Example.fromlist([sent], fields))
        id = [i for i in range(len(examples))]
        lengths = [(len(ex.src), 0) for ex in examples]
        if n_tokens is not None:
            batches_id = dynamic_batch(n_tokens, id, lengths, keep_order=True)
        else:
            batches_id = standard_batch(batch_size, id, lengths, keep_order=True)
        
        for batch_id in batches_id:
            batch = [examples[i] for i in batch_id]
            yield self.src_field.process([getattr(x, "src") for x in batch], \
                device=self.device)
    
    def convert_to_str(self, sent_id):
        eos_token = self.trg_vocab.stoi["<eos>"]
        eos = torch.nonzero(sent_id == eos_token).view(-1)
        t = eos[0] if len(eos) > 0 else len(sent_id)
        return [self.trg_vocab.itos[j] for j in sent_id[1: t]]

    def state_dict(self):
        return {
            "src_vocab": self.src_vocab,
            "trg_vocab": self.trg_vocab
        }
    
    def load_state_dict(self, state_dict):
        self.src_vocab = state_dict["src_vocab"]
        self.trg_vocab = state_dict["trg_vocab"]
        self.src_field.vocab = self.src_vocab
        self.trg_field.vocab = self.trg_vocab
    
    def save_vocab(self, src_vocab, trg_vocab):
        self.src_field.vocab = src_vocab
        self.trg_field.vocab = trg_vocab
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        if self.save_dir is None:
            print("No save dir specified, cannot save vocab.")
            return
        state_dict = self.state_dict()
        save_file = os.path.join(self.save_dir, 'vocab.pt')
        torch.save(state_dict, save_file)
        print("Saved vocab to {}".format(save_file))
    
    def load_vocab(self):
        if self.save_dir is None:
            print("No save dir specified, cannot load vocab.")
            return False
        try:
            save_file = os.path.join(self.save_dir, 'vocab.pt')
            state_dict = torch.load(save_file)
            self.load_state_dict(state_dict)
            print("Load vocab from file successfully.")
            return True
        except:
            print("Vocab objects not found in save_dir.")
            return False

        