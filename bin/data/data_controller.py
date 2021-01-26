from torchtext.data import Field, BucketIterator, Example
from .iterators import Iterator, dynbatch, exbatch
    
import os, io, torch

from .datasets import TranslationDataset

datasets = {"translation": TranslationDataset}

class DataController(object):
    def __init__(self, dataset="translation", \
                       train_ds_kwargs={}, valid_ds_kwargs={}, \
                       train_batch_sz=32, valid_batch_sz=32, \
                       train_n_tokens=None, valid_n_tokens=None, \
                       src_vocab_max_size=None, trg_vocab_max_size=None, \
                       src_vocab_min_freq=1, trg_vocab_min_freq=1, \
                       save_dir=None, train=True, \
                       device="cpu", **kwargs):

        self.save_dir = save_dir
        self.device = device
        # Initialize source and target fields
        self.src_field = Field()
        self.trg_field = Field(init_token="<sos>", eos_token="<eos>")
        
        # Define dataset type
        if dataset not in datasets:
            raise ValueError("Dataset type {} did not exist in our system".\
                format(dataset))
        self.dataset = datasets[dataset]
        if train:
            # Build train dataset
            try:
                train_ds = self.dataset(self.src_field, self.trg_field, \
                    **train_ds_kwargs)
                # Build train iterator
                self.train_iter = self.create_iterator(train_ds, \
                    train=True, batch_size=train_batch_sz, \
                    n_tokens=train_n_tokens, device=device)
            except:
                raise OSError("Training data not found.")
            # Build valid dataset
            try:
                valid_ds = self.dataset(self.src_field, self.trg_field, \
                    **valid_ds_kwargs)
                # Build valid iterator      
                self.valid_iter = self.create_iterator(valid_ds, \
                    train=False, batch_size=valid_batch_sz, \
                    n_tokens=valid_n_tokens, device=device)
            except:
                print("[WARNING] No validation dataset specified.")
            
        if not self.load_vocab():
            if train:
                print("Building vocab objects from "
                    "training data instead...", end=' ')
                # Build vocab from training data
                self.src_field.build_vocab(train_ds, \
                    min_freq=src_vocab_min_freq, max_size=src_vocab_max_size)
                self.trg_field.build_vocab(train_ds, \
                    min_freq=trg_vocab_min_freq, max_size=trg_vocab_max_size)
                print("Done !")
                # Assign vocab attributes from both fields
                self.src_vocab = self.src_field.vocab
                self.trg_vocab = self.trg_field.vocab
                # Save vocab to file
                self.save_vocab()  
            else:
                raise Exception("Inference terminated due to vocab not found!")
        else:
            # Load vocab objects into respected fiels
            self.src_field.vocab = self.src_vocab
            self.trg_field.vocab = self.trg_vocab
    
    def create_infer_iter(self, src_sents, batch_size=32, n_tokens=None):
        """
        Create iterator for inference
        """
        examples = []
        for sent in src_sents:
            examples.append(Example.fromlist([sent], [("src", self.src_field)]))
        if n_tokens is not None:
            batches = dynbatch(n_tokens, examples, {"src": self.src_field})
        else:
            batches = exbatch(batch_size, examples)
        
        for batch in batches:
            yield self.src_field.process([getattr(x, "src") for x in batch], \
                device=self.device)

    def create_iterator(self, ds, batch_size=32, n_tokens=None, train=True, \
        device="cpu"):
        return Iterator(ds, batch_size, n_tokens, train, device)
    
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
    
    def save_vocab(self):
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

        