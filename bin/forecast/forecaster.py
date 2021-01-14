import io
import time

from torch import nn
from ..utils import print_progress

class Forecaster(nn.Module):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.model = controller.model
        self.data = controller.data
        
        # Define decode strategy
        self.sos_token = self.data.trg_vocab.stoi["<sos>"]
        self.eos_token = self.data.trg_vocab.stoi["<eos>"]
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
    
    def infer_from_file(self, src_path, save_path, batch_size):    
        """
        Infer from source file (required pre-tokenized in this file) and save 
        predicted sentences to target file .
        Args:
            src_path: (str) - Path to source file
            save_path: (str) - Path to save target file
            batch_size: (int) - Number of examples in a batch
        """
        self.eval()
        with io.open(src_path, 'r', encoding="utf-8") as fi, \
             io.open(save_path, 'w', encoding="utf-8") as fo:
            # Load src sents from file
            src_sents = [sent.strip().split() for sent in fi]

            # Init some local vars
            total_n_sents = len(src_sents)
            trg_sents = []
            n_sents = 0
            start_time = time.perf_counter()
            print_progress(n_sents, total_n_sents, max_len=40,
                            prefix="INFER", suffix="DONE", time_used=0)

            # Create batch iterator
            batch_iter = self.data.create_infer_iter(src_sents, batch_size)
            for src in batch_iter:
                # Append generated result to final results
                trg_sents += [' '.join(self.data.convert_to_str(tokens))
                                for tokens in self(src)]
                # Update progress
                n_sents += src.size(0)
                time_used = time.perf_counter() - start_time
                print_progress(n_sents, total_n_sents, max_len=40,
                                prefix="INFER", suffix="DONE",
                                time_used=time_used)
            
            # Write final results to file
            for line in trg_sents:
                print(line.strip(), file=fo)
            print("Successful infer to file {}".format(save_path))
    
       
    