import io
import time
import torch
from torch import nn
from ..models import find_model
from ..data import DataController
from ..utils import print_progress
from .strategies import find_forecast_strategy

class Forecaster(nn.Module):
    def __init__(self, controller, data_kwargs, model, model_kwargs, \
        strategy=None, strategy_kwargs={}, sos_token="<sos>", device="cpu", \
        batch_size=32, n_tokens=None):
        super().__init__()
        self.controller = controller
        self.device = device
        self.batch_size = batch_size
        self.n_tokens = n_tokens
        self.data = DataController(save_dir=controller.save_dir, \
            device=self.device, **data_kwargs)
        if not self.data.load_vocab():
            raise RuntimeError("Cannot load vocab. Inference terminated!")

        # Model must be built after loading vocabulary, otherwise raise Error
        self.model = find_model(model)(data=self.data, device=device, \
            **model_kwargs)

        self.strategy = find_forecast_strategy(strategy)(\
            data=self.data, model=self.model, device=self.device, \
            sos_token=sos_token, **strategy_kwargs)
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
    
    @torch.no_grad()
    def infer_from_file(self, src_path, save_path):    
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
            src_sents = [sent.strip() for sent in fi]

            # Init some local vars
            total_n_sents = len(src_sents)
            trg_sents = []
            n_sents = 0
            start_time = time.perf_counter()
            print_progress(n_sents, total_n_sents, max_len=40,
                            prefix="INFER", suffix="DONE", time_used=0)

            # Create batch iterator
            batch_iter = self.data.create_infer_iter(src_sents, self.batch_size, \
                n_tokens=self.n_tokens)
            for src in batch_iter:
                trg = [' '.join(self.data.convert_to_str(tokens))
                                for tokens in self.strategy(src)]
                # Append generated result to final results
                trg_sents += trg
                # Update progress
                n_sents += src.size(1)
                time_used = time.perf_counter() - start_time
                print_progress(n_sents, total_n_sents, max_len=40,
                                prefix="INFER", suffix="DONE",
                                time_used=time_used)
            
            # Write final results to file
            for line in trg_sents:
                print(line.strip(), file=fo)
            print("Successful infer to file {}".format(save_path))
    
       
    