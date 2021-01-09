import time
import torch
import os
import io
import re

from .models import find_model
from .train import Trainer
from .data import DataController
from .forecast import Forecaster
from .utils import print_progress


class Controller(object):
    def __init__(self, mode, model, model_kwargs={}, data_kwargs={},
                 device="cpu", save_dir='./run', **kwargs):

        # Make directory for saving checkpoints and vocab
        self.save_dir = save_dir
        if not isinstance(save_dir, str) or \
                not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        self.data = DataController(save_dir=self.save_dir,
                                   train=(mode == "train"),
                                   device=device, **data_kwargs)
        self.model = find_model(model)(data=self.data, device=device, \
            **model_kwargs)
        self.device = device
    
    def train(self, train_config, ckpt=None):
        trainer = Trainer(self, **train_config)
        trainer.to(self.device)
        # Load checkpoint
        ckpt_file = self.find_checkpoint(ckpt=ckpt)
        if ckpt_file is not None:
            print("Loading checkpoint file {} ...".format(ckpt_file), end=' ')
            state_dict = torch.load(ckpt_file)
            trainer.load_state_dict(state_dict)
            print("Done!")
        # Run train
        trainer.run()

    def infer(self, forecaster, src_sents, batch_size=32):
        trg_sents = []
        # Init some local vars for reporting
        total_n_sents = len(src_sents)
        n_sents = 0
        start_time = time.perf_counter()
        print_progress(n_sents, total_n_sents, max_len=40,
                        prefix="INFER", suffix="DONE", time_used=0)

        # Create batch iterator
        batch_iter = self.data.create_infer_iter(src_sents, batch_size,
                                                    device=self.device)
        for src in batch_iter:
            # Append generated result to final results
            trg_sents += [' '.join(self.data.convert_to_str(tokens))
                            for tokens in forecaster.run(src)]
            # Update progress
            n_sents += src.size(0)
            time_used = time.perf_counter() - start_time
            print_progress(n_sents, total_n_sents, max_len=40,
                            prefix="INFER", suffix="DONE",
                            time_used=time_used)
        return trg_sents        

    def infer_from_file(self, src_path, save_path='output.txt', ckpt=None, \
            batch_size=32, **kwargs):
        forecaster = Forecaster(self, **kwargs)
        forecaster.to(self.device)
        # Load checkpoint
        ckpt_file = self.find_checkpoint(ckpt=ckpt)
        if ckpt_file is not None:
            print("Loading checkpoint file {} ...".format(ckpt_file), end=' ')
            state_dict = torch.load(ckpt_file)
            forecaster.load_state_dict(state_dict)
            print("Done!")

        with io.open(src_path, 'r', encoding="utf-8") as fi, \
                io.open(save_path, 'w', encoding="utf-8") as fo:
            src_sents = [sent.strip().split() for sent in fi]
            trg_sents = self.infer(forecaster, src_sents, batch_size)
            # Write final results to file
            for line in trg_sents:
                print(line.strip(), file=fo)
            print("Successful infer to file {}".format(save_path))

    def find_checkpoint(self, ckpt=None):
        pattern = re.compile(r"checkpoint_(\d+).pt")
        last_ckpt = -1
        for f in os.listdir(self.save_dir):
            if not os.path.isfile(os.path.join(self.save_dir, f)):
                continue
            match = pattern.search(f)
            if match is not None:
                ckpt_id = int(match.groups()[0])
                if ckpt_id == ckpt:
                    return os.path.join(self.save_dir, f)
                last_ckpt = max(last_ckpt, ckpt_id)
        if last_ckpt < 0:
            return None
        return os.path.join(self.save_dir,
                            "checkpoint_{}.pt".format(last_ckpt))

    def save_to_file(self, state_dict, ckpt=0):
        save_file = os.path.join(self.save_dir,
                                 'checkpoint_{}.pt'.format(ckpt))
        torch.save(state_dict, save_file)
        print("Saved checkpoint {} to file {}".format(ckpt, save_file))
