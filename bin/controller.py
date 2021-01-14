import time
import torch
import os
import io
import re

from .models import find_model
from .train import Trainer
from .data import DataController
from .forecast import find_forecast_strategy


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
    
    def save_to_file(self, state_dict, ckpt=0):
        save_file = os.path.join(self.save_dir,
                                 'checkpoint_{}.pt'.format(ckpt))
        torch.save(state_dict, save_file)
        print("Saved checkpoint {} to file {}".format(ckpt, save_file))
    
    def train(self, train_config, ckpt=None):
        trainer = Trainer(self, **train_config)
        trainer.to(self.device)
        # Load checkpoint
        ckpt_file = self._find_checkpoint(ckpt=ckpt)
        if ckpt_file is not None:
            print("Loading checkpoint file {} ...".format(ckpt_file), end=' ')
            state_dict = torch.load(ckpt_file)
            trainer.load_state_dict(state_dict)
            print("Done!")
        # Run train
        trainer.run()

    def infer(self, src_path, save_path='output.txt', ckpt=None, \
            batch_size=32, strategy="beam_search", strategy_kwargs={}):
        """
        Frontend infer command. Please refer internal implementation of 
        Forecaster for more details.
        Args:
            ckpt: (int) - Index of checkpoint to load
            ...
        """
        forecaster = find_forecast_strategy(strategy)(controller=self, \
            **strategy_kwargs)
        forecaster.to(self.device)
        # Load checkpoint
        ckpt_file = self._find_checkpoint(ckpt=ckpt)
        if ckpt_file is not None:
            print("Loading checkpoint file {} ...".format(ckpt_file), end=' ')
            state_dict = torch.load(ckpt_file)
            forecaster.load_state_dict(state_dict)
            print("Done!")
        forecaster.infer_from_file(src_path, save_path, batch_size)

    def compile(self, ckpt=None, export_path="export.pt", \
        strategy="beam_search", strategy_kwargs={}, **kwargs):
        """
        Load and compile forecaster module to TorchScript and save it to file
        Args:
            ckpt: (int) - Index of checkpoint to load
            export_path: (str) - Path to save TorchScript
            ...
        """
        forecaster = find_forecast_strategy(strategy)(controller=self, \
            **strategy_kwargs)
        forecaster.eval()
        # Load checkpoint
        ckpt_file = self._find_checkpoint(ckpt=ckpt)
        if ckpt_file is not None:
            print("Loading checkpoint file {} ...".format(ckpt_file), end=' ')
            state_dict = torch.load(ckpt_file)
            forecaster.load_state_dict(state_dict)
            print("Done!")
        # TorchScript
        forecaster = torch.jit.script(forecaster)
        torch.jit.save(forecaster, export_path)
        print("Model compiled successfully")

    def _find_checkpoint(self, ckpt=None):
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

    
