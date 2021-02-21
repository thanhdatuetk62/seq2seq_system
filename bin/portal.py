from .strategies import find_forecast_strategy
from torchtext.data import Field
from .data import count_lines, build_vocab, EagerLoader
import time
import torch
import os
import io
import re

from .models import find_model
from .train import Trainer


class Portal(object):
    def __init__(self, device="cpu", save_dir='./run', keep_checkpoints=100, **options):
        # Make directory for saving checkpoints and vocab
        self.device = device
        self.save_dir = save_dir

        # Other useful options for each mode
        self.options = options

        # Create directory if not exist
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # Create fields for containing vocabulary and convert text to tensors.
        self.src_field = Field()
        self.trg_field = Field(eos_token="<eos>")

        # Keep track number of checkpoints in this save_dir
        self.keep_checkpoints = keep_checkpoints
    
    @property
    def fields(self):
        return [("src", self.src_field), ("trg", self.trg_field)]
    
    @property
    def SRC(self):
        return [("src", self.src_field)]
    
    @property
    def TRG(self):
        return [("trg", self.trg_field)]
    
    @property
    def src_vocab(self):
        return self.src_field.vocab
    
    @property
    def trg_vocab(self):
        return self.trg_field.vocab

    def save_checkpoint(self, state_dict, ckpt=0):
        save_file = os.path.join(self.save_dir, 'checkpoint_{}.pt'.format(ckpt))
        torch.save(state_dict, save_file)
        print("Saved checkpoint {} to file {}".format(ckpt, save_file))
        while True:
            saved_checkpoints = self._find_all_checkpoints()
            if len(saved_checkpoints) <= self.keep_checkpoints:
                break
            earliest = min(saved_checkpoints.keys())
            name = saved_checkpoints[earliest]
            os.remove(name)
            print("Removed checkpoint {} from save_dir".format(name))
    
    def save_vocab(self):
        save_file = os.path.join(self.save_dir, "vocab.pt")
        torch.save({
            "src_vocab": self.src_field.vocab, 
            "trg_vocab": self.trg_field.vocab}, save_file)
        print("Saved vocab to {}".format(save_file))

    def build_vocab(self):
        # Get config from options [external yml file]
        build_vocab_config = self.options["build_vocab_config"]
        src_vocab, trg_vocab = build_vocab(**build_vocab_config)

        # Load vocabs into res fields
        self.src_field.vocab = src_vocab
        self.trg_field.vocab = trg_vocab

        # Save it to sys file
        self.save_vocab()
    
    def share_vocab(self):
        print("Sharing source vocab and target vocab ...", end=" ")
        shared_vocab = self.src_field.vocab
        trg_vocab = self.trg_field.vocab
        # Merge src_vocab and trg_vocab
        shared_vocab.extend(trg_vocab)
        # Reassign vocab to each fields
        self.src_field.vocab = shared_vocab
        self.trg_field.vocab = shared_vocab
        print("Done! [Vocab size: {}]".format(len(shared_vocab)))
        return len(shared_vocab)

    def load_vocab(self, train=False):
        if self.save_dir is None:
            raise RuntimeError("No save dir specified, cannot load vocab.")
        try:
            save_file = os.path.join(self.save_dir, 'vocab.pt')
            state_dict = torch.load(save_file)
            # Load vocabs into res fields
            self.src_field.vocab = state_dict["src_vocab"]
            self.trg_field.vocab = state_dict["trg_vocab"]
            print("Load vocab from file successfully.")
        except:
            print("Vocabs not found in {}. Building from scratch ...".format(self.save_dir))
            if train:
                self.build_vocab()
            else:
                raise RuntimeError("Cannot find vocab file in {}".format(self.save_dir))

    def convert_to_str(self, sent_id):
        eos_token = self.trg_vocab.stoi["<eos>"]
        eos = torch.nonzero(sent_id == eos_token).view(-1)
        t = eos[0] if len(eos) > 0 else len(sent_id)
        return [self.trg_vocab.itos[j] for j in sent_id[1: t]]
    
    def train(self, ckpt=None):
        """Train command"""
        # Must load vocab before define training model
        self.load_vocab(train=True)

        # Define training model
        model = self.options["model"]
        model_config = self.options["model_config"]
        model = find_model(model)(portal=self, device=self.device, **model_config)

        # Define decode strategy
        strategy = self.options.get("strategy", None)
        if strategy is not None:
            strategy_config = self.options.get("strategy_config", {})
            strategy = find_forecast_strategy(strategy)(portal=self, \
                model=model, device=self.device, **strategy_config)
        
        # Define training paradigm
        train_config = self.options["train_config"]
        trainer = Trainer(self, model=model, strategy=strategy, **train_config)

        # Cast to the specified device type
        trainer.to(self.device)

        # Load checkpoint
        ckpt_file = self._select_checkpoint(ckpt=ckpt)
        if ckpt_file is not None:
            print("Loading checkpoint file {} ...".format(ckpt_file), end=' ')
            state_dict = torch.load(ckpt_file)
            trainer.load_state_dict(state_dict)
            print("Done!")
        
        # Run train
        trainer.run()

    def infer(self, src_path, save_path, ckpt=None, src_prefix="", \
        trg_prefix="<sos>", batch_size=64, n_tokens=None):
        """Inference command"""
        # Must load vocab before define model
        self.load_vocab(train=False)

        # Define training model
        model = self.options["model"]
        model_config = self.options.get("model_config", {})
        model = find_model(model)(portal=self, device=self.device, **model_config)

        # Define decode strategy
        strategy = self.options.get("strategy", None)
        if strategy is not None:
            strategy_config = self.options.get("strategy_config", {})
            strategy = find_forecast_strategy(strategy)(portal=self, \
                model=model, sos_token=trg_prefix, device=self.device, **strategy_config)
        
        # Cast to the specified device type
        strategy.to(self.device)

        # Load checkpoint
        ckpt_file = self._select_checkpoint(ckpt=ckpt)
        if ckpt_file is not None:
            print("Loading checkpoint file {} ...".format(ckpt_file), end=' ')
            state_dict = torch.load(ckpt_file)
            strategy.load_state_dict(state_dict)
            print("Done!")
        
        # Start infer from file
        print("Inferencing from file {} ...".format(src_path))

        corpora = {"EVAL": {"path": src_path, "src_prefix": src_prefix}}
        eval_loader = EagerLoader(self.SRC, batch_size, n_tokens, train=False, \
            device=self.device, bitext=False, **corpora)
        
        # Turn off training mode
        strategy.eval()

        # Write file concurrently
        with io.open(save_path, 'w', encoding="utf-8") as f:
            start_time = time.perf_counter()
            output = [''] * count_lines(src_path)
            for batch, pos in eval_loader:
                trg = [' '.join(self.convert_to_str(tokens))
                                for tokens in strategy(batch.src)]
                for sent, (id, name) in zip(trg, pos):
                    output[id] = sent
            for line in output:
                print(line, file=f)
            time_used = time.perf_counter() - start_time
            print("Done in {:4f}s. Saved output to {}".format(time_used, save_path))

    def compile(self, ckpt=None, export_path="export.pt", \
        strategy="beam_search", strategy_kwargs={}, **kwargs):
        """
        Load and compile forecaster module to TorchScript and save it to file
        Args:
            ckpt: (int) - Index of checkpoint to load
            export_path: (str) - Path to save TorchScript
            ...
        """
        raise NotImplementedError
        # forecaster = find_forecast_strategy(strategy)(controller=self, \
        #     **strategy_kwargs)
        # forecaster.eval()
        # # Load checkpoint
        # ckpt_file = self._select_checkpoint(ckpt=ckpt)
        # if ckpt_file is not None:
        #     print("Loading checkpoint file {} ...".format(ckpt_file), end=' ')
        #     state_dict = torch.load(ckpt_file)
        #     forecaster.load_state_dict(state_dict)
        #     print("Done!")
        # # TorchScript
        # forecaster = torch.jit.script(forecaster)
        # torch.jit.save(forecaster, export_path)
        # print("Model compiled successfully")

    def _select_checkpoint(self, ckpt=None):
        ckpts = self._find_all_checkpoints()
        if len(ckpts) == 0:
            return None
        key = ckpt if (ckpt is not None and ckpt in ckpts) else max(ckpts.keys())
        return ckpts[key]

    def _find_all_checkpoints(self):
        pattern = re.compile(r"checkpoint_(\d+).pt")
        ckpts = {}
        for f in os.listdir(self.save_dir):
            if not os.path.isfile(os.path.join(self.save_dir, f)):
                continue
            match = pattern.search(f)
            if match is not None:
                ckpt_id = int(match.groups()[0])
                ckpts[ckpt_id] = os.path.join(self.save_dir, f)
        return ckpts

    
