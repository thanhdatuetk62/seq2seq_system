import time
import torch
import os
import io
import re

from .models import find_model
from .data import DataController
from .utils.misc import print_progress


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
        self.model = find_model(model)(controller=self, data=self.data,
                                       device=device, **model_kwargs)

        self.device = device

    def train(self, ckpt=None, n_epochs=40, report_steps=1, valid_epochs=1,
              eval_epochs=1, save_checkpoint_epochs=1, **kwargs):
        epoch = -1
        self.model.init_train(**kwargs)

        # Load checkpoint
        ckpt_file = self.find_checkpoint(ckpt=ckpt)
        if ckpt_file is not None:
            print("Loading checkpoint file {} ...".format(ckpt_file), end=' ')
            state_dict = torch.load(ckpt_file)
            epoch = state_dict["epoch"]
            self.model.load_state_dict(state_dict)
            print("Done!")

        # Turn on training mode
        self.model.train()

        if epoch < 0:
            self.model.init_params()

        print("Start training ...")
        # print(self.validate(), " - VALID LOSS")
        # return

        total_n_sents = len(self.data.train_iter.dataset)
        while epoch < n_epochs:
            epoch += 1
            report_time = time.perf_counter()
            start_time = time.perf_counter()
            n_sents = 0
            total_loss = 0.0
            # Init progress bar
            print_progress(n_sents, total_n_sents, max_len=40,
                           prefix="EPOCH {}".format(epoch),
                           suffix="DONE", time_used=0)

            self.data.train_iter.init_epoch()
            for step, batch in enumerate(self.data.train_iter):
                # Perform one training step
                total_loss += self.model.train_step(batch.src, batch.trg)
                n_sents += batch.batch_size
                # Update progress bar
                print_progress(n_sents, total_n_sents, max_len=40,
                               prefix="EPOCH {}".format(epoch), suffix="DONE",
                               time_used=time.perf_counter()-start_time)

                if (step + 1) % report_steps == 0:
                    # prepare stats
                    speed = (time.perf_counter() - report_time) / report_steps
                    avg_loss = total_loss / report_steps
                    print("\x1b[1K\rEPOCH {} - STEP {} - TRAIN_LOSS = {:.4f} -\
 SPEED = {:.2f} sec/batch".format(epoch, step+1, avg_loss, speed))
                    total_loss = 0.0
                    report_time = time.perf_counter()

            if (epoch + 1) % save_checkpoint_epochs == 0:
                # Save checkpoint to file
                state_dict = self.model.state_dict()
                # Embed epoch info
                state_dict["epoch"] = epoch
                self.save_to_file(state_dict, ckpt=epoch)

            if self.data.valid_iter is None:
                # No validation dataset, ignore validate and evaluate step
                continue

            if (epoch + 1) % valid_epochs == 0:
                # Validate on validation dataset
                start_time = time.perf_counter()
                valid_loss = self.validate()
                time_used = time.perf_counter() - start_time
                print("EPOCH {} - VALID_LOSS = {:.4f} - TOTAL_TIME = {:.2f}"
                      .format(epoch, valid_loss, time_used))

            if hasattr(self.model, "eval_metric") and \
                    (epoch + 1) % eval_epochs == 0:
                # Evaluate on validation dataset
                start_time = time.perf_counter()
                eval_score = self.evaluate()
                time_used = time.perf_counter() - start_time
                print("EPOCH {} - EVAL_SCORE = {:.4f} - TOTAL_TIME = {:.2f}"
                      .format(epoch, eval_score, time_used))

    def validate(self):
        print("Running validation ...")
        losses = [self.model.validate_step(batch.src, batch.trg)
                  for batch in self.data.valid_iter]
        return sum(losses) / len(losses)

    def evaluate(self):
        print("Running evaluation ...")

        total_n_sents = len(self.data.valid_iter.dataset)
        n_sents = 0
        start_time = time.perf_counter()
        print_progress(n_sents, total_n_sents, max_len=40,
                       prefix="EVAL", suffix="DONE", time_used=0)
        ignores = {self.data.trg_vocab.stoi["<sos>"], 
                   self.data.trg_vocab.stoi["<eos>"],
                   self.data.trg_vocab.stoi["<pad>"]}
        candidate_corpus, references_corpus = [], []
        
        for batch in self.data.valid_iter:
            # Generate targe tokens
            c = [self._convert_to_str(sent) 
                 for sent in self.model.forecast(batch.src)]
            r = [[[self.data.trg_vocab.itos[j] for j in sent if j.item() not in ignores]]
                 for sent in batch.trg]
            candidate_corpus += c
            references_corpus += r

            # Update progress
            n_sents += batch.batch_size
            time_used = time.perf_counter() - start_time
            print_progress(n_sents, total_n_sents, max_len=40,
                           prefix="EVAL", suffix="DONE", time_used=time_used)
        return self.model.eval_metric(candidate_corpus, references_corpus)

    def _convert_to_str(self, sent_id):
        eos_token = self.data.trg_vocab.stoi["<eos>"]
        eos = torch.nonzero(sent_id == eos_token).view(-1)
        t = eos[0] if len(eos) > 0 else len(sent_id)
        return [self.data.trg_vocab.itos[j] for j in sent_id[1: t]]

    def infer(self, src_path, save_path='output.txt', ckpt=None,
              batch_size=32, **kwargs):

        # Init infer forecast strategy
        self.model.init_infer(**kwargs)

        # Load checkpoint
        ckpt_file = self.find_checkpoint(ckpt=ckpt)
        if ckpt_file is not None:
            print("Loading checkpoint file {} ...".format(ckpt_file), end=' ')
            state_dict = torch.load(ckpt_file)
            self.model.load_state_dict(state_dict)
            print("Done!")

        src_sents, trg_sents = [], []

        with io.open(src_path, 'r', encoding="utf-8") as fi, \
                io.open(save_path, 'w', encoding="utf-8") as fo:
            src_sents = [sent.strip().split() for sent in fi]

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
                trg_sents += [' '.join(self._convert_to_str(tokens))
                              for tokens in self.model.forecast(src)]

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
