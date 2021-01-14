import time
import torch
from torch import nn
from torch.optim import optimizer

from .optim import find_optimizer, find_scheduler
from ..metrics import find_eval_metric, find_loss_metric
from ..forecast import find_forecast_strategy
from ..utils import print_progress

from torch.cuda.amp import GradScaler, autocast

class Trainer(nn.Module):
    def __init__(self, controller, optimizer="adam", optimizer_kwargs={},
                 scheduler="noam", scheduler_kwargs={},
                 loss_metric="xent", loss_kwargs={}, 
                 eval_metric=None, strategy=None, strategy_kwargs={},
                 n_epochs=50, report_steps=200, valid_epochs=1, eval_epochs=1, 
                 save_checkpoint_epochs=1):
        super().__init__()
        self.controller = controller
        self.data = controller.data
        self.model = controller.model
        self.epoch = -1
        self.n_epochs = n_epochs
        self.report_steps = report_steps
        self.valid_epochs = valid_epochs
        self.eval_epochs = eval_epochs
        self.save_checkpoint_epochs = save_checkpoint_epochs

        # Define optimizer/scheduler
        optimizer = find_optimizer(optimizer)(self.parameters(), \
            **optimizer_kwargs)
        # Create learning rate scheduler
        if scheduler is not None:
            self.optimizer = find_scheduler(scheduler)(optimizer, \
                **scheduler_kwargs)
        else:
            self.optimizer = optimizer
        
        # Create loss function for training and validation (default: xent)
        self.loss_metric = find_loss_metric(loss_metric)(
            ignore_index=self.data.trg_vocab.stoi["<pad>"],
            tgt_vocab_size=len(self.data.trg_vocab), **loss_kwargs)

        # Prepare for evaluation (Inference)
        if eval_metric is not None:
            self.eval_metric = find_eval_metric(eval_metric)
            self.strategy = find_forecast_strategy(strategy)(\
                controller=controller, **strategy_kwargs)
    
    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "epoch": self.epoch,
            "optimizer": self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.epoch = state_dict["epoch"]
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def run(self):
        # Turn on training mode
        self.model.train()

        if self.epoch < 0:
            self.model.init_params()

        print("Start training ...")
        
        scaler = GradScaler()
        total_n_sents = len(self.data.train_iter.dataset)
        while self.epoch + 1 < self.n_epochs:
            self.epoch += 1
            report_time = time.perf_counter()
            start_time = time.perf_counter()
            n_sents = 0
            total_loss = 0.0
            # Init progress bar
            print_progress(n_sents, total_n_sents, max_len=40,
                           prefix="EPOCH {}".format(self.epoch),
                           suffix="DONE", time_used=0)

            self.data.train_iter.init_epoch()
            for step, batch in enumerate(self.data.train_iter):
                # Perform one training step
                with autocast():
                    self.train()
                    loss = self.model.train_step(batch.src, batch.trg, \
                        self.loss_metric)
                # Update params
                self.optimizer.zero_grad(set_to_none=True)
                # loss.backward()
                # self.optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # Add loss value to the total loss
                total_loss +=loss.item()

                # Update progress bar
                n_sents += batch.batch_size
                print_progress(n_sents, total_n_sents, max_len=40,
                               prefix="EPOCH {}".format(self.epoch), 
                               suffix="DONE",
                               time_used=time.perf_counter()-start_time)

                if (step + 1) % self.report_steps == 0:
                    # prepare stats
                    speed = (time.perf_counter() - report_time) / \
                        self.report_steps
                    avg_loss = total_loss / self.report_steps
                    print("\x1b[1K\rEPOCH {} - STEP {} - TRAIN_LOSS = {:.4f} -\
SPEED = {:.2f} sec/batch".format(self.epoch, step+1, avg_loss, speed))
                    total_loss = 0.0
                    report_time = time.perf_counter()

            if (self.epoch + 1) % self.save_checkpoint_epochs == 0:
                # Save checkpoint to file
                self.controller.save_to_file(self.state_dict(), ckpt=self.epoch)

            if self.data.valid_iter is None:
                # No validation dataset, ignore validate and evaluate step
                continue

            if (self.epoch + 1) % self.valid_epochs == 0:
                # Validate on validation dataset
                start_time = time.perf_counter()
                valid_loss = self.validate()
                time_used = time.perf_counter() - start_time
                print("EPOCH {} - VALID_LOSS = {:.4f} - TOTAL_TIME = {:.2f}"
                      .format(self.epoch, valid_loss, time_used))

            if hasattr(self, "eval_metric") and \
                    (self.epoch + 1) % self.eval_epochs == 0:
                # Evaluate on validation dataset
                start_time = time.perf_counter()
                eval_score = self.evaluate()
                time_used = time.perf_counter() - start_time
                print("EPOCH {} - EVAL_SCORE = {:.4f} - TOTAL_TIME = {:.2f}"
                      .format(self.epoch, eval_score, time_used))
    
    def validate(self):
        print("Running Validation ...")
        self.eval()
        losses = [self.model.validate_step(batch.src, batch.trg, \
            self.loss_metric) for batch in self.data.valid_iter]
        return sum(losses) / len(losses)

    def evaluate(self):
        print("Running evaluation ...")
        self.eval()
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
            c = [self.data.convert_to_str(sent) 
                 for sent in self.strategy(batch.src)]
            r = [[[self.data.trg_vocab.itos[j] for j in sent 
                   if j.item() not in ignores]] for sent in batch.trg]
            candidate_corpus += c
            references_corpus += r

            # Update progress
            n_sents += batch.batch_size
            time_used = time.perf_counter() - start_time
            print_progress(n_sents, total_n_sents, max_len=40,
                           prefix="EVAL", suffix="DONE", time_used=time_used)
                        
        return self.eval_metric(candidate_corpus, references_corpus)
