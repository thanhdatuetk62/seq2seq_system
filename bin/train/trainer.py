import time
import torch
from torch import nn

from .optim import find_optimizer, find_scheduler
from ..metrics import find_eval_metric, find_loss_metric
from ..data import EagerLoader, LazyLoader

from torch.cuda.amp import GradScaler, autocast

class Trainer(nn.Module):
    def __init__(self, portal, model, lazy=False,
                 train_loader={}, valid_loader=None, 
                 optimizer="adam", optimizer_kwargs={},
                 scheduler="noam", scheduler_kwargs={},
                 loss_metric="xent", loss_kwargs={}, accum_steps=1,
                 n_epochs=50, n_steps=None, report_steps=200,
                 valid_epochs=1, eval_epochs=1, eval_metric=None, strategy=None, 
                 valid_steps=1000, eval_steps=1000, use_float32=False,
                 save_checkpoint_steps=5000, save_checkpoint_epochs=1):
        
        super().__init__()
        # Initialize iteration loop (both epoch and sampling method!)
        self.epoch = -1
        self.step = -1

        # Link model to the trainer
        self.model = model

        # Link controller to the this module for saving checkpoints
        self.portal = portal
        self.device = portal.device

        # Specified number of batchs to accumulate for update weights.
        self.accum_steps = accum_steps

        # Specified number of epochs/steps to run
        self.n_epochs = n_epochs
        self.n_steps = n_steps

        # Specified report interval (counted by steps)
        self.report_steps = report_steps

        # Specified evaluate interval (counted by epochs/steps) on validation set
        self.valid_epochs = valid_epochs
        self.eval_epochs = eval_epochs
        self.valid_steps = valid_steps
        self.eval_steps = eval_steps

        # Determine how often saving checkpoint
        self.save_checkpoint_epochs = save_checkpoint_epochs
        self.save_checkpoint_steps = save_checkpoint_steps

        # [Unstable] Use AMP PF32
        self.use_float32 = use_float32

        # Define data loading method (i.e Feed batch of samples into model)
        use_sampling = (n_steps is not None)
        loader_cls = (LazyLoader if lazy else EagerLoader)
        self.train_loader = loader_cls(fields=portal.fields, \
            sampling=use_sampling, train=True, device=self.device, **train_loader)
        self.valid_loader = loader_cls(fields=portal.fields, \
            sampling=use_sampling, train=False, device=self.device, **valid_loader)

        # Define optimizer/scheduler
        optimizer = find_optimizer(optimizer)(model.parameters(), \
            **optimizer_kwargs)
        # Create learning rate scheduler
        if scheduler is not None:
            self.optimizer = find_scheduler(scheduler)(optimizer, \
                **scheduler_kwargs)
        else:
            self.optimizer = optimizer
        
        # Create loss function for training and validation
        self.loss_metric = find_loss_metric(loss_metric)(
            ignore_index=self.portal.trg_vocab.stoi["<pad>"],
            tgt_vocab_size=len(self.portal.trg_vocab), **loss_kwargs)

        # Prepare for evaluation
        if (eval_metric is not None) and (strategy is not None):
            self.eval_metric = find_eval_metric(eval_metric)
            self.strategy = strategy
    
    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "epoch": self.epoch,
            "step": self.step, 
            "optimizer": self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def run(self):
        if self.n_steps is not None:
            print("Training using sampling method")
            self.run_steps()
        else:
            print("Training using epochs method")
            self.run_epochs()

    def run_steps(self):
        # Turn on training mode
        if self.step < 0:
            self.model.init_params()
        
        self.model.train()
        print("Start training ...")
        
        report_time = time.perf_counter()
        start_time = time.perf_counter()
        total_loss = 0.0
        accum_loss = 0.0

        for batch, pos in self.train_loader:
            self.step += 1
            if self.step == self.n_steps:
                break

            # Add loss value to the total loss
            loss = self.train_step(batch.src, batch.trg)
            total_loss += loss.item()
            accum_loss += loss
            
            if (self.step + 1) % self.accum_steps == 0:
                accum_loss = accum_loss / self.accum_steps
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                accum_loss = 0.0

            if (self.step + 1) % self.report_steps == 0:
                # prepare stats
                speed = (time.perf_counter() - report_time) / self.report_steps
                avg_loss = total_loss / self.report_steps
                print("STEP {} - TRAIN_LOSS = {:.4f} - SPEED = {:.2f} sec/batch".format(self.step+1, avg_loss, speed))
                total_loss = 0.0
                report_time = time.perf_counter()
        
            if (self.step + 1) % self.save_checkpoint_steps == 0:
                # Save checkpoint to file
                self.portal.save_checkpoint(self.state_dict(), ckpt=self.step)

            if len(self.valid_loader) == 0:
                # No validation dataset, ignore validate and evaluate step
                continue

            if (self.step + 1) % self.valid_steps == 0:
                # Validate on validation dataset
                start_time = time.perf_counter()
                valid_loss = self.validate()
                time_used = time.perf_counter() - start_time
                print("STEP {} - VALID_LOSS = {:.4f} - TOTAL_TIME = {:.2f}".format(self.step + 1, valid_loss, time_used))

            if hasattr(self, "eval_metric") and (self.step + 1) % self.eval_steps == 0:
                # Evaluate on validation dataset
                start_time = time.perf_counter()
                eval_score = self.evaluate()
                time_used = time.perf_counter() - start_time
                print("STEP {} - EVAL_SCORE = {:.4f} - TOTAL_TIME = {:.2f}".format(self.step + 1, eval_score, time_used))

    def run_epochs(self):
        # Turn on training mode
        if self.epoch < 0:
            self.model.init_params()
        
        self.model.train()
        print("Start training ...")
        
        while self.epoch + 1 < self.n_epochs:
            self.epoch += 1
            report_time = time.perf_counter()
            start_time = time.perf_counter()
            total_loss = 0.0
            accum_loss = 0.0
            
            for step, (batch, pos) in enumerate(self.train_loader):
                # Add loss value to the total loss
                loss = self.train_step(batch.src, batch.trg)
                total_loss += loss.item()
                accum_loss += loss
                
                if (self.step + 1) % self.accum_steps == 0:
                    accum_loss = accum_loss / self.accum_steps
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                    accum_loss = 0.0

                if (step + 1) % self.report_steps == 0:
                    # prepare stats
                    speed = (time.perf_counter() - report_time) / self.report_steps
                    avg_loss = total_loss / self.report_steps
                    print("EPOCH {} - STEP {} - TRAIN_LOSS = {:.4f} - SPEED = {:.2f} sec/batch".format(self.epoch, step+1, avg_loss, speed))
                    total_loss = 0.0
                    report_time = time.perf_counter()
            
            if (self.epoch + 1) % self.save_checkpoint_epochs == 0:
                # Save checkpoint to file
                self.portal.save_checkpoint(self.state_dict(), ckpt=self.epoch)

            if len(self.valid_loader) == 0:
                # No validation dataset, ignore validate and evaluate step
                continue

            if (self.epoch + 1) % self.valid_epochs == 0:
                # Validate on validation dataset
                start_time = time.perf_counter()
                valid_loss = self.validate()
                time_used = time.perf_counter() - start_time
                print("EPOCH {} - VALID_LOSS = {:.4f} - TOTAL_TIME = {:.2f}".format(self.epoch, valid_loss, time_used))

            if hasattr(self, "eval_metric") and (self.epoch + 1) % self.eval_epochs == 0:
                # Evaluate on validation dataset
                start_time = time.perf_counter()
                eval_score = self.evaluate()
                time_used = time.perf_counter() - start_time
                print("EPOCH {} - EVAL_SCORE = {:.4f} - TOTAL_TIME = {:.2f}".format(self.epoch, eval_score, time_used))
    
    def train_step(self, src, trg):
        """
        Compute loss for each train step and update params
        Arguments:
            src: (Tensor [S x N]) - Input to Encoder
            trg: (Tensor [T x N]) - Input to Decoder
        """
        self.train()
        with autocast(enabled=self.use_float32):
            out = self.model(src, trg[:-1])

            # Flatten tensors for computing loss
            preds = out.view(-1, len(self.portal.trg_vocab))
            ys = trg[1:].contiguous().view(-1)

            # Feed inputs into loss metric
            loss = self.loss_metric(preds, ys)
            return loss

    @torch.no_grad()
    def validate(self):
        print("Running Validation ...")
        self.eval()
        
        loss, m = 0.0, 0
        for batch, pos in self.valid_loader:
            # loss = self.model.loss_step(src, trg, self.loss_metric)
            out = self.model(batch.src, batch.trg[:-1])

            # Flatten tensors for computing loss
            preds = out.view(-1, len(self.portal.trg_vocab))
            ys = batch.trg[1:].contiguous().view(-1)
            
            # Feed inputs into loss metric
            loss += self.loss_metric(preds, ys).item()
            m += 1
        return loss / m

    @torch.no_grad()
    def evaluate(self):
        print("Running evaluation ...")
        self.eval()

        ignores = {self.portal.trg_vocab.stoi["<eos>"],
                   self.portal.trg_vocab.stoi["<pad>"]}

        candidate_corpus, references_corpus = [], []

        for batch, pos in self.valid_loader:
            # Generate targe tokens
            sos_tokens = batch.trg[0]
            c = [self.portal.convert_to_str(sent) 
                 for sent in self.strategy(batch.src, sos_tokens)]
            r = [[[self.portal.trg_vocab.itos[j] for j in sent 
                   if j.item() not in ignores]] for sent in batch.trg[1:].t()]
            candidate_corpus += c
            references_corpus += r
        
        return self.eval_metric(candidate_corpus, references_corpus) * 100.0
