from typing import Tuple, List
from tqdm.autonotebook import tqdm, trange
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import AverageMeter, FoldScore, to_numpy, to_numpy

class BaseLearner:
    def __init__(
        self,
        cfg,
        model: nn.Module,
        optimizer: torch.optim,
        train_loader:torch.utils.data.DataLoader,
        valid_loader:torch.utils.data.DataLoader,
        loss_fn,
        metric_fn,
        fold: int=1,
        scheduler=None,
        logger=None,
        ):

        self.cfg = cfg
        self.fold = fold
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.logger = logger

        self.t_total = len(self.train_loader) // self.cfg.gradient_accumulation_steps * self.cfg.epochs
        
        # if scheduler:
        #     self.scheduler = scheduler(self.optimizer, num_warmup_steps=0.1, num_training_steps=self.t_total)
        # else:
        self.scheduler = None
        
        self.global_meter = AverageMeter()
                
        self.trn_fold_scores = []
        self.val_fold_scores = []

    def setting_info(self):
        self.debug("Num Examples: {}".format(len(self.train_loader.dataset)))
        self.debug("Gradient Accumulation Step: {}".format(self.cfg.gradient_accumulation_steps))
        self.debug("Total Optimization Steps: {}".format(self.t_total))
        
    def train(self):
        self.setting_info()

        self.model.to(self.cfg.device)

        self.model.train()
        self.model.zero_grad()
        epoch_iterator = trange(self.cfg.epochs, desc="Epoch")

        for epoch in epoch_iterator:
            self.info('epoch {}: \t Start training...'.format(epoch+1))

            self.train_preds, self.train_targets = [], []

            trn_loss, trn_metric = self.train_epoch(epoch)

            self.info('train loss:{:.4f}\ttrain score:{:.4f}'.format(trn_loss, trn_metric))

            # valの予測が欲しい場合はそれをリターンするのでここに戻した方がいいと思われる
            val_loss, val_metric = self.validate(epoch)

            self.info('val loss:{:.4f}\tval score:{:.4f}'.format(val_loss, val_metric))
            
            self.trn_fold_scores.append(FoldScore(self.fold, epoch+1, self.get_optimizer_lr(), trn_loss, trn_metric))
            self.val_fold_scores.append(FoldScore(self.fold, epoch+1, self.get_optimizer_lr(), val_loss, val_metric))

            self.on_epoch_end()

    def validate(self, epoch):
        self.info('epoch {}: \t Start validation...'.format(epoch+1))

        self.valid_preds, self.valid_targets = [], []
        self.model.eval()
        
        val_loss, val_metrics = self.valid_epoch()

        return val_loss, val_metrics

    def train_epoch(self, epoch) -> Tuple[float, float]:
        batch_iterator = tqdm(self.train_loader, desc="Iteration")

        trn_meter = AverageMeter()
        
        for batch_idx, (inputs, targets) in enumerate(batch_iterator):
            
            inputs, targets = self.inputs_to_device(inputs, targets)

            preds, loss = self.train_batch(batch_idx, inputs, targets)

            self.train_preds.append(to_numpy(preds))
            self.train_targets.append(to_numpy(targets))

            # epochのmeterはバッチステップでupdateしている
            trn_meter.update(loss)

            base_lr = self.get_optimizer_lr()
            batch_iterator.set_postfix(loss='{:.4f}'.format(trn_meter()), base_lr='{:.6f}'.format(base_lr))

            if self.cfg.logging_steps > 0 and self.global_meter.step % self.cfg.logging_steps == 0:
                pass
                # self.info("logging loss: {}".format(self.global_meter()))
                # 本来ならここでグローバルなログをloggingした方が良い?

        metric_score = self.metric_fn(np.concatenate(self.train_preds), np.concatenate(self.train_targets))
            
        return trn_meter(), metric_score
            

    def train_batch(self, batch_idx, inputs, targets):
        preds, loss = self.get_loss_batch(inputs, targets)

        if self.cfg.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        self.global_meter.update(loss.item())

        if (batch_idx + 1) % self.cfg.gradient_accumulation_steps == 0:
            if self.cfg.max_grad_norm > 0:
                if self.cfg.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.cfg.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

            # 勾配累積のステップ時にoptimizerをupdate
            # このタイミングでglobal_meterもupdate?
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_meter.step += 1

            if self.scheduler is not None:
                self.scheduler.step()

            self.on_batch_end()
            
        return preds, loss.item()

    def get_loss_batch(self, inputs, targets):
        preds = self.model(inputs)
        loss = self.loss_fn(preds, targets)

        return preds, loss

    def valid_epoch(self):
        batch_iterator = tqdm(self.valid_loader, desc="Iteration")

        val_meter = AverageMeter()

        with torch.no_grad():
            for inputs, targets in batch_iterator:
                
                inputs, targets = self.inputs_to_device(inputs, targets)
                
                preds, loss = self.valid_batch(inputs, targets)

                self.valid_preds.append(to_numpy(preds))
                self.valid_targets.append(to_numpy(targets))

                val_meter.update(loss)

                batch_iterator.set_postfix(loss='{:.4f}'.format(val_meter()))

        metric_score = self.metric_fn(np.concatenate(self.valid_preds), np.concatenate(self.valid_targets))

        return val_meter(), metric_score

    def valid_batch(self, inputs, targets):
        preds = self.model(inputs)
        loss = self.loss_fn(preds, targets)

        return preds, loss.item()
    
    def get_optimizer_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def on_epoch_end(self):
        pass

    def on_batch_end(self):
        pass

    def inputs_to_device(self, inputs, targets):
        return inputs, targets

    def set_scheduler(self, scheduler, params):
        self.scheduler = scheduler(**params)

    def logging_loss(self):
        pass

    def info(self, s):
        if self.logger is not None: self.logger.info(s)
        else: print(s)

    def debug(self, s):
        if self.logger is not None: self.logger.debug(s)
        else: print(s)