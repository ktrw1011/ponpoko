from typing import Tuple, List, Optional
import dataclasses
import pathlib
from pathlib import Path

from tqdm.autonotebook import tqdm, trange
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import AverageMeter, FoldScore, to_numpy, to_numpy


class Baseinfer:
    def __init__(
        self,
        model: nn.Module,
        test_loader:torch.utils.data.DataLoader,
        logger=None,
        ):
        self.model = model
        self.test_loader = test_loader
        self.logger = logger

    def infer(self):
        self.info('Start inference...')

        self.preds = []
        batch_iterator = tqdm(self.test_loader, desc="Iteration")

        with torch.no_grad():
            for inputs in batch_iterator:
                
                inputs = self.inputs_to_device(inputs)
                
                preds = self.infer_batch(inputs)

                self.preds.append(to_numpy(preds))

        self.on_infer_end(self.preds)

    def infer_batch(self, inputs):
        preds = self.model(inputs)

        return preds

    def inputs_to_device(self, inputs):
        return inputs

    def on_infer_end(self, preds):
        pass

    def load_model(self, checkpoint_file: Path):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.info(f"load {checkpoint_file}")

    def info(self, s):
        if self.logger is not None: self.logger.info(s)
        else: print(s)

    def debug(self, s):
        if self.logger is not None: self.logger.debug(s)
        else: print(s)