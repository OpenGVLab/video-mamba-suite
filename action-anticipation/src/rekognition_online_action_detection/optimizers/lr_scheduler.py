# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from bisect import bisect_left

import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


def _get_warmup_factor_at_iter(warmup_method,
                               this_iter,
                               warmup_iters,
                               warmup_factor):
    if this_iter >= warmup_iters:
        return 1.0

    if warmup_method == 'constant':
        return warmup_factor
    elif warmup_method == 'linear':
        alpha = this_iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError('Unknown warmup method: {}'.format(warmup_method))


class MultiStepLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                'Milestones should be a list of increasing integers. Got {}'.format(milestones)
            )
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr
            * self.gamma ** bisect_left(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self):
        # The new interface
        return self.get_lr()


class WarmupMultiStepLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 warmup_factor=0.3,
                 warmup_iters=500,
                 warmup_method='linear',
                 last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                'Milestones should be a list of increasing integers. Got {}'.format(milestones)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method,
            self.last_epoch,
            self.warmup_iters,
            self.warmup_factor,
        )
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_left(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self):
        # The new interface
        return self.get_lr()


class CosineLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 max_iters,
                 last_epoch=-1):
        self.max_iters = max_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self):
        # The new interface
        return self.get_lr()


class WarmupCosineLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 max_iters,
                 gamma=0.1,
                 warmup_factor=0.3,
                 warmup_iters=500,
                 warmup_method='linear',
                 last_epoch=-1):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method,
            self.last_epoch,
            self.warmup_iters,
            self.warmup_factor,
        )
        return [
            base_lr
            * warmup_factor
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]

    def _compute_values(self):
        # The new interface
        return self.get_lr()


def build_scheduler(cfg, optimizer, num_iters_per_epoch):
    """Unlike the PyTorch version, our schedulers adjust the learning rate
    according to iteration rather than epoch.
    """
    # Set last epoch (here 'epoch' is actually 'iters')
    last_epoch = (cfg.SOLVER.START_EPOCH - 1) * num_iters_per_epoch

    if cfg.SOLVER.SCHEDULER.SCHEDULER_NAME == 'multistep':
        # Convert milestones epochs to iters
        milestones = [(i - 1) * num_iters_per_epoch for i in cfg.SOLVER.SCHEDULER.MILESTONES]

        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=cfg.SOLVER.SCHEDULER.GAMMA,
            last_epoch=last_epoch)
    elif cfg.SOLVER.SCHEDULER.SCHEDULER_NAME == 'warmup_multistep':
        # Convert milestones epochs to iters
        milestones = [(i - 1) * num_iters_per_epoch for i in cfg.SOLVER.SCHEDULER.MILESTONES]

        # Convert warmup epochs to iters
        cfg.SOLVER.SCHEDULER.WARMUP_ITERS = int(cfg.SOLVER.SCHEDULER.WARMUP_EPOCHS * num_iters_per_epoch)

        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=cfg.SOLVER.SCHEDULER.GAMMA,
            warmup_factor=cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.SCHEDULER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.SCHEDULER.WARMUP_METHOD,
            last_epoch=last_epoch)
    elif cfg.SOLVER.SCHEDULER.SCHEDULER_NAME == 'cosine':
        # Get max number of iters
        max_iters = cfg.SOLVER.NUM_EPOCHS * num_iters_per_epoch

        scheduler = CosineLR(
            optimizer,
            max_iters=max_iters,
            last_epoch=last_epoch)
    elif cfg.SOLVER.SCHEDULER.SCHEDULER_NAME == 'warmup_cosine':
        # Get max number of iters
        max_iters = cfg.SOLVER.NUM_EPOCHS * num_iters_per_epoch

        # Convert warmup epochs to iters
        cfg.SOLVER.SCHEDULER.WARMUP_ITERS = int(cfg.SOLVER.SCHEDULER.WARMUP_EPOCHS * num_iters_per_epoch)

        scheduler = WarmupCosineLR(
            optimizer,
            max_iters=max_iters,
            gamma=cfg.SOLVER.SCHEDULER.GAMMA,
            warmup_factor=cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.SCHEDULER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.SCHEDULER.WARMUP_METHOD,
            last_epoch=last_epoch)
    else:
        raise RuntimeError('Unknown lr scheduler: {}'.format(cfg.SOLVER.SCHEDULER.SCHEDULER_NAME))
    return scheduler
