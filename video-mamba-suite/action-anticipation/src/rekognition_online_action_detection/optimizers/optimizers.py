# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['build_optimizer']

import torch.optim as optim


def build_optimizer(cfg, model):
    if cfg.SOLVER.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            [{'params': model.parameters(), 'initial_lr': cfg.SOLVER.BASE_LR}],
            lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=cfg.SOLVER.MOMENTUM,
        )
    elif cfg.SOLVER.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            [{'params': model.parameters(), 'initial_lr': cfg.SOLVER.BASE_LR}],
            lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(
            [{'params': model.parameters(), 'initial_lr': cfg.SOLVER.BASE_LR}],
            lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise RuntimeError('Unknown optimizer: {}'.format(cfg.SOLVER.OPTIMIZER))
    return optimizer
