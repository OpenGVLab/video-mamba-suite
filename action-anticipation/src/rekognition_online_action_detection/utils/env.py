# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    'setup_random_seed',
    'setup_environment',
]

import os
import random

import torch
import numpy as np


def setup_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_environment(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.SEED is not None:
        setup_random_seed(cfg.SEED)
    return device
