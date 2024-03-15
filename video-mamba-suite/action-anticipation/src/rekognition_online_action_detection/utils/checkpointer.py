# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['setup_checkpointer']

import os.path as osp

import torch


class Checkpointer(object):

    def __init__(self, cfg, phase):

        # Load pretrained checkpoint
        self.checkpoint = self._load_checkpoint(cfg.MODEL.CHECKPOINT)
        if self.checkpoint is not None and phase == 'train':
            cfg.SOLVER.START_EPOCH += self.checkpoint.get('epoch', 0)
        elif self.checkpoint is None and phase != 'train':
            raise RuntimeError('Cannot find checkpoint {}'.format(cfg.MODEL.CHECKPOINT))

        self.output_dir = cfg.OUTPUT_DIR

    def load(self, model, optimizer=None):
        if self.checkpoint is not None:
            model.load_state_dict(self.checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

    def save(self, epoch, model, optimizer):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, osp.join(self.output_dir, 'epoch-{}.pth'.format(epoch)))

    def _load_checkpoint(self, checkpoint):
        if checkpoint is not None and osp.isfile(checkpoint):
            return torch.load(checkpoint, map_location=torch.device('cpu'))
        return None


def setup_checkpointer(cfg, phase):
    return Checkpointer(cfg, phase)
