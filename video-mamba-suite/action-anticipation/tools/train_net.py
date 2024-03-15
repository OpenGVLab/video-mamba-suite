# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
sys.path.append("path/to/repo/src")

from rekognition_online_action_detection.utils.parser import load_cfg
from rekognition_online_action_detection.utils.env import setup_environment
from rekognition_online_action_detection.utils.checkpointer import setup_checkpointer
from rekognition_online_action_detection.utils.logger import setup_logger
from rekognition_online_action_detection.datasets import build_data_loader
from rekognition_online_action_detection.models import build_model
from rekognition_online_action_detection.criterions import build_criterion
from rekognition_online_action_detection.optimizers import build_optimizer
from rekognition_online_action_detection.optimizers import build_scheduler
from rekognition_online_action_detection.engines import do_train


def main(cfg):
    # Setup configurations
    device = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='train')
    logger = setup_logger(cfg, phase='train')

    # Build data loaders
    data_loaders = {
        phase: build_data_loader(cfg, phase)
        for phase in cfg.SOLVER.PHASES
    }

    # Build model
    model = build_model(cfg, device)

    # Build criterion
    criterion = build_criterion(cfg, device)

    # Build optimizer
    optimizer = build_optimizer(cfg, model)

    # Load pretrained model and optimizer
    checkpointer.load(model, optimizer)

    # Build scheduler
    scheduler = build_scheduler(
        cfg, optimizer, len(data_loaders['train']))

    do_train(
        cfg,
        data_loaders,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        checkpointer,
        logger,
    )


if __name__ == '__main__':
    main(load_cfg())
