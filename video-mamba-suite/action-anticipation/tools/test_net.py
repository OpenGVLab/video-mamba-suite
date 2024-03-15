# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
sys.path.append("path/to/repo/src")

from rekognition_online_action_detection.utils.parser import load_cfg
from rekognition_online_action_detection.utils.env import setup_environment
from rekognition_online_action_detection.utils.checkpointer import setup_checkpointer
from rekognition_online_action_detection.utils.logger import setup_logger
from rekognition_online_action_detection.models import build_model
from rekognition_online_action_detection.engines import do_inference


def main(cfg):
    # Setup configurations
    device = setup_environment(cfg)
    checkpointer = setup_checkpointer(cfg, phase='test')
    logger = setup_logger(cfg, phase='test')

    # Build model
    model = build_model(cfg, device)

    # Load pretrained model
    checkpointer.load(model)

    do_inference(
        cfg,
        model,
        device,
        logger,
    )


if __name__ == '__main__':
    main(load_cfg())
