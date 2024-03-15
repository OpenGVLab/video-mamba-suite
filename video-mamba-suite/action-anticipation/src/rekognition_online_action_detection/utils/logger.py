# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['setup_logger']

import os
import sys
import logging
import pprint


def setup_logger(cfg, phase, quiet=False):
    logger = logging.getLogger('rekognition')
    level = logging.DEBUG if cfg.VERBOSE else logging.INFO
    logger.setLevel(level)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    logger.addHandler(ch)

    if phase == 'train':
        log_file = os.path.join(cfg.OUTPUT_DIR, 'log.txt')
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    else:
        log_file = os.path.splitext(cfg.MODEL.CHECKPOINT)[0] + '.txt'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    if not quiet:
        logger.info(pprint.pformat(cfg))

    return logger
