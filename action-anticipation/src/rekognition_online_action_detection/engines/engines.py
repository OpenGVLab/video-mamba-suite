# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from rekognition_online_action_detection.utils.registry import Registry

TRAINERS = Registry()
INFERENCES = Registry()


def do_train(cfg,
             data_loaders,
             model,
             criterion,
             optimizer,
             scheduler,
             device,
             checkpointer,
             logger):
    return TRAINERS[cfg.MODEL.MODEL_NAME](
        cfg,
        data_loaders,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        checkpointer,
        logger)


def do_inference(cfg,
                 model,
                 device,
                 logger):
    return INFERENCES[cfg.MODEL.MODEL_NAME](
        cfg,
        model,
        device,
        logger)
