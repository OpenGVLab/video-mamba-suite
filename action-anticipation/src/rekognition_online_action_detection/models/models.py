# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['build_model']

from rekognition_online_action_detection.utils.registry import Registry

META_ARCHITECTURES = Registry()


def build_model(cfg, device=None):
    model = META_ARCHITECTURES[cfg.MODEL.MODEL_NAME](cfg)
    from .weights_init import weights_init
    model.apply(weights_init)
    return model.to(device)
