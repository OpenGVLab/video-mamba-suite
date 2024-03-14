# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ['build_feature_head']

import torch
import torch.nn as nn

from rekognition_online_action_detection.utils.registry import Registry

FEATURE_HEADS = Registry()
FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'flow_nv_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50': 2048,
    'obj_ek55_fasterrcnn': 352,
}


@FEATURE_HEADS.register('THUMOS')
@FEATURE_HEADS.register('TVSeries')
@FEATURE_HEADS.register('EK55')
@FEATURE_HEADS.register('EK100')
class BaseFeatureHead(nn.Module):

    def __init__(self, cfg):
        super(BaseFeatureHead, self).__init__()

        if cfg.INPUT.MODALITY in ['visual', 'motion', 'object',
                                  'visual+motion', 'visual+motion+object']:
            self.with_visual = 'visual' in cfg.INPUT.MODALITY
            self.with_motion = 'motion' in cfg.INPUT.MODALITY
            self.with_object = 'object' in cfg.INPUT.MODALITY
        else:
            raise RuntimeError('Unknown modality of {}'.format(cfg.INPUT.MODALITY))

        if self.with_visual and self.with_motion and self.with_object:
            visual_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
            motion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            object_size = FEATURE_SIZES[cfg.INPUT.OBJECT_FEATURE]
            fusion_size = visual_size + motion_size + object_size
        elif self.with_visual and self.with_motion:
            visual_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
            motion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            fusion_size = visual_size + motion_size
        elif self.with_visual:
            fusion_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
        elif self.with_motion:
            fusion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
        elif self.with_object:
            fusion_size = FEATURE_SIZES[cfg.INPUT.OBJECT_FEATURE]

        self.d_model = fusion_size

        if cfg.MODEL.FEATURE_HEAD.LINEAR_ENABLED:
            if cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES != -1:
                self.d_model = cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES
            self.input_linear = nn.Sequential(
                nn.Linear(fusion_size, self.d_model),
                nn.ReLU(inplace=True),
            )
        else:
            self.input_linear = nn.Identity()
        if cfg.MODEL.FEATURE_HEAD.DROPOUT > 0:
            self.dropout = nn.Dropout(cfg.MODEL.FEATURE_HEAD.DROPOUT)
        else:
            self.dropout = None

    def forward(self, visual_input, motion_input, object_input):
        if self.with_visual and self.with_motion and self.with_object:
            fusion_input = torch.cat((visual_input, motion_input,
                                      object_input), dim=-1)
        elif self.with_visual and self.with_motion:
            fusion_input = torch.cat((visual_input, motion_input), dim=-1)
        elif self.with_visual:
            fusion_input = visual_input
        elif self.with_motion:
            fusion_input = motion_input
        elif self.with_object:
            fusion_input = object_input
        if self.dropout is not None:
            fusion_input = self.dropout(fusion_input)
        fusion_input = self.input_linear(fusion_input)
        return fusion_input


def build_feature_head(cfg):
    feature_head = FEATURE_HEADS[cfg.DATA.DATA_NAME]
    return feature_head(cfg)
