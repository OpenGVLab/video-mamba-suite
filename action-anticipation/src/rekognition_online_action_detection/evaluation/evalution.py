# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from .postprocessing import postprocessing as default_pp
from .metrics import perframe_average_precision
from .metrics import perstage_average_precision

from rekognition_online_action_detection.utils.registry import Registry

compute_result = Registry()


@compute_result.register('perframe')
def eval_perframe(cfg, ground_truth, prediction, **kwargs):
    class_names = kwargs.get('class_names', cfg.DATA.CLASS_NAMES)
    ignore_index = kwargs.get('ignore_index', cfg.DATA.IGNORE_INDEX)
    metrics = kwargs.get('metrics', cfg.DATA.METRICS)
    postprocessing = kwargs.get('postprocessing', default_pp(cfg.DATA.DATA_NAME))
    return perframe_average_precision(
        ground_truth=ground_truth,
        prediction=prediction,
        class_names=class_names,
        ignore_index=ignore_index,
        metrics=metrics,
        postprocessing=postprocessing,
    )


@compute_result.register('perstage')
def eval_perstage(cfg, ground_truth, prediction, **kwargs):
    class_names = kwargs.get('class_names', cfg.DATA.CLASS_NAMES)
    ignore_index = kwargs.get('ignore_index', cfg.DATA.IGNORE_INDEX)
    metrics = kwargs.get('metrics', cfg.DATA.METRICS)
    postprocessing = kwargs.get('postprocessing', default_pp(cfg.DATA.DATA_NAME))
    return perstage_average_precision(
        ground_truth=ground_truth,
        prediction=prediction,
        class_names=class_names,
        ignore_index=ignore_index,
        metrics=metrics,
        postprocessing=postprocessing,
    )
