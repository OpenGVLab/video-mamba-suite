# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
import argparse

import numpy as np
import pickle as pkl

from rekognition_online_action_detection.evaluation import compute_result


def eval_perframe(pred_scores_file):
    pred_scores = pkl.load(open(pred_scores_file, 'rb'))
    cfg = pred_scores['cfg']
    perframe_gt_targets = pred_scores['perframe_gt_targets']
    perframe_pred_scores = pred_scores['perframe_pred_scores']

    # Compute results
    result = compute_result['perframe'](
        cfg,
        np.concatenate(list(perframe_gt_targets.values()), axis=0),
        np.concatenate(list(perframe_pred_scores.values()), axis=0),
    )
    logging.info('Action detection perframe m{}: {:.5f}'.format(
        cfg.DATA.METRICS, result['mean_AP']
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_scores_file', type=str, required=True)
    args = parser.parse_args()

    eval_perframe(args.pred_scores_file)
