# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
import argparse

import numpy as np
import pickle as pkl
import pandas as pd
import os.path as osp

from rekognition_online_action_detection.evaluation import compute_result
from external.rulstm.RULSTM.utils import (get_marginal_indexes, marginalize, softmax,
                                          topk_accuracy_multiple_timesteps,
                                          topk_recall_multiple_timesteps,
                                          tta)


def eval_fusion(pred_scores_files, weights):
    ###  Comments: Per-frame scores cost too much space...
    # perframe_gt_targets_fusion = {}
    # perframe_pred_scores_fusion = {}
    # for weight, pred_scores_file in zip(weights, pred_scores_files):
    #     pred_scores = pkl.load(open(pred_scores_file, 'rb'))
    #     cfg = pred_scores['cfg']
    #     assert cfg.DATA.DATA_NAME in ['EK55', 'EK100']
    #     perframe_gt_targets = pred_scores['perframe_gt_targets']
    #     perframe_pred_scores = pred_scores['perframe_pred_scores']

    #     for session in perframe_pred_scores:
    #         if session not in perframe_pred_scores_fusion:
    #             perframe_pred_scores_fusion[session] = weight * perframe_pred_scores[session]
    #         else:
    #             perframe_pred_scores_fusion[session] += weight * perframe_pred_scores[session]
    #         perframe_gt_targets_fusion[session] = perframe_gt_targets[session]

    # # Compute results
    # result = compute_result['perframe'](
    #     cfg,
    #     np.concatenate(list(perframe_gt_targets_fusion.values()), axis=0),
    #     np.concatenate(list(perframe_pred_scores_fusion.values()), axis=0),
    # )
    # logging.info('Action detection perframe m{}: {:.5f}'.format(
    #     cfg.DATA.METRICS, result['mean_AP']
    # ))

    action_scores = None
    for weight, pred_scores_file in zip(weights, pred_scores_files):
        pred_scores = pkl.load(open(pred_scores_file, 'rb'))
        cfg = pred_scores['cfg']
        assert cfg.DATA.DATA_NAME in ['EK55', 'EK100']
        if action_scores is None:
            action_scores = weight * pred_scores['action_scores']
        else:
            action_scores += weight * pred_scores['action_scores']
    # segment-level evaluation on EK55/EK100
    if cfg.DATA.DATA_NAME == 'EK55':
        path_to_data = 'external/rulstm/RULSTM/data/ek55/'
        segment_list = pd.read_csv(osp.join(path_to_data, 'validation.csv'),
                                   names=['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action'],
                                   skipinitialspace=True)
        predictions, labels, ids = [], [], []
        discarded_labels, discarded_ids = [], []
        for idx in segment_list['id']:
            segment = segment_list[segment_list['id'] == idx]
            start_step = int(np.floor(segment['start_f'].values.astype(int) / 30 * cfg.DATA.FPS))
            vid = segment['video'].values[0].strip()
            # preds = (perframe_pred_scores_fusion[vid][start_step][None, 1:, ::-1]).transpose(0, 2, 1)
            if start_step >= cfg.MODEL.LSTR.ANTICIPATION_LENGTH:
                # predictions.append(preds)
                labels.append(segment[['verb', 'noun', 'action']].values.astype(int))
                ids.append(idx)
            else:
                discarded_labels.append(segment[['verb', 'noun', 'action']].values.astype(int))
                discarded_ids.append(idx)
        
        actions = pd.read_csv(osp.join(path_to_data, 'actions.csv'), index_col='id')

        # action_scores = np.concatenate(predictions)
        labels = np.concatenate(labels)
        ids = np.array(ids)

        vi = get_marginal_indexes(actions, 'verb')
        ni = get_marginal_indexes(actions, 'noun')

        action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))
        # action_probs = action_scores.reshape(-1, action_scores.shape[-1])
        # action_probs = action_probs / action_probs.sum(-1, keepdims=True)
        print(action_probs.shape)
        verb_scores = marginalize(action_probs, vi).reshape(action_scores.shape[0],
                                                            action_scores.shape[1],
                                                            -1)
        noun_scores = marginalize(action_probs, ni).reshape(action_scores.shape[0],
                                                            action_scores.shape[1],
                                                            -1)

        include_discarded = (cfg.DATA.DATA_NAME == 'EK100')
        if include_discarded:
            dlab = np.array(discarded_labels)
            dislab = np.array(discarded_ids)
            ids = np.concatenate([ids, dislab])
            num_disc = len(dlab)
            labels = np.concatenate([labels, dlab])
            verb_scores = np.concatenate((verb_scores, np.zeros((num_disc, *verb_scores.shape[1:]))))
            noun_scores = np.concatenate((noun_scores, np.zeros((num_disc, *noun_scores.shape[1:]))))
            action_scores = np.concatenate((action_scores, np.zeros((num_disc, *action_scores.shape[1:]))))

        verb_labels, noun_labels, action_labels = labels[:, 0], labels[:, 1], labels[:, 2]
        verb_accuracies = topk_accuracy_multiple_timesteps(verb_scores, verb_labels)
        noun_accuracies = topk_accuracy_multiple_timesteps(noun_scores, noun_labels)
        action_accuracies = topk_accuracy_multiple_timesteps(action_scores, action_labels)

        many_shot_verbs = pd.read_csv(osp.join(path_to_data, 'EPIC_many_shot_verbs.csv'))['verb_class'].values
        many_shot_nouns = pd.read_csv(osp.join(path_to_data, 'EPIC_many_shot_nouns.csv'))['noun_class'].values
        actions = pd.read_csv(osp.join(path_to_data, 'actions.csv'))
        a_to_vn = {a[1]['id']: tuple(a[1][['verb', 'noun']].values)
                   for a in actions.iterrows()}
        many_shot_actions = [a for a, (v, n) in a_to_vn.items()
                             if v in many_shot_verbs or n in many_shot_nouns]
        
        verb_recalls = topk_recall_multiple_timesteps(verb_scores, verb_labels,
                                                      k=5, classes=many_shot_verbs)
        noun_recalls = topk_recall_multiple_timesteps(noun_scores, noun_labels,
                                                      k=5, classes=many_shot_nouns)
        action_recalls = topk_recall_multiple_timesteps(action_scores, action_labels,
                                                        k=5, classes=many_shot_actions)
        
        all_accuracies = np.concatenate([verb_accuracies, noun_accuracies, action_accuracies,
                                         verb_recalls, noun_recalls, action_recalls])
        all_accuracies = all_accuracies[[0, 1, 6, 2, 3, 7, 4, 5, 8]]
        indices = [
            ('Verb', 'Top-1 Accuracy'), ('Verb', 'Top-5 Accuracy'),
            ('Verb', 'Mean Top-5 Recall'),
            ('Noun', 'Top-1 Accuracy'), ('Noun', 'Top-5 Accuracy'),
            ('Noun', 'Mean Top-5 Recall'),
            ('Action', 'Top-1 Accuracy'), ('Action', 'Top-5 Accuracy'),
            ('Action', 'Mean Top-5 Recall'),
        ]
        cc = np.linspace(1 / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_LENGTH,
                         1 / cfg.DATA.FPS, cfg.MODEL.LSTR.ANTICIPATION_LENGTH, dtype=str)
        scores = pd.DataFrame(all_accuracies * 100, columns=cc,
                              index=pd.MultiIndex.from_tuples(indices))
        print(scores)

        tta_verb = tta(verb_scores, verb_labels)
        tta_noun = tta(noun_scores, noun_labels)
        tta_action = tta(action_scores, action_labels)
        print(f'\nMean TtA(5): VERB {tta_verb:0.2f} '
              f'NOUN: {tta_noun:0.2f} ACTION: {tta_action:0.2f}')
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_scores_files', nargs='+', required=True)
    parser.add_argument('--weights', type=float, nargs='+', required=True)
    args = parser.parse_args()

    assert len(args.pred_scores_files) == len(args.weights)
    eval_fusion(args.pred_scores_files, args.weights)
