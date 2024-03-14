# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
sys.path.append("path/to/repo")
sys.path.append("path/to/repo/src")

import os
import os.path as osp
from tqdm import tqdm
import pandas as pd

import torch
import numpy as np
import pickle as pkl

from rekognition_online_action_detection.datasets import build_dataset
from rekognition_online_action_detection.evaluation import compute_result


def do_perframe_det_batch_inference(cfg, model, device, logger):
    # Setup model to test mode
    model.eval()
    cfg.MODEL.CRITERIONS = [['MCE', {}]]
    
    data_loader = torch.utils.data.DataLoader(
        dataset=build_dataset(cfg, phase='test', tag='BatchInference'),
        batch_size=cfg.DATA_LOADER.BATCH_SIZE,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
    )

    # Collect scores and targets
    pred_scores = {}
    pred_scores_verb = {}
    pred_scores_noun = {}
    gt_targets = {}

    with torch.no_grad():
        pbar = tqdm(data_loader, desc='BatchInference')
        for batch_idx, data in enumerate(pbar, start=1):
            target = data[-4]
            if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                target, verb_target, noun_target = target

            score = model(*[x.to(device) for x in data[:-4]])
            if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                score, score_verb, score_noun = score
                score = score.cpu().numpy()
                score_verb = score_verb.cpu().numpy()
                score_noun = score_noun.cpu().numpy()
                cfg.DATA.NUM_VERBS = 126 if cfg.DATA.DATA_NAME == 'EK55' else 98
                cfg.DATA.NUM_NOUNS = 353 if cfg.DATA.DATA_NAME == 'EK55' else 301
            else:
                score = (score.cpu().numpy()
                         if cfg.DATA.DATA_NAME.startswith('EK')
                         else score.softmax(dim=-1).cpu().numpy())
            for bs, (session, query_indices, num_frames) in enumerate(zip(*data[-3:])):
                if session not in pred_scores:
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES))
                        if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                            pred_scores_verb[session] = np.zeros((num_frames, cfg.DATA.NUM_VERBS, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES))
                            pred_scores_noun[session] = np.zeros((num_frames, cfg.DATA.NUM_NOUNS, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES))
                    else:
                        pred_scores[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))
                if session not in gt_targets:
                    gt_targets[session] = np.zeros((num_frames, cfg.DATA.NUM_CLASSES))

                if query_indices[0] in torch.arange(0, cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE):
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                            full_indices = torch.cat((query_indices,
                                                     torch.arange(query_indices[-1] + cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE,
                                                                  query_indices[-1] + t_a + 1 + cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE)),
                                                     dim=0)
                            pred_scores[session][full_indices, :, t_a] = score[bs][:full_indices.shape[0]]
                            if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                                pred_scores_verb[session][full_indices, :, t_a] = score_verb[bs][:full_indices.shape[0]]    
                                pred_scores_noun[session][full_indices, :, t_a] = score_noun[bs][:full_indices.shape[0]]    
                        gt_targets[session][query_indices] = target[bs][:query_indices.shape[0]]
                    else:
                        pred_scores[session][query_indices] = score[bs]
                        gt_targets[session][query_indices] = target[bs]
                else:
                    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                            if query_indices[-1] + t_a + 1 < num_frames:
                                pred_scores[session][query_indices[-1] + t_a + 1, :, t_a] = score[bs][t_a - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]
                                if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                                    pred_scores_verb[session][query_indices[-1] + t_a + 1, :, t_a] = score_verb[bs][t_a - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]    
                                    pred_scores_noun[session][query_indices[-1] + t_a + 1, :, t_a] = score_noun[bs][t_a - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]
                        gt_targets[session][query_indices[-1]] = target[bs][-1]
                    else:
                        pred_scores[session][query_indices[-1]] = score[bs][-1]
                        gt_targets[session][query_indices[-1]] = target[bs][-1]

    # Save scores and targets
    # pkl.dump({
    #     'cfg': cfg,
    #     'perframe_pred_scores': pred_scores,
    #     'perframe_gt_targets': gt_targets,
    # }, open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '.pkl', 'wb'))

    # Compute results
    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
        maps_list = []
        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
            result = compute_result['perframe'](
                cfg,
                np.concatenate(list(gt_targets.values()), axis=0),
                np.concatenate(list(pred_scores.values()), axis=0)[:, :, t_a],
            )
            logger.info('Action anticipation ({:.2f}s) perframe m{}: {:.5f}'.format(
                (t_a + 1) / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE,
                cfg.DATA.METRICS, result['mean_AP']
            ))
            maps_list.append(result['mean_AP'])
        logger.info('Action anticipation (mean) perframe m{}: {:.5f}'.format(
            cfg.DATA.METRICS, np.mean(maps_list)
        ))
    else:
        result = compute_result['perframe'](
            cfg,
            np.concatenate(list(gt_targets.values()), axis=0),
            np.concatenate(list(pred_scores.values()), axis=0),
        )
        logger.info('Action detection perframe m{}: {:.5f}'.format(
            cfg.DATA.METRICS, result['mean_AP']
        ))

    # segment-level evaluation on EK55/EK100
    if cfg.DATA.DATA_NAME.startswith('EK'):
        try:
            from external.rulstm.RULSTM.utils import (get_marginal_indexes, marginalize, softmax,
                                                      topk_accuracy_multiple_timesteps,
                                                      topk_recall_multiple_timesteps,
                                                      tta)
        except:
            raise ModuleNotFoundError
        if cfg.DATA.DATA_NAME == 'EK55':
            path_to_data = 'external/rulstm/RULSTM/data/ek55/'
        elif cfg.DATA.DATA_NAME == 'EK100':
            path_to_data = 'external/rulstm/RULSTM/data/ek100/'
        segment_list = pd.read_csv(osp.join(path_to_data, 'validation.csv'),
                                   names=['id', 'video', 'start_f', 'end_f', 'verb', 'noun', 'action'],
                                   skipinitialspace=True)
        predictions, labels, ids = [], [], []
        predictions_verb, predictions_noun = [], []
        discarded_labels, discarded_ids = [], []
        for segment in segment_list.iterrows():
            start_step = int(np.floor(segment[1]['start_f'] / 30 * cfg.DATA.FPS))
            vid = segment[1]['video'].strip()
            preds = (pred_scores[vid][start_step][None, 1:, ::-1]).transpose(0, 2, 1)
            if cfg.MODEL.LSTR.V_N_CLASSIFIER:
                preds_v = (pred_scores_verb[vid][start_step][None, 1:, ::-1]).transpose(0, 2, 1)
                preds_n = (pred_scores_noun[vid][start_step][None, 1:, ::-1]).transpose(0, 2, 1)
            else:
                preds_v = None
                preds_n = None
            if start_step >= cfg.MODEL.LSTR.ANTICIPATION_LENGTH:
                # pt = []
                # for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_LENGTH):
                #     pt.append(pred_scores[vid][start_step - t_a, 1:, t_a][None, None, :])
                # preds = np.concatenate(pt, axis=1)
                predictions.append(preds)
                if preds_v is not None:
                    predictions_verb.append(preds_v)
                if preds_n is not None:
                    predictions_noun.append(preds_n)
                labels.append(segment[1][['verb', 'noun', 'action']].values.astype(int))
                ids.append(segment[1]['id'])
            else:
                discarded_labels.append(segment[1][['verb', 'noun', 'action']].values.astype(int))
                discarded_ids.append(segment[1]['id'])
        
        actions = pd.read_csv(osp.join(path_to_data, 'actions.csv'), index_col='id')

        action_scores = np.concatenate(predictions)
        if len(predictions_verb) > 0:
            verb_scores = np.concatenate(predictions_verb)
        else:
            verb_scores = None
        if len(predictions_noun) > 0:
            noun_scores = np.concatenate(predictions_noun)
        else:
            noun_scores = None
        labels = np.array(labels)
        ids = np.array(ids)

        vi = get_marginal_indexes(actions, 'verb')
        ni = get_marginal_indexes(actions, 'noun')

        pkl.dump({
            'cfg': cfg,
            'action_scores': action_scores,
            'verb_scores': verb_scores,
            'noun_scores': noun_scores,
        }, open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '.ek55_style.pkl', 'wb'))

        action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))
        # action_probs = action_probs / action_probs.sum(-1, keepdims=True)
        print(action_probs.shape)
        if verb_scores is None:
            verb_scores = marginalize(action_probs, vi).reshape(action_scores.shape[0],
                                                                action_scores.shape[1],
                                                                -1)
        else:
            verb_scores = softmax(verb_scores.reshape(-1, verb_scores.shape[-1]))
            verb_scores = verb_scores.reshape(action_scores.shape[0],
                                              action_scores.shape[1],
                                              -1)
        if noun_scores is None:
            noun_scores = marginalize(action_probs, ni).reshape(action_scores.shape[0],
                                                                action_scores.shape[1],
                                                                -1)
        else:
            noun_scores = softmax(noun_scores.reshape(-1, noun_scores.shape[-1]))
            noun_scores = noun_scores.reshape(action_scores.shape[0],
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

        if cfg.DATA.DATA_NAME == 'EK55':
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
        else:
            overall_verb_recalls = topk_recall_multiple_timesteps(verb_scores, verb_labels, k=5)
            overall_noun_recalls = topk_recall_multiple_timesteps(noun_scores, noun_labels, k=5)
            overall_action_recalls = topk_recall_multiple_timesteps(action_scores, action_labels, k=5)

            unseen = pd.read_csv(osp.join(path_to_data, 'validation_unseen_participants_ids.csv'), names=['id'], squeeze=True)
            tail_verbs = pd.read_csv(osp.join(path_to_data, 'validation_tail_verbs_ids.csv'), names=['id'], squeeze=True)
            tail_nouns = pd.read_csv(osp.join(path_to_data, 'validation_tail_nouns_ids.csv'), names=['id'], squeeze=True)
            tail_actions = pd.read_csv(osp.join(path_to_data, 'validation_tail_actions_ids.csv'), names=['id'], squeeze=True)

            unseen_bool_idx = pd.Series(ids).isin(unseen).values
            tail_verbs_bool_idx = pd.Series(ids).isin(tail_verbs).values
            tail_nouns_bool_idx = pd.Series(ids).isin(tail_nouns).values
            tail_actions_bool_idx = pd.Series(ids).isin(tail_actions).values

            tail_verb_recalls = topk_recall_multiple_timesteps(
                verb_scores[tail_verbs_bool_idx], verb_labels[tail_verbs_bool_idx], k=5)
            tail_noun_recalls = topk_recall_multiple_timesteps(
                noun_scores[tail_nouns_bool_idx], noun_labels[tail_nouns_bool_idx], k=5)
            tail_action_recalls = topk_recall_multiple_timesteps(
                action_scores[tail_actions_bool_idx], action_labels[tail_actions_bool_idx], k=5)
            
            unseen_verb_recalls = topk_recall_multiple_timesteps(
                verb_scores[unseen_bool_idx], verb_labels[unseen_bool_idx], k=5)
            unseen_noun_recalls = topk_recall_multiple_timesteps(
                noun_scores[unseen_bool_idx], noun_labels[unseen_bool_idx], k=5)
            unseen_action_recalls = topk_recall_multiple_timesteps(
                action_scores[unseen_bool_idx], action_labels[unseen_bool_idx], k=5)

            all_accuracies = np.concatenate(
                [overall_verb_recalls, overall_noun_recalls, overall_action_recalls,
                 unseen_verb_recalls, unseen_noun_recalls, unseen_action_recalls,
                 tail_verb_recalls, tail_noun_recalls, tail_action_recalls]
            ) #9 x 8

            indices = [
                ('Overall Mean Top-5 Recall', 'Verb'), ('Overall Mean Top-5 Recall', 'Noun'),
                ('Overall Mean Top-5 Recall', 'Action'),
                ('Unseen Mean Top-5 Recall', 'Verb'), ('Unseen Mean Top-5 Recall', 'Noun'),
                ('Unseen Mean Top-5 Recall', 'Action'),
                ('Tail Mean Top-5 Recall', 'Verb'), ('Tail Mean Top-5 Recall', 'Noun'),
                ('Tail Mean Top-5 Recall', 'Action'),
            ]

        cc = np.linspace(1 / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_LENGTH,
                        1 / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE,
                        cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES, dtype=str)
        scores = pd.DataFrame(all_accuracies * 100, columns=cc,
                            index=pd.MultiIndex.from_tuples(indices))
        logger.info(scores)

        tta_verb = tta(verb_scores, verb_labels)
        tta_noun = tta(noun_scores, noun_labels)
        tta_action = tta(action_scores, action_labels)
        logger.info(f'\nMean TtA(5): VERB {tta_verb:0.2f} '
                    f'NOUN: {tta_noun:0.2f} ACTION: {tta_action:0.2f}')
