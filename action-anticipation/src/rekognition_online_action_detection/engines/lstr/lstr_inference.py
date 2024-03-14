# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from bisect import bisect_right
import time
import pickle as pkl

import torch
import numpy as np

from rekognition_online_action_detection.evaluation import compute_result

from ..base_inferences.perframe_det_batch_inference import do_perframe_det_batch_inference


from ..engines import INFERENCES as registry


@registry.register('LSTR')
def do_lstr_batch_inference(cfg,
                            model,
                            device,
                            logger):
    if cfg.MODEL.LSTR.INFERENCE_MODE == 'stream':
        do_lstr_stream_inference(cfg,
                                 model,
                                 device,
                                 logger)
    else:
        do_perframe_det_batch_inference(cfg,
                                        model,
                                        device,
                                        logger)


def do_lstr_stream_inference(cfg, model, device, logger):
    # Setup model to test mode
    model.eval()

    # Collect scores and targets
    pred_scores = []
    gt_targets = []

    def to_device(x, dtype=np.float32):
        return torch.as_tensor(x.astype(dtype)).unsqueeze(0).to(device)

    long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
    long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
    long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
    work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
    work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
    work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES

    # if len(cfg.DATA.TEST_SESSION_SET) != 1:
    #     raise RuntimeError('Only support testing one video each time for stream inference, will fix later')

    pred_scores_all = {}
    gt_targets_all = {}

    with torch.no_grad():
        for session_idx, session in enumerate(cfg.DATA.TEST_SESSION_SET):
            model.clear_cache()

            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                pred_scores = [[] for _ in range(cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES)]
            else:
                pred_scores = []
            gt_targets = []

            visual_inputs = np.load(osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.VISUAL_FEATURE, session + '.npy'), mmap_mode='r')
            motion_inputs = np.load(osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.MOTION_FEATURE, session + '.npy'), mmap_mode='r')
            object_inputs = np.load(osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.OBJECT_FEATURE, session + '.npy'), mmap_mode='r')
            target = np.load(osp.join(cfg.DATA.DATA_ROOT, cfg.INPUT.TARGET_PERFRAME, session + '.npy'))

            start_time = time.time()

            for work_start, work_end in zip(range(0, target.shape[0] + 1),
                                            range(work_memory_length, target.shape[0] + 1)):
                # Get target
                # target = target[::work_memory_sample_rate]

                # Get work memory
                work_indices = np.arange(work_start, work_end).clip(0)
                work_indices = work_indices[::work_memory_sample_rate]
                work_visual_inputs = to_device(visual_inputs[work_indices])
                work_motion_inputs = to_device(motion_inputs[work_indices])
                work_object_inputs = to_device(object_inputs[work_indices])

                # Get long memory
                PRECISE = True
                if PRECISE:
                    long_end = work_start - long_memory_sample_rate
                    if long_end < 0:
                        long_indices = [0 for _ in range(long_memory_num_samples)]
                        long_indices_set = [long_indices for _ in range(long_memory_sample_rate)]
                        long_visual_inputs = to_device(visual_inputs[long_indices])
                        long_motion_inputs = to_device(motion_inputs[long_indices])
                        long_object_inputs = to_device(object_inputs[long_indices])
                    else:
                        long_indices = long_indices_set[long_end % long_memory_sample_rate][1:] + [long_end]
                        long_indices_set[long_end % long_memory_sample_rate] = long_indices
                        long_visual_inputs = to_device(visual_inputs[[long_end]])
                        long_motion_inputs = to_device(motion_inputs[[long_end]])
                        long_object_inputs = to_device(object_inputs[[long_end]])
                else:
                    long_end = work_start - 1
                    if long_end == -1:
                        long_indices = [0 for _ in range(long_memory_num_samples)]
                        long_visual_inputs = to_device(visual_inputs[long_indices])
                        long_motion_inputs = to_device(motion_inputs[long_indices])
                        long_object_inputs = to_device(object_inputs[long_indices])
                    elif long_end % long_memory_sample_rate == 0:
                        long_indices = long_indices[1:] + [long_end]
                        long_visual_inputs = to_device(visual_inputs[[long_end]])
                        long_motion_inputs = to_device(motion_inputs[[long_end]])
                        long_object_inputs = to_device(object_inputs[[long_end]])
                    else:
                        long_visual_inputs = None
                        long_motion_inputs = None
                        long_object_inputs = None

                # Get memory key padding mask
                memory_key_padding_mask = np.zeros(len(long_indices))
                last_zero = bisect_right(long_indices, 0) - 1
                if last_zero > 0:
                    memory_key_padding_mask[:last_zero] = float('-inf')
                memory_key_padding_mask = torch.as_tensor(memory_key_padding_mask.astype(np.float32)).unsqueeze(0).to(device)

                score = model.stream_inference(
                    long_visual_inputs,
                    long_motion_inputs,
                    long_object_inputs,
                    work_visual_inputs,
                    work_motion_inputs,
                    work_object_inputs,
                    memory_key_padding_mask,
                    cache_num=long_memory_sample_rate if PRECISE else 1,
                    cache_id=long_end % long_memory_sample_rate if PRECISE else 0)[0]

                if cfg.DATA.DATA_NAME.startswith('EK'):
                    score = score.cpu().numpy()
                else:
                    score = score.softmax(dim=-1).cpu().numpy()

                if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                    if work_start == 0:
                        upsample_score = score[:-cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES].repeat(work_memory_sample_rate, axis=0)
                        upsample_score = upsample_score[work_start:work_end-1]
                        anticipate_score = score[-cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES:]
                        # coarse but sufficient
                        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                            combined_score = np.concatenate((upsample_score, anticipate_score[:t_a + 1]),
                                                            axis=0)
                            pred_scores[t_a].extend(list(combined_score))
                        gt_targets.extend(list(target[:work_end-1]))
                    else:
                        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                            pred_scores[t_a].append(list(score[t_a - cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES]))
                        gt_targets.append(list(target[work_end - 1]))
                else:
                    if work_start == 0:
                        gt_targets.extend(list(target[:work_end]))
                        pred_scores.extend(list(score))
                    else:
                        gt_targets.append(list(target[work_end - 1]))
                        pred_scores.append(list(score[-1]))

            end_time = time.time()
            logger.info('Running time: {:.3f} seconds'.format(end_time - start_time))

            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                    result = compute_result['perframe'](
                        cfg,
                        gt_targets,
                        pred_scores[t_a][:-1 - t_a],
                    )
                    sec = (t_a + 1) / cfg.DATA.FPS * cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE
                    logger.info('mAP of video ({:.2f}s) {}: {:.5f}'.format(sec, session, result['mean_AP']))
            else:
                result = compute_result['perframe'](
                    cfg,
                    gt_targets,
                    pred_scores,
                )
                logger.info('mAP of video {}: {:.5f}'.format(session, result['mean_AP']))


            gt_targets_all[session] = np.array(gt_targets)
            if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
                for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
                    pred_scores[t_a] = np.array(pred_scores[t_a][:- 1 - t_a])
                pred_scores_all[session] = np.stack(pred_scores, axis=0).transpose(1, 2, 0)
                pred_scores_all[session] = pred_scores_all[session]
            else:
                pred_scores_all[session] = np.array(pred_scores)


    # pkl.dump({
    #     'cfg': cfg,
    #     'perframe_pred_scores': pred_scores_all,
    #     'perframe_gt_targets': gt_targets_all,
    # }, open(osp.splitext(cfg.MODEL.CHECKPOINT)[0] + '.stream.pkl', 'wb'))

    if cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES > 0:
        maps_list = []
        for t_a in range(0, cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES):
            result = compute_result['perframe'](
                cfg,
                np.concatenate(list(gt_targets_all.values()), axis=0),
                np.concatenate(list(pred_scores_all.values()), axis=0)[:, :, t_a],
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
            np.concatenate(list(gt_targets_all.values()), axis=0),
            np.concatenate(list(pred_scores_all.values()), axis=0),
        )
        logger.info('Action detection perframe m{}: {:.5f}'.format(
            cfg.DATA.METRICS, result['mean_AP']
        ))

    gt_targets = gt_targets_all
    pred_scores = pred_scores_all
    pred_scores_verb = {}
    pred_scores_noun = {}

    # segment-level evaluation on EK55/EK100
    if cfg.DATA.DATA_NAME.startswith('EK'):
        try:
            import pandas as pd
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
                raise NotImplementedError(
                        'For TeSTra with decomposed classifier (V+N), please use batch mode, ie ``MODEL.LSTR.INFERENCE_MODE batch``'
                )
                preds_v = (pred_scores_verb[vid][start_step][None, 1:, ::-1]).transpose(0, 2, 1)
                preds_n = (pred_scores_noun[vid][start_step][None, 1:, ::-1]).transpose(0, 2, 1)
            else:
                preds_v = None
                preds_n = None
            if start_step >= cfg.MODEL.LSTR.ANTICIPATION_LENGTH:
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

        action_probs = softmax(action_scores.reshape(-1, action_scores.shape[-1]))
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
