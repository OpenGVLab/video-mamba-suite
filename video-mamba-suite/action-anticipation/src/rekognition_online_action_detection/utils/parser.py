# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import argparse
import json

from rekognition_online_action_detection.config.defaults import get_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Rekognition Face Anti-Spoofing')
    parser.add_argument(
        '--config_file',
        default='',
        type=str,
        help='path to yaml config file',
    )
    parser.add_argument(
        '--gpu',
        default='0',
        type=str,
        help='specify visible devices'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs='*',
        help='modify config options using the command-line',
    )
    return parser.parse_args()


def assert_and_infer_cfg(cfg, args):
    # Setup the visible devices
    cfg.GPU = args.gpu

    # Infer data info
    with open(cfg.DATA.DATA_INFO, 'r') as f:
        data_info = json.load(f)[cfg.DATA.DATA_NAME]

    cfg.DATA.DATA_ROOT = data_info['data_root'] if cfg.DATA.DATA_ROOT is None else cfg.DATA.DATA_ROOT
    cfg.DATA.CLASS_NAMES = data_info['class_names'] if cfg.DATA.CLASS_NAMES is None else cfg.DATA.CLASS_NAMES
    cfg.DATA.NUM_CLASSES = data_info['num_classes'] if cfg.DATA.NUM_CLASSES is None else cfg.DATA.NUM_CLASSES
    cfg.DATA.IGNORE_INDEX = data_info['ignore_index'] if cfg.DATA.IGNORE_INDEX is None else cfg.DATA.IGNORE_INDEX
    cfg.DATA.METRICS = data_info['metrics'] if cfg.DATA.METRICS is None else cfg.DATA.METRICS
    cfg.DATA.FPS = data_info['fps'] if cfg.DATA.FPS is None else cfg.DATA.FPS
    cfg.DATA.TRAIN_SESSION_SET = data_info['train_session_set'] if cfg.DATA.TRAIN_SESSION_SET is None else cfg.DATA.TRAIN_SESSION_SET
    cfg.DATA.TEST_SESSION_SET = data_info['test_session_set'] if cfg.DATA.TEST_SESSION_SET is None else cfg.DATA.TEST_SESSION_SET

    # Ignore two mis-labeled videos
    if False and cfg.DATA_NAME == 'THUMOS':
        cfg.DATA.TEST_SESSION_SET.remove('video_test_0000270')
        cfg.DATA.TEST_SESSION_SET.remove('video_test_0001496')

    # Input assertions
    if cfg.INPUT.MODALITY == 'twostream':
        cfg.INPUT.MODALITY = 'visual+motion'
    if cfg.INPUT.MODALITY == 'threestream':
        cfg.INPUT.MODALITY = 'visual+motion+object'
    assert cfg.INPUT.MODALITY in ['visual', 'motion', 'object',
                                  'visual+motion', 'visual+motion+object']

    # Infer memory
    if cfg.MODEL.MODEL_NAME in ['LSTR']:
        cfg.MODEL.LSTR.AGES_MEMORY_LENGTH = cfg.MODEL.LSTR.AGES_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.LONG_MEMORY_LENGTH = cfg.MODEL.LSTR.LONG_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.WORK_MEMORY_LENGTH = cfg.MODEL.LSTR.WORK_MEMORY_SECONDS * cfg.DATA.FPS
        cfg.MODEL.LSTR.TOTAL_MEMORY_LENGTH = \
            cfg.MODEL.LSTR.AGES_MEMORY_LENGTH + \
            cfg.MODEL.LSTR.LONG_MEMORY_LENGTH + \
            cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
        cfg.MODEL.LSTR.ANTICIPATION_LENGTH = cfg.MODEL.LSTR.ANTICIPATION_SECONDS * cfg.DATA.FPS
        assert cfg.MODEL.LSTR.AGES_MEMORY_LENGTH % cfg.MODEL.LSTR.AGES_MEMORY_SAMPLE_RATE == 0
        assert cfg.MODEL.LSTR.LONG_MEMORY_LENGTH % cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE == 0
        assert cfg.MODEL.LSTR.WORK_MEMORY_LENGTH % cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE == 0
        assert cfg.MODEL.LSTR.ANTICIPATION_LENGTH % cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE == 0
        cfg.MODEL.LSTR.AGES_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.AGES_MEMORY_LENGTH // cfg.MODEL.LSTR.AGES_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH // cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH // cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
        cfg.MODEL.LSTR.TOTAL_MEMORY_NUM_SAMPLES = \
            cfg.MODEL.LSTR.AGES_MEMORY_NUM_SAMPLES + \
            cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES + \
            cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES = cfg.MODEL.LSTR.ANTICIPATION_LENGTH // cfg.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE

        assert cfg.MODEL.LSTR.INFERENCE_MODE in ['batch', 'stream']

    # Infer output dir
    config_name = osp.splitext(args.config_file)[0].split('/')[1:]
    cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, *config_name)
    if cfg.SESSION:
        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, cfg.SESSION)


def load_cfg():
    args = parse_args()
    cfg = get_cfg()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    assert_and_infer_cfg(cfg, args)
    return cfg
