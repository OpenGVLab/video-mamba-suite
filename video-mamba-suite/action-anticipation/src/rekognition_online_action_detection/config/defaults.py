# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Config Definition
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# Metadata
# ---------------------------------------------------------------------------- #
_C.SEED = 0

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.MODEL_NAME = ''
_C.MODEL.CHECKPOINT = ''

# ---------------------------------------------------------------------------- #
# Feature Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FEATURE_HEAD = CN()
_C.MODEL.FEATURE_HEAD.LINEAR_ENABLED = True
_C.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES = 1024
_C.MODEL.FEATURE_HEAD.DROPOUT = 0.

# ---------------------------------------------------------------------------- #
# Transformer Network
# ---------------------------------------------------------------------------- #
_C.MODEL.LSTR = CN()
_C.MODEL.LSTR.MAMBA_LAYER = 1
# Hyperparameters
_C.MODEL.LSTR.NUM_HEADS = 8
_C.MODEL.LSTR.DIM_FEEDFORWARD = 1024
_C.MODEL.LSTR.DROPOUT = 0.2
_C.MODEL.LSTR.DROPOUT_CLS = 0.0
_C.MODEL.LSTR.ACTIVATION = 'relu'
# Memory choices
_C.MODEL.LSTR.AGES_MEMORY_SECONDS = 0
_C.MODEL.LSTR.AGES_MEMORY_SAMPLE_RATE = 1
_C.MODEL.LSTR.LONG_MEMORY_SECONDS = 0
_C.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE = 1
_C.MODEL.LSTR.WORK_MEMORY_SECONDS = 8
_C.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE = 1
_C.MODEL.LSTR.WORK_MEMORY_USE_PE = True
_C.MODEL.LSTR.LONG_MEMORY_USE_PE = False
_C.MODEL.LSTR.LONG_MEMORY_INCLUDE_WORK = False
_C.MODEL.LSTR.LONG_MEMORY_INCLUDE_WORK2 = False
# Anticipation choices
_C.MODEL.LSTR.ANTICIPATION_SECONDS = 0
_C.MODEL.LSTR.ANTICIPATION_SAMPLE_RATE = 1
# for EK55/100
_C.MODEL.LSTR.V_N_CLASSIFIER = False
_C.MODEL.LSTR.LOSS_ANTICIPATE_ONLY = False
_C.MODEL.LSTR.LOSS_WEIGHT_V = 1.
_C.MODEL.LSTR.LOSS_WEIGHT_N = 1.
_C.MODEL.LSTR.FUSION_WEIGHT_V = 1.
_C.MODEL.LSTR.FUSION_WEIGHT_N = 1.
# Design choices
_C.MODEL.LSTR.ENC_ATTENTION_TYPE = 'dotproduct'
_C.MODEL.LSTR.ENC_ATTENTION_DECAY = 1.0
_C.MODEL.LSTR.ENC_MODULE = [
    [16, 1, True], [32, 2, True]
]
_C.MODEL.LSTR.DEC_MODULE = [-1, 2, True]
_C.MODEL.LSTR.FC_NORM = False
# Inference modes
_C.MODEL.LSTR.INFERENCE_MODE = 'batch'
# Position Embedding
# _C.MODEL.LSTR.POSITION_EMBEDDING = 'default'

# ---------------------------------------------------------------------------- #
# Criterion
# ---------------------------------------------------------------------------- #
_C.MODEL.CRITERIONS = [['MCE', {}]]

# ---------------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------------- #
_C.DATA = CN()
_C.DATA.DATA_INFO = 'data/data_info.json'
_C.DATA.DATA_NAME = None
_C.DATA.DATA_ROOT = None
_C.DATA.CLASS_NAMES = None
_C.DATA.NUM_CLASSES = None
_C.DATA.IGNORE_INDEX = None
_C.DATA.METRICS = None
_C.DATA.FPS = None
_C.DATA.TRAIN_SESSION_SET = None
_C.DATA.TEST_SESSION_SET = None
_C.DATA.EK_EXT_PATH = 'external/rulstm/RULSTM/data/ek55/'
_C.DATA.CLIP_MIXUP_RATE = 0.
_C.DATA.CLIP_MIXUP_SAMPLE = 'uniform'
_C.DATA.CLIP_MIXUP_RATIO = 0.

# ---------------------------------------------------------------------------- #
# Input
# ---------------------------------------------------------------------------- #
_C.INPUT = CN()
_C.INPUT.MODALITY = 'twostream'
# 'twostream'='visual+motion', 'threestream'='visual+motion+object'
_C.INPUT.VISUAL_FEATURE = 'rgb_anet_resnet50'
_C.INPUT.MOTION_FEATURE = 'flow_anet_resnet50'
_C.INPUT.OBJECT_FEATURE = 'flow_anet_resnet50'
_C.INPUT.TARGET_PERFRAME = 'target_perframe'

# ---------------------------------------------------------------------------- #
# Data Loader
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()
_C.DATA_LOADER.BATCH_SIZE = 32
_C.DATA_LOADER.NUM_WORKERS = 4
_C.DATA_LOADER.PIN_MEMORY = False

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.START_EPOCH = 1
_C.SOLVER.NUM_EPOCHS = 50

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.SOLVER.OPTIMIZER = 'adam'
_C.SOLVER.BASE_LR = 0.00005
_C.SOLVER.WEIGHT_DECAY = 0.00005
_C.SOLVER.MOMENTUM = 0.9

# ---------------------------------------------------------------------------- #
# Scheduler
# ---------------------------------------------------------------------------- #
_C.SOLVER.SCHEDULER = CN()
_C.SOLVER.SCHEDULER.SCHEDULER_NAME = 'multistep'
_C.SOLVER.SCHEDULER.MILESTONES = []
_C.SOLVER.SCHEDULER.GAMMA = 0.1
_C.SOLVER.SCHEDULER.WARMUP_FACTOR = 0.3
_C.SOLVER.SCHEDULER.WARMUP_EPOCHS = 10.0
_C.SOLVER.SCHEDULER.WARMUP_METHOD = 'linear'

# ---------------------------------------------------------------------------- #
# Others
# ---------------------------------------------------------------------------- #
_C.SOLVER.PHASES = ['train', 'test']
_C.SOLVER.SAVE_EVERY = 1

# ---------------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = 'checkpoints'
_C.SESSION = ''

# ---------------------------------------------------------------------------- #
# Misc
# ---------------------------------------------------------------------------- #
_C.VERBOSE = False


def get_cfg():
    return _C.clone()
