from __future__ import division, print_function

import os
import torch
import torchvision
import json
import datetime
import time
import numpy as np
import pandas as pd
import pickle as pkl
import sys

from torchvision import transforms
from torch import nn
from eval_video_dataset import EvalVideoDataset
sys.path.insert(0, '..')
from common import utils
from common import transforms as T
from models.model import Model


MODEL_URLS = {
    # main TSP models
    'r2plus1d_34-tsp_on_activitynet'        : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_5-0d2cf854.pth',
    'r2plus1d_34-tsp_on_thumos14'           : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_thumos14-max_gvf-backbone_lr_0.0001-fc_lr_0.004-epoch_4-e6a30b2f.pth',

    # main TAC baseline models
    'r2plus1d_34-tac_on_activitynet'        : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tac_on_activitynet-backbone_lr_0.0001-fc_lr_0.002-epoch_5-98ccac94.pth',
    'r2plus1d_34-tac_on_thumos14'           : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tac_on_thumos14-backbone_lr_0.00001-fc_lr_0.002-epoch_3-54b5c8aa.pth',
    'r2plus1d_34-tac_on_kinetics'           : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tac_on_kinetics-0547130e.pth',

    # other models from the GVF and backbone architecture ablation studies
    'r2plus1d_34-tsp_on_activitynet-avg_gvf': 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_activitynet-avg_gvf-backbone_lr_0.0001-fc_lr_0.004-epoch_5-8b74eaa2.pth',
    'r2plus1d_34-tsp_on_activitynet-no_gvf' : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_34-tsp_on_activitynet-no_gvf-backbone_lr_0.0001-fc_lr_0.004-epoch_5-fb38fdd2.pth',

    'r2plus1d_18-tsp_on_activitynet'        : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_18-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_6-22835b73.pth',
    'r2plus1d_18-tac_on_activitynet'        : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_18-tac_on_activitynet-backbone_lr_0.0001-fc_lr_0.004-epoch_5-9f56941a.pth',
    'r2plus1d_18-tac_on_kinetics'           : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r2plus1d_18-tac_on_kinetics-76ce975c.pth',

    'r3d_18-tsp_on_activitynet'             : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r3d_18-tsp_on_activitynet-max_gvf-backbone_lr_0.0001-fc_lr_0.002-epoch_6-85584422.pth',
    'r3d_18-tac_on_activitynet'             : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r3d_18-tac_on_activitynet-backbone_lr_0.001-fc_lr_0.01-epoch_5-31fd6e95.pth',
    'r3d_18-tac_on_kinetics'                : 'https://github.com/HumamAlwassel/TSP/releases/download/model_weights/r3d_18-tac_on_kinetics-dcd952c6.pth',
}


def evaluate(model, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter=' ')
    header = 'Feature extraction:'
    with torch.no_grad():
        for sample in metric_logger.log_every(data_loader, 10, header, device=device):
            clip = sample['clip'].to(device, non_blocking=True)
            logits, features = model(clip, return_features=True)
            data_loader.dataset.save_features(features, sample)
            # print(len(logits))
            # print(logits[0].shape, logits[1].shape)
            data_loader.dataset.save_output(logits, sample, ["action-label"])


def main(args):
    print(args)
    print('TORCH VERSION: ', torch.__version__)
    print('TORCHVISION VERSION: ', torchvision.__version__)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print('LOADING DATA')
    normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                            std=[0.22803, 0.22145, 0.216989])

    transform = torchvision.transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 171)),
        normalize,
        T.CenterCrop((112, 112))
    ])

    metadata_df = pd.read_csv(args.metadata_csv_filename)
    shards = np.linspace(0,len(metadata_df),args.num_shards+1).astype(int)
    start_idx, end_idx = shards[args.shard_id], shards[args.shard_id+1]
    print(f'shard-id: {args.shard_id + 1} out of {args.num_shards}, '
        f'total number of videos: {len(metadata_df)}, shard size {end_idx-start_idx} videos')

    metadata_df = metadata_df.iloc[start_idx:end_idx].reset_index()
    metadata_df['is-computed-already'] = metadata_df['filename'].map(lambda f:
        os.path.exists(os.path.join(args.output_dir, os.path.basename(f).split('.')[0] + '.npy')))
    metadata_df = metadata_df[metadata_df['is-computed-already']==False].reset_index(drop=True)
    print(f'Number of videos to process after excluding the ones already computed on disk: {len(metadata_df)}')

    dataset = EvalVideoDataset(
        metadata_df=metadata_df,
        root_dir=args.data_path,
        clip_length=args.clip_len,
        frame_rate=args.frame_rate,
        stride=args.stride,
        output_dir=args.output_dir,
        transforms=transform)

    print('CREATING DATA LOADER')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print(f'LOADING MODEL')
    if args.local_checkpoint:
        print(f'from the local checkpoint: {args.local_checkpoint}')
        pretrained_state_dict = torch.load(args.local_checkpoint, map_location='cpu')['model']
    else:
        print(f'from the GitHub released model: {args.released_checkpoint}')
        args.backbone = args.released_checkpoint.split('-')[0]
        pretrained_state_dict = torch.hub.load_state_dict_from_url(
            MODEL_URLS[args.released_checkpoint], progress=True, check_hash=True, map_location='cpu'
            )['model']

    # model with a dummy classifier layer
    model = Model(backbone=args.backbone, num_classes=[1], num_heads=1, concat_gvf=False)
    model.to(device)

    # remove the classifier layers from the pretrained model and load the backbone weights
    pretrained_state_dict = {k: v for k,v in pretrained_state_dict.items() if 'fc' not in k}
    state_dict = model.state_dict()
    pretrained_state_dict['fc.weight'] = state_dict['fc.weight']
    pretrained_state_dict['fc.bias'] = state_dict['fc.bias']
    model.load_state_dict(pretrained_state_dict)

    print('START FEATURE EXTRACTION')
    evaluate(model, data_loader, device)


if __name__ == '__main__':
    from opts import parse_args
    args = parse_args()
    main(args)
