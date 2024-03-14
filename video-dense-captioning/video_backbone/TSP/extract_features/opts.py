import argparse


RELEASED_GITHUB_MODELS = [
    # main TSP models
    'r2plus1d_34-tsp_on_activitynet',
    'r2plus1d_34-tsp_on_thumos14',

    # main TAC baseline models
    'r2plus1d_34-tac_on_activitynet',
    'r2plus1d_34-tac_on_thumos14',
    'r2plus1d_34-tac_on_kinetics',

    # other models from the GVF and backbone architecture ablation studies
    'r2plus1d_34-tsp_on_activitynet-avg_gvf',
    'r2plus1d_34-tsp_on_activitynet-no_gvf',

    'r2plus1d_18-tsp_on_activitynet',
    'r2plus1d_18-tac_on_activitynet',
    'r2plus1d_18-tac_on_kinetics',

    'r3d_18-tsp_on_activitynet',
    'r3d_18-tac_on_activitynet',
    'r3d_18-tac_on_kinetics',
]


def parse_args():
    parser = argparse.ArgumentParser(description='Features extraction script')

    parser.add_argument('--data-path', required=True,
                        help='Path to the directory containing the videos files')
    parser.add_argument('--metadata-csv-filename', required=True,
                        help='Path to the metadata CSV file')

    parser.add_argument('--backbone', default='r2plus1d_34',
                        choices=['r2plus1d_34', 'r2plus1d_18', 'r3d_18'],
                        help='Encoder backbone architecture (default r2plus1d_34). '
                             'Supported backbones are r2plus1d_34, r2plus1d_18, and r3d_18')
    parser.add_argument('--device', default='cuda',
                        help='Device to train on (default: cuda)')

    parser.add_argument('--released-checkpoint', default='r2plus1d-34_tsp-on-activitynet_max-gvf',
                        choices=RELEASED_GITHUB_MODELS,
                        help='Model checkpoint name to load from the released GitHub pretrained models. '
                             'The backbone parameter is set automatically if loading from a released model. '
                             'If `local-checkpoint` flag is not None, then this parameter is ignored and '
                             'a checkpoint is loaded from the given `local-checkpoint` path on disk.')
    parser.add_argument('--local-checkpoint', default=None,
                        help='Path to checkpoint on disk. If set, then read checkpoint from local disk. '
                            'Otherwise, load checkpoint from the released GitHub models.')

    parser.add_argument('--clip-len', default=16, type=int,
                        help='Number of frames per clip (default: 16)')
    parser.add_argument('--frame-rate', default=15, type=int,
                        help='Frames-per-second rate at which the videos are sampled (default: 15)')
    parser.add_argument('--stride', default=16, type=int,
                        help='Number of frames (after resampling with frame-rate) between consecutive clips (default: 16)')

    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size per GPU (default: 32)')
    parser.add_argument('--workers', default=6, type=int,
                        help='Number of data loading workers (default: 6)')

    parser.add_argument('--output-dir', required=True,
                        help='Path for saving features')
    parser.add_argument('--shard-id', default=0, type=int,
                        help='Shard id number. Must be between [0, num-shards)')
    parser.add_argument('--num-shards', default=1, type=int,
                        help='Number of shards to split the metadata-csv-filename')

    args = parser.parse_args()

    return args
