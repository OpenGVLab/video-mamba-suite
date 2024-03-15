from __future__ import division, print_function

import argparse
import os
import glob
import pandas as pd

from torchvision.io import read_video_timestamps
from joblib import Parallel, delayed


def get_video_stats(filename):
    pts, video_fps = read_video_timestamps(filename=filename, pts_unit='sec')
    if video_fps:
        stats = {'filename': os.path.basename(filename),
                 'video-duration': len(pts)/video_fps,
                 'fps': video_fps,
                 'video-frames': len(pts)}
    else:
        stats = {'filename': os.path.basename(filename),
                 'video-duration': None,
                 'fps': None,
                 'video-frames': None}
        print(f'WARNING: {filename} has an issue. video_fps = {video_fps}, len(pts) = {len(pts)}.')
    return stats


def main(args):
    print(args)

    filenames = glob.glob(os.path.join(args.video_folder, f'*.{args.ext}'))
    print(f'Number of video files: {len(filenames)}')

    all_stats = Parallel(n_jobs=args.workers)(
        delayed(get_video_stats)(
            filename=filename,
        ) for filename in filenames)

    df = pd.DataFrame(all_stats)
    df.to_csv(args.output_csv, index=False)
    print(f'Saved metadata to {args.output_csv}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a metadata CSV file with columns '
                                                 '[filename, video-duration, fps, video-frames] '
                                                 'for a given input video folder.')

    parser.add_argument('--video-folder', required=True, type=str,
                      help='Path to folder containing the raw video files')
    parser.add_argument('--ext', default='mp4', type=str,
                      help='Video files extension (default: mp4)')
    parser.add_argument('--output-csv', required=True, type=str,
                      help='Where to save the metadata CSV file')
    parser.add_argument('--workers', default=20, type=int,
                      help='Number of parallel processes to use to generate the output (default: 20)')

    args = parser.parse_args()

    main(args)
