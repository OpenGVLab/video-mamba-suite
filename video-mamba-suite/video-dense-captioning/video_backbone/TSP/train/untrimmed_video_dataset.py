from __future__ import division, print_function

import os
import pandas as pd
import numpy as np
import torch
import h5py

from torch.utils.data import Dataset
from torchvision.io import read_video


class UntrimmedVideoDataset(Dataset):
    '''
    UntrimmedVideoDataset:
        This dataset takes in temporal segments from untrimmed videos and samples fixed-length
        clips from each segment. Each item in the dataset is a dictionary with the keys:
            - "clip": A Tensor (dtype=torch.float) of the clip frames after applying transforms
            - "label-Y": A label from the `label_columns` (one key for each label) or -1 if label is missing for that clip
            - "gvf": The global video feature (GVF) vector if `global_video_features` parameter is not None
    '''

    def __init__(self, csv_filename, root_dir, clip_length, frame_rate, clips_per_segment, temporal_jittering,
            label_columns, label_mappings, seed=42, transforms=None, global_video_features=None, debug=False):
        '''
        Args:
            csv_filename (string): Path to the CSV file with temporal segments information and annotations.
                The CSV file must include the columns [filename, fps, t-start, t-end, video-duration] and
                the label columns given by the parameter `label_columns`.
            root_dir (string): Directory with all the video files.
            clip_length (int): The number of frames per clip.
            frame_rate (int): The effective frame rate (fps) to sample clips.
            clips_per_segment (int): The number of clips to sample per segment in the CSV file.
            temporal_jittering (bool): If True, clips are randomly sampled between t-start and t-end of
                each segment. Otherwise, clips are are sampled uniformly between t-start and t-end.
            seed (int): Seed of the random number generator used for the temporal jittering.
            transforms (callable): A function/transform that takes in a TxHxWxC video
                and returns a transformed version.
            label_columns (list of string): A list of the label columns in the CSV file.
                If more than one column is specified, the sample return a label for each.
            label_mappings (list of dict): A list of dictionaries to map the corresponding label
                from `label_columns` from a category string to an integer ID value.
            global_video_features (string): Path to h5 file containing global video features (optional)
            debug (bool): If true, create a debug dataset with 100 samples.
        '''
        df = UntrimmedVideoDataset._clean_df_and_remove_short_segments(pd.read_csv(csv_filename), clip_length, frame_rate)
        self.df = UntrimmedVideoDataset._append_root_dir_to_filenames_and_check_files_exist(df, root_dir)
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        self.clips_per_segment = clips_per_segment

        self.temporal_jittering = temporal_jittering
        self.rng = np.random.RandomState(seed=seed)
        self.uniform_sampling = np.linspace(0, 1, clips_per_segment)

        self.transforms = transforms

        self.label_columns = label_columns
        self.label_mappings = label_mappings
        for label_column, label_mapping in zip(label_columns, label_mappings):
            self.df[label_column] = self.df[label_column].map(lambda x: -1 if pd.isnull(x) else label_mapping[x])

        self.global_video_features = global_video_features
        self.debug = debug

    def __len__(self):
        return len(self.df) * self.clips_per_segment if not self.debug else 100

    def __getitem__(self, idx):
        sample = {}
        row = self.df.iloc[idx % len(self.df)]
        filename, fps, t_start, t_end = row['filename'], row['fps'], row['t-start'], row['t-end']

        # compute clip_t_start and clip_t_end
        clip_length_in_sec = self.clip_length / self.frame_rate
        ratio = self.rng.uniform() if self.temporal_jittering else self.uniform_sampling[idx//len(self.df)]
        clip_t_start = t_start + ratio * (t_end - t_start - clip_length_in_sec)
        clip_t_end = clip_t_start + clip_length_in_sec

        # get a tensor [clip_length, H, W, C] of the video frames between clip_t_start and clip_t_end seconds
        vframes, _, _ = read_video(filename=filename, start_pts=clip_t_start, end_pts=clip_t_end, pts_unit='sec')
        idxs = UntrimmedVideoDataset._resample_video_idx(self.clip_length, fps, self.frame_rate)
        vframes = vframes[idxs][:self.clip_length] # [:self.clip_length] for removing extra frames if isinstance(idxs, slice)
        if vframes.shape[0] != self.clip_length:
            raise RuntimeError(f'<UntrimmedVideoDataset>: got clip of length {vframes.shape[0]} != {self.clip_length}.'
                               f'filename={filename}, clip_t_start={clip_t_start}, clip_t_end={clip_t_end}, '
                               f'fps={fps}, t_start={t_start}, t_end={t_end}')

        # apply transforms
        sample['clip'] = self.transforms(vframes)

        # add labels
        for label_column in self.label_columns:
            sample[label_column] = row[label_column]

        # add global video feature if it exists
        if self.global_video_features:
            f = h5py.File(self.global_video_features, 'r')
            sample['gvf'] = torch.tensor(f[os.path.basename(filename).split('.')[0]][()])
            f.close()

        return sample

    @staticmethod
    def _clean_df_and_remove_short_segments(df, clip_length, frame_rate):
        # restrict all segments to be between [0, video-duration]
        df['t-end'] = np.minimum(df['t-end'], df['video-duration'])
        df['t-start'] = np.maximum(df['t-start'], 0)

        # remove segments that are too short to fit at least one clip
        segment_length = (df['t-end'] - df['t-start']) * frame_rate
        mask = segment_length >= clip_length
        num_segments = len(df)
        num_segments_to_keep = sum(mask)
        if num_segments - num_segments_to_keep > 0:
            df = df[mask].reset_index(drop=True)
            print(f'<UntrimmedVideoDataset>: removed {num_segments - num_segments_to_keep}='
                f'{100*(1 - num_segments_to_keep/num_segments):.2f}% from the {num_segments} '
                f'segments from the input CSV file because they are shorter than '
                f'clip_length={clip_length} frames using frame_rate={frame_rate} fps.')

        return df

    @staticmethod
    def _append_root_dir_to_filenames_and_check_files_exist(df, root_dir):
        df['filename'] = df['filename'].map(lambda f: os.path.join(root_dir, f))
        filenames = df.drop_duplicates('filename')['filename'].values
        for f in filenames:
            if not os.path.exists(f):
                raise ValueError(f'<UntrimmedVideoDataset>: file={f} does not exists. '
                                 f'Double-check root_dir and csv_filename inputs.')
        return df

    @staticmethod
    def _resample_video_idx(num_frames, original_fps, new_fps):
        step = float(original_fps) / new_fps
        if step.is_integer():
            # optimization: if step is integer, don't need to perform
            # advanced indexing
            step = int(step)
            return slice(None, None, step)
        idxs = torch.arange(num_frames, dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return idxs
