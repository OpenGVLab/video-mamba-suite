import datetime
import numpy as np
import os

import decord
import torch

from avion.data.classification_dataset import read_metadata
from avion.data.transforms import TubeMaskingGenerator


def video_loader_by_frames(
    root, vid, frame_ids, threads=1,
    fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
    fast_msc=False, msc_params=(224,),
    fast_cc=False, cc_params=(224,),
    hflip_prob=0., vflip_prob=0.,
):
    if fast_rrc:
        width, height = rrc_params[0], rrc_params[0]
    elif fast_msc:
        width, height = msc_params[0], msc_params[0]
    elif fast_cc:
        width, height = cc_params[0], cc_params[0]
    else:
        width, height = -1, -1
    vr = decord.VideoReader(
        os.path.join(root, vid), num_threads=threads,
        width=width, height=height,
        use_rrc=fast_rrc, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
        use_msc=fast_msc,
        use_centercrop=fast_cc,
        hflip_prob=hflip_prob, vflip_prob=vflip_prob,
    )
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = torch.zeros((len(frame_ids), 240, 320, 3))
    return torch.from_numpy(frames.astype(np.float32))


class KineticsDataset(torch.utils.data.Dataset):
    def __init__(self, root, metadata,
                 transform=None,
                 is_training=True,
                 clip_length=32,
                 clip_stride=2,
                 threads=1,
                 # fused augmentations need to be specified here
                 fast_rrc=False,
                 rrc_params=(224, (0.5, 1.0)),
                 fast_msc=False,
                 msc_params=(224, ),
                 fast_cc=False,
                 cc_params=(224,),
                 hflip_prob=0.5,
                 vflip_prob=0.,
                 verbose=False,
                 # for masking
                 mask_type='tube',
                 window_size=(8, 14, 14),
                 mask_ratio=0.9,
                 # for quick prototype
                 subsample_stride=None):
        super().__init__()

        self.root = root
        self.samples = read_metadata(metadata)
        self.full_samples = self.samples.copy()
        if isinstance(subsample_stride, int):
            self.samples = self.samples[::subsample_stride]
        self.transform = transform
        self.is_training = is_training
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_msc = fast_msc
        self.msc_params = msc_params
        self.fast_cc = fast_cc
        self.cc_params = cc_params
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.verbose = verbose

        if mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                window_size, mask_ratio,
            )
        elif mask_type == 'later':
            self.masked_position_generator = None
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        if self.verbose:
            print("[{}] __getitem__() starts at {}".format(os.getpid(), datetime.datetime.now()))

        video_id, num_frames, _ = self.samples[i]

        if num_frames > (self.clip_length + 1) * self.clip_stride:
            start_id = np.random.randint(num_frames - (self.clip_length + 1) * self.clip_stride)
        else:
            start_id = 0
        frame_ids = np.arange(start_id, start_id + self.clip_length * self.clip_stride, step=self.clip_stride)
        if self.is_training:
            shift = np.random.randint(self.clip_stride, size=self.clip_length)
            frame_ids += shift
        frame_ids = frame_ids % num_frames
        if self.is_training:
            frames = video_loader_by_frames(
                self.root, video_id + '.mp4' if '.' not in video_id else video_id, frame_ids, threads=self.threads,
                fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                fast_msc=self.fast_msc, msc_params=self.msc_params,
                fast_cc=False, cc_params=self.cc_params,
                hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
            )
        else:
            frames = video_loader_by_frames(
                self.root, video_id + '.mp4' if '.' not in video_id else video_id, frame_ids, threads=self.threads,
                fast_rrc=False, rrc_params=self.rrc_params,
                fast_cc=True, cc_params=self.cc_params,
                hflip_prob=0., vflip_prob=0.,
            )

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        if self.verbose:
            print("[{}] __getitem__() end at {}".format(os.getpid(), datetime.datetime.now()))

        if self.masked_position_generator is None:
            return frames
        else:
            return frames, self.masked_position_generator()

    def __len__(self):
        return len(self.samples)
