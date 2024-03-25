import os
import numpy as np
import decord
import mmengine
import io

from pytorchvideo.transforms import RandAugment, Normalize
import torch
from torch.utils.data._utils.collate import default_collate
import torchvision

from avion.data.random_erasing import RandomErasing
from avion.data.transforms import Permute, AdaptiveTemporalCrop, SpatialCrop


def read_metadata(metadata_fname):
    samples = []
    with open(metadata_fname) as split_f:
        data = split_f.readlines()
        for line in data:
            line_info = line.strip().split(',')
            assert len(line_info) == 2
            samples.append((line_info[0], int(line_info[1])))
    return samples


class VideoClsDataset(torch.utils.data.Dataset):
    def __init__ (self, root, metadata, mode='train',
                  clip_length=16, clip_stride=4,
                  threads=1,
                  crop_size=224, shorter_side_size=224,
                  new_height=256, new_width=340,
                  keep_aspect_ratio=True,
                  # fused augmentations need to be specified here
                  fast_rrc=False,
                  rrc_params=(224, (0.5, 1.0)),
                  fast_cc=False,
                  cc_params=(224,),
                  hflip_prob=0.5,
                  vflip_prob=0.,
                  num_segment=1, num_crop=1,
                  test_num_segment=5, test_num_crop=3,
                  args=None):
        self.root = root
        self.samples = read_metadata(metadata)
        assert mode in ['train', 'validation', 'test']
        self.mode = mode
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.crop_size = crop_size
        self.shorter_side_size = shorter_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_cc = fast_cc
        self.cc_params = cc_params
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.num_segment = num_segment
        self.num_crop = num_crop
        self.test_num_segment = test_num_segment
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        if mode == 'train':
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        if mode == 'train' and not fast_rrc:
            transforms_list = [
                Permute([3, 0, 1, 2]),    # T H W C -> C T H W
                torchvision.transforms.RandomResizedCrop(self.crop_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
            ]
            transforms_list += [
                Permute([1, 0, 2, 3]),
                RandAugment(magnitude=7, num_layers=4),
                Permute([1, 0, 2, 3]),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            if self.rand_erase:
                transforms_list += [
                    Permute([1, 0, 2, 3]),
                    RandomErasing(probability=self.args.reprob, mode='pixel', max_count=1, num_splits=1, cube=True, device='cpu'),
                    Permute([1, 0, 2, 3]),
                ]
            self.data_transform = torchvision.transforms.Compose(transforms_list)
        elif mode == 'validation' and not fast_cc:
            self.data_transform = torchvision.transforms.Compose([
                Permute([3, 0, 1, 2]),
                torchvision.transforms.CenterCrop(self.crop_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif mode == 'test':
            self.data_transform = torchvision.transforms.Compose([
                Permute([3, 0, 1, 2]),
                torchvision.transforms.Resize(self.shorter_side_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                AdaptiveTemporalCrop(self.clip_length, self.test_num_segment, self.clip_stride),
                SpatialCrop(crop_size=self.shorter_side_size, num_crops=self.test_num_crop),
            ])
        else:
            assert (mode == 'train' and fast_rrc) or (mode == 'validation' and fast_cc)
            self.data_transform = None

    def __getitem__(self, index):
        if self.mode == 'train' and not self.fast_rrc:
            args = self.args
            buffer = self._load_frames(self.root, self.samples[index][0], norm=True)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    index = np.random.randint(self.__len__())
                    buffer = self._load_frames(self.root, self.samples[index][0], norm=True)

            if args.repeated_aug > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.repeated_aug):
                    buffer_ = torch.from_numpy(buffer.astype(np.float32))
                    new_frames = self.data_transform(buffer_)
                    frame_list.append(new_frames)
                    label_list.append(self.samples[index][1])
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = torch.from_numpy(buffer.astype(np.float32))
                new_frames = self.data_transform(buffer)
                return new_frames, self.samples[index][1], index, {}
        elif self.mode == 'train' and self.fast_rrc:
            args = self.args
            buffer = self._load_frames(
                self.root, self.samples[index][0], norm=True,
                fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
            )
            if len(buffer) == 0:
                while len(buffer) == 0:
                    index = np.random.randint(self.__len__())
                    buffer = self._load_frames(
                        self.root, self.samples[index][0], norm=True,
                        fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                        hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                    )
                    buffer = torch.from_numpy(buffer.astype(np.float32))

            if args.repeated_aug > 1:
                frame_list = [buffer, ]
                label_list = [self.samples[index][1], ]
                index_list = [index, ]
                for _ in range(args.repeated_aug - 1):
                    buffer = self._load_frames(
                        self.root, self.samples[index][0], norm=True,
                        fast_rrc=self.fast_rrc, rrc_params=self.rrc_params,
                        hflip_prob=self.hflip_prob, vflip_prob=self.vflip_prob,
                    )
                    buffer = torch.from_numpy(buffer.astype(np.float32))
                    frame_list.append(buffer)
                    label_list.append(self.samples[index][1])
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                return buffer, self.samples[index][1], index, {}
        elif self.mode == 'validation' and not self.fast_cc:
            buffer = self._load_frames(self.root, self.samples[index][0], norm=True)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    index = np.random.randint(self.__len__())
                    buffer = self._load_frames(self.root, self.samples[index][0], norm=True)
            buffer = torch.from_numpy(buffer.astype(np.float32))
            buffer = self.data_transform(buffer)
            return buffer, self.samples[index][1], index, self.samples[index][0]
        elif self.mode == 'validation' and self.fast_cc:
            buffer = self._load_frames(
                self.root, self.samples[index][0], norm=True,
                fast_cc=True, cc_params=self.cc_params,
            )
            if len(buffer) == 0:
                while len(buffer) == 0:
                    index = np.random.randint(self.__len__())
                    buffer = self._load_frames(
                        self.root, self.samples[index][0], norm=True,
                        fast_cc=True, cc_params=self.cc_params,
                    )
            buffer = torch.from_numpy(buffer.astype(np.float32))
            return buffer, self.samples[index][1], index, self.samples[index][0]
        elif self.mode == 'test':
            buffer = self._load_frames(self.root, self.samples[index][0], norm=True)
            buffer = torch.from_numpy(buffer.astype(np.float32))
            buffer = self.data_transform(buffer)
            if isinstance(buffer, list):
                buffer = torch.stack(buffer)
            return buffer, self.samples[index][1], index, self.samples[index][0]


    def _load_frames(self, root, vid, norm=False,
                     fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                     fast_cc=False, cc_params=(224,),
                     hflip_prob=0., vflip_prob=0.):
        fname = os.path.join(root, vid)

        if not mmengine.exists(fname):
            print('No such video: ', fname)
            return []
        
        fname = mmengine.get(fname)
        if len(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", len(fname))
            return []
        fname = io.BytesIO(fname)

        try:
            if self.keep_aspect_ratio:
                if fast_rrc:
                    width, height = rrc_params[0], rrc_params[0]
                elif fast_cc:
                    width, height = cc_params[0], cc_params[0]
                else:
                    width, height = -1, -1
                vr = decord.VideoReader(
                    fname, num_threads=self.threads,
                    width=width,
                    height=height,
                    use_rrc=fast_rrc, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
                    use_centercrop=fast_cc,
                    hflip_prob=hflip_prob, vflip_prob=vflip_prob,
                )
            else:
                vr = decord.VideoReader(
                    fname, num_treads=self.threads,
                    width=self.new_width, height=self.new_height)
        except (IndexError, decord.DECORDError) as error:
            print(error)
            print("Fail to load video: ", fname)
            return []

        if self.mode == 'test':
            all_index = [x for x in range(0, len(vr))]
            while len(all_index) < self.clip_length * self.clip_stride:
                all_index.append(all_index[-1])
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            if norm:
                buffer = buffer.astype(np.float32)
                buffer /= 255.
            return buffer
        # handle temporal segments
        total_length = int(self.clip_length * self.clip_stride)
        seg_len = len(vr) // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= total_length:
                index = np.arange(0, seg_len, step=self.clip_stride)
                index = np.concatenate((index, np.ones(self.clip_length - len(index)) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_id = np.random.randint(total_length, seg_len)
                start_id = end_id - total_length
                index = np.linspace(start_id, end_id, num=self.clip_length)
                index = np.clip(index, start_id, end_id - 1).astype(np.int64)
            index = index + i * seg_len
            all_index.extend(list((index)))

        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        if norm:
            buffer = buffer.astype(np.float32)
            buffer /= 255.
        return buffer

    def __len__(self):
        return len(self.samples)


def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    inputs, labels, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data
