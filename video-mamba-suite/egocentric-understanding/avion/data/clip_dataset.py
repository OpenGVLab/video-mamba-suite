import csv
import glob
import os.path as osp
import pickle
import random
import numpy as np
import pandas as pd
import torch
import io
import decord
import mmengine
import json
import time
from datetime import datetime

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    frame_ids = np.convolve(np.linspace(start_frame, end_frame, num_segments + 1), [0.5, 0.5], mode='valid')
    if jitter:
        seg_size = float(end_frame - start_frame - 1) / num_segments
        shift = (np.random.rand(num_segments) - 0.5) * seg_size
        frame_ids += shift
    return frame_ids.astype(int).tolist()


def get_video_reader(videoname, num_threads, fast_rrc, rrc_params, fast_rcc, rcc_params):
    videoname = io.BytesIO(mmengine.get(videoname))
    
    video_reader = None
    if fast_rrc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rrc_params[0], height=rrc_params[0],
            use_rrc=True, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
        )
    elif fast_rcc:
        video_reader = decord.VideoReader(
            videoname,
            num_threads=num_threads,
            width=rcc_params[0], height=rcc_params[0],
            use_rcc=True,
        )
    else:
        video_reader = decord.VideoReader(videoname, num_threads=num_threads)
    return video_reader


def video_loader(root, vid, ext, second, end_second,
                 chunk_len=300, fps=30, clip_length=32,
                 threads=1,
                 fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False, rcc_params=(224, ),
                 jitter=False, sparse=True, down_sample_rate=1,):
    assert fps > 0, 'fps should be greater than 0'
    if chunk_len == -1:
        vr = get_video_reader(
            osp.join(root, '{}{}'.format(vid, ext)),
            num_threads=threads,
            fast_rrc=fast_rrc, rrc_params=rrc_params,
            fast_rcc=fast_rcc, rcc_params=rcc_params,
        )
        end_second = min(end_second, len(vr) / fps)

        frame_offset = int(np.round(second * fps))
        if sparse:
            # calculate frame_ids
            total_duration = max(int((end_second - second) * fps), clip_length)
            # sampling uniform frames in the interval
            frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)
        else:
            # calculate frame_ids
            frame_ids = np.arange(frame_offset + down_sample_rate // 2, 
                                  min(int(np.round(end_second * fps)), len(vr)) // down_sample_rate * down_sample_rate, 
                                  down_sample_rate)
        # load frames
        assert max(frame_ids) < len(vr)
        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    
        return torch.from_numpy(frames.astype(np.float32))

    else:
        chunk_start = int(second) // chunk_len * chunk_len
        chunk_end = int(end_second) // chunk_len * chunk_len
        # while True:
            
        #     video_filename = osp.join(root, vid, ("%04d"+ext) % (chunk_end//chunk_len))
        #     if not mmengine.exists(video_filename):
        #         print("{} does not exists!".format(video_filename),flush=True)
        #         chunk_end -= chunk_len
        #     else:
        #         vr = decord.VideoReader(io.BytesIO(mmengine.get(video_filename)))
        #         end_second = min(end_second, (len(vr) - 1) / fps + chunk_end)
        #         assert chunk_start <= chunk_end
        #         break
        # calculate frame_ids
        frame_ids = get_frame_ids(
            int(np.round(second * fps)),
            int(np.round(end_second * fps)),
            num_segments=clip_length, jitter=jitter
        )
        all_frames = []
        # allocate absolute frame-ids into the relative ones
        for chunk in range(chunk_start, chunk_end + chunk_len, chunk_len):
            rel_frame_ids = list(filter(lambda x: int(chunk * fps) <= x < int((chunk + chunk_len) * fps), frame_ids))
            rel_frame_ids = [int(frame_id - chunk * fps) for frame_id in rel_frame_ids]
            vr = get_video_reader(
                osp.join(root, vid, ("%04d"+ext) % (chunk//chunk_len)),
                num_threads=threads,
                fast_rrc=fast_rrc, rrc_params=rrc_params,
                fast_rcc=fast_rcc, rcc_params=rcc_params,
            )
            try:
                frames = vr.get_batch(rel_frame_ids).asnumpy()
            except decord.DECORDError as error:
                print(error)
                frames = vr.get_batch([0] * len(rel_frame_ids)).asnumpy()
            except IndexError:
                print(root, vid, ext, second, end_second)
            all_frames.append(frames)
            if sum(map(lambda x: x.shape[0], all_frames)) == clip_length:
                break
        res = torch.from_numpy(np.concatenate(all_frames, axis=0).astype(np.float32))
        assert res.shape[0] == clip_length, "{}, {}, {}, {}, {}, {}, {}".format(root, vid, second, end_second, res.shape[0], rel_frame_ids, frame_ids)
        return res


class VideoCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata, is_trimmed=True):
        self.dataset = dataset
        self.root = root
        self.metadata = metadata
        self.is_trimmed = is_trimmed

        if self.dataset == 'ego4d':
            with open(metadata, 'rb') as f:
                self.samples = pickle.load(f)
        elif self.dataset in ['ek100_cls', 'ek100_mir']:
            # print("now loading fps dict...")
            # video_list = list(mmengine.list_dir_or_file(self.root,list_file=False, list_dir=True))
            # fps_dict = {video: decord.VideoReader(io.BytesIO(mmengine.get(osp.join(self.root, video,"0000.mp4")))).get_avg_fps() for video in list(video_list)} 
            # print("fps dict loaded!")
            self.samples = []
            with open(metadata) as f:
                csv_reader = csv.reader(f)
                _ = next(csv_reader)  # skip the header
                for row in csv_reader:
                    pid, vid = row[1:3]
                    start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
                    narration = row[8]
                    verb, noun = int(row[10]), int(row[12])
                    # vid_path = '{}'.format(pid, vid) # the original directory structure
                    vid_path = '{}'.format(vid) # we use flatten directory structure
                    fps = 30
                    # start_frame = int(np.round(fps * start_timestamp))
                    # end_frame = int(np.ceil(fps * end_timestamp))
                    self.samples.append((vid_path, start_timestamp, end_timestamp, fps, narration, verb, noun))
            if self.dataset == 'ek100_mir':
                self.metadata_sentence = pd.read_csv(metadata[:metadata.index('.csv')] + '_sentence.csv')
                if 'train' in metadata:
                    self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_train.pkl'), 'rb'))
                elif 'test' in metadata:
                    self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_test.pkl'), 'rb'))
                else:
                    raise ValueError('{} should contain either "train" or "test"!'.format(metadata))
                self.relevancy = .1
        elif self.dataset == "egoschema":
            with open(metadata, "r") as f:
                self.samples = json.load(f)
        elif self.dataset == 'internvid-10m':
            with open(metadata, 'r') as f:
                self.samples = json.load(f)
        else:
            raise NotImplementedError

    def get_raw_item(
        self, i, is_training=True, num_clips=1,
        chunk_len=300, clip_length=32, clip_stride=2,
        sparse_sample=False,
        narration_selection='random',
        threads=1,
        fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False, rcc_params=(224,),
    ):

        if self.dataset == 'ego4d':
            vid, start_second, end_second, narration = self.samples[i][:4]
            frames = video_loader(self.root, vid, '.mp4',
                                  start_second, end_second,
                                  chunk_len=chunk_len,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            if isinstance(narration, list):
                if narration_selection == 'random':
                    narration = random.choice(narration)
                elif narration_selection == 'concat':
                    narration = '. '.join(narration)
                elif narration_selection == 'list':
                    pass
                else:
                    raise ValueError
            return vid, frames, narration
        elif self.dataset == 'ek100_mir':
            vid_path, start_second, end_second, fps, narration, verb, noun = self.samples[i]
            frames = video_loader(self.root, vid_path, '.mp4',
                                  start_second, end_second,
                                  chunk_len=chunk_len, fps=fps,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            if is_training:
                positive_list = np.where(self.relevancy_mat[i] > self.relevancy)[0].tolist()
                if positive_list != []:
                    pos = random.sample(positive_list, min(len(positive_list), 1))[0]
                    if pos < len(self.metadata_sentence) and pos < self.relevancy_mat.shape[1]:
                        return vid_path, frames, (self.metadata_sentence.iloc[pos][1], self.relevancy_mat[i][pos])
            else:
                return vid_path, frames, (narration, 1)
        elif self.dataset == 'ek100_cls':
            vid_path, start_second, end_second, fps, narration, verb, noun = self.samples[i]
            frames = video_loader(self.root, vid_path, '.mp4',
                                  start_second, end_second,
                                  chunk_len=chunk_len, fps=fps,
                                  clip_length=clip_length,
                                  threads=threads,
                                  fast_rrc=fast_rrc,
                                  rrc_params=rrc_params,
                                  fast_rcc=fast_rcc,
                                  rcc_params=rcc_params,
                                  jitter=is_training)
            return vid_path, frames, '{}:{}'.format(verb, noun)
        elif self.dataset == "internvid-10m":
            sample = self.samples[i]
            vid_path = sample["video"]
            narration = sample["caption"]
            vid_path = vid_path.replace(".mp4","")
            frames = video_loader(self.root, vid, '.mp4',
                        start_second, end_second,
                        chunk_len=chunk_len,
                        clip_length=clip_length,
                        threads=threads,
                        fast_rrc=fast_rrc,
                        rrc_params=rrc_params,
                        fast_rcc=fast_rcc,
                        rcc_params=rcc_params,
                        jitter=is_training)
        elif self.dataset == "egoschema":
            sample = self.samples[i]
            vid = sample["q_uid"]
            question = sample["question"]
            uid = sample["q_uid"]
            qa_caption = []
            for i in range(5):
                option = sample[f"option {i}"]
                text = f"{question} Is it '{option.lower()}'"
                # text = f"{option.lower()}, {option.lower()}, {option.lower()}"
                qa_caption.append(text)
            frames = video_loader(self.root, vid, '.mp4',
                                    0, 999999999,
                                    chunk_len=-1,
                                    clip_length=clip_length,
                                    threads=threads,
                                    fast_rrc=fast_rrc,
                                    rrc_params=rrc_params,
                                    fast_rcc=fast_rcc,
                                    rcc_params=rcc_params,
                                    jitter=is_training)
            return uid, frames, qa_caption
            
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class VideoCaptionDatasetCLIP(VideoCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None,
                 is_training=True, tokenizer=None,
                 chunk_len=300,
                 clip_length=32, clip_stride=2,
                 threads=1,
                 fast_rrc=False,
                 rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False,
                 rcc_params=(224,),
                 subsample_stride=None):
        super().__init__(dataset, root, metadata)

        self.full_samples = self.samples.copy()
        if isinstance(subsample_stride, int):
            self.samples = self.samples[::subsample_stride]
        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params

    def __getitem__(self, i):
        try:
            uid, frames, caption = self.get_raw_item(
                i, is_training=self.is_training,
                chunk_len=self.chunk_len,
                clip_length=self.clip_length,
                clip_stride=self.clip_stride,
                threads=self.threads,
                fast_rrc=self.fast_rrc,
                rrc_params=self.rrc_params,
                fast_rcc=self.fast_rcc,
                rcc_params=self.rcc_params,
            )
            
            # ek100_mir will also output relevancy value
            if isinstance(caption, tuple):
                caption, relevancy = caption
                    
            else:
                relevancy = 0.

            # apply transformation
            if self.transform is not None:
                frames = self.transform(frames)

            # tokenize caption
            if self.tokenizer is not None:
                if isinstance(caption, str): # one caption
                    caption = self.tokenizer(caption)[0]
                else:
                    caption = self.tokenizer(caption)

            if isinstance(caption, tuple):
                caption, mask = caption
                return uid, frames, caption, mask, relevancy
            else:
                return uid, frames, caption, relevancy
        except Exception as e:
            print(f'Error with sample {i}')
            ids = np.random.randint(0, 10000)
            return self.__getitem__(ids)
            


class VideoClassyDataset(VideoCaptionDatasetBase):
    def __init__(
        self, dataset, root, metadata, transform=None,
        is_training=True, label_mapping=None,
        num_clips=1,
        chunk_len=300,
        clip_length=32, clip_stride=2,
        threads=1,
        fast_rrc=False,
        rrc_params=(224, (0.5, 1.0)),
        fast_rcc=False,
        rcc_params=(224,),
        sparse_sample=False,
        is_trimmed=True):
        super().__init__(dataset, root, metadata, is_trimmed=is_trimmed)

        self.transform = transform
        self.is_training = is_training
        self.label_mapping = label_mapping
        self.num_clips = num_clips
        self.chunk_len = chunk_len
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.threads = threads
        self.fast_rrc = fast_rrc
        self.rrc_params = rrc_params
        self.fast_rcc = fast_rcc
        self.rcc_params = rcc_params
        self.sparse_sample = sparse_sample

    def __getitem__(self, i):
        vid_path, frames, label = self.get_raw_item(
            i, is_training=self.is_training,
            chunk_len=self.chunk_len,
            num_clips=self.num_clips,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            threads=self.threads,
            fast_rrc=self.fast_rrc,
            rrc_params=self.rrc_params,
            fast_rcc=self.fast_rcc,
            rcc_params=self.rcc_params,
            sparse_sample=self.sparse_sample,
        )

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        if self.label_mapping is not None:
            if isinstance(label, list):
                # multi-label case
                res_array = np.zeros(len(self.label_mapping))
                for lbl in label:
                    res_array[self.label_mapping[lbl]] = 1.
                label = res_array
            else:
                label = self.label_mapping[label]

        return vid_path, frames, label


def get_downstream_dataset(transform, crop_size, args, subset='train', label_mapping=None):
    if subset == 'train':
        return VideoClassyDataset(
            args.dataset, args.root, args.train_metadata, transform,
            is_training=True, label_mapping=label_mapping,
            num_clips=args.num_clips,
            chunk_len=args.video_chunk_length,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            threads=args.decode_threads,
            fast_rrc=args.fused_decode_crop, rrc_params=(crop_size, (0.5, 1.0)),
        )
    elif subset == 'val':
        return VideoClassyDataset(
            args.dataset, args.root, args.val_metadata, transform,
            is_training=False, label_mapping=label_mapping,
            num_clips=args.num_clips,
            chunk_len=args.video_chunk_length,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            threads=args.decode_threads,
            fast_rcc=args.fused_decode_crop, rcc_params=(crop_size, ),
            is_trimmed=not args.dataset == 'charades_ego',
        )
    else:
        assert ValueError("subset should be either 'train' or 'val'")
