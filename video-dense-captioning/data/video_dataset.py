from collections import defaultdict
from itertools import chain
import json

import os

import scipy
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.interpolate import interp1d

def collate_fn(batch):
    batch_size = len(batch)
    feature_size = batch[0][0].shape[1]
    feature_list, gt_timestamps_list, labels, caption_list, gt_raw_timestamp, raw_duration, raw_caption, key = zip(
        *batch)

    max_video_length = max([x.shape[0] for x in feature_list])
    max_caption_length = max(chain(*[[len(caption) for caption in captions] for captions in caption_list]))
    total_caption_num = sum(chain([len(captions) for captions in caption_list]))
    # total_proposal_num = sum(chain([len(timestamp) for timestamp in timestamps_list]))

    # timestamps = list(chain(*timestamps_list))
    gt_timestamps = list(chain(*gt_timestamps_list))

    video_tensor = torch.FloatTensor(batch_size, max_video_length, feature_size).zero_()
    video_length = torch.FloatTensor(batch_size, 3).zero_()  # true length, sequence length
    video_mask = torch.BoolTensor(batch_size, max_video_length).zero_()

    caption_tensor = torch.LongTensor(total_caption_num, max_caption_length).zero_()

    caption_length = torch.LongTensor(total_caption_num).zero_()
    caption_mask = torch.BoolTensor(total_caption_num, max_caption_length).zero_()
    caption_gather_idx = torch.LongTensor(total_caption_num).zero_()
    # proposal_gather_idx = torch.LongTensor(total_proposal_num).zero_()

    # max_proposal_num = max(len(x) for x in timestamps_list)
    max_caption_num = max(len(x) for x in caption_list)

    # lnt_boxes_tensor = torch.zeros(batch_size, max_proposal_num, 4)
    gt_boxes_tensor = torch.zeros(batch_size, max_caption_num, 2)

    # index information for finding corresponding gt captions
    # gt_idx_tensor = torch.LongTensor(total_proposal_num, 3).zero_()

    # num_queries = len(query_gather_idx[0])
    # total_query_match_num = sum([len(list(chain(*q_cap))) for q_cap in query_gather_idx])
    # query_gather_idx_tensor = torch.LongTensor(total_query_match_num, 2)
    # query_idx_for_loss = torch.LongTensor(batch_size, 2)

    total_caption_idx = 0
    total_proposal_idx = 0

    for idx in range(batch_size):
        video_len = feature_list[idx].shape[0]
        # proposal_length = len(timestamps_list[idx])
        gt_proposal_length = len(gt_timestamps_list[idx])

        video_tensor[idx, :video_len, :] = torch.from_numpy(feature_list[idx])
        video_length[idx, 0] = float(video_len)
        video_length[idx, 1] = raw_duration[idx]
        video_length[idx, 2] = gt_proposal_length
        video_mask[idx, :video_len] = True

        # proposal_gather_idx[total_proposal_idx:total_proposal_idx + proposal_length] = idx
        # gt_idx_tensor[total_proposal_idx: total_proposal_idx + proposal_length, 0] = torch.from_numpy(
        #     total_caption_idx + gt_idx[idx])
        # gt_idx_tensor[total_proposal_idx: total_proposal_idx + proposal_length, 1] = idx
        # gt_idx_tensor[total_proposal_idx: total_proposal_idx + proposal_length, 2] = torch.from_numpy(gt_idx[idx])

        caption_gather_idx[total_caption_idx:total_caption_idx + gt_proposal_length] = idx

        # vid_query_idx = torch.as_tensor(query_gather_idx[idx])
        # vid_query_idx2 = torch.as_tensor(query_gather_idx2[idx])
        # query_gather_idx_tensor[total_caption_idx * num_queries: (total_caption_idx + gt_proposal_length) * num_queries,
        # 0] = total_caption_idx + vid_query_idx.reshape(-1)
        # query_gather_idx_tensor[total_caption_idx * num_queries: (total_caption_idx + gt_proposal_length) * num_queries,
        # 1] = num_queries * idx + vid_query_idx2.reshape(-1)

        # query_idx_for_loss[idx, 0] = total_caption_idx * num_queries
        # query_idx_for_loss[idx, 1] = vid_query_idx.shape[1]  # len(caption) for the idx-th video

        # lnt_boxes_tensor[idx, :proposal_length] = torch.tensor([[(ts[1]+ts[0])/(2*raw_duration[idx]), 0.5, (ts[1]-ts[0])/raw_duration[idx], 0.5] for ts in raw_timestamp[idx]]).float()
        gt_boxes_tensor[idx, :gt_proposal_length] = torch.tensor(
            [[(ts[1] + ts[0]) / (2 * raw_duration[idx]), (ts[1] - ts[0]) / raw_duration[idx]] for ts in
             gt_raw_timestamp[idx]]).float()

        for iidx, captioning in enumerate(caption_list[idx]):
            _caption_len = len(captioning)
            caption_length[total_caption_idx + iidx] = _caption_len
            caption_tensor[total_caption_idx + iidx, :_caption_len] = torch.from_numpy(captioning)
            caption_mask[total_caption_idx + iidx, :_caption_len] = True
        total_caption_idx += gt_proposal_length

    gt_boxes_mask = (gt_boxes_tensor != 0).sum(2) > 0

    target = [{'boxes': torch.tensor(
        [[(ts[1] + ts[0]) / (2 * raw_duration[i]), (ts[1] - ts[0]) / raw_duration[i]] for ts in
         gt_raw_timestamp[i]]).float(),
               'labels': torch.tensor(labels[i]).long(),
               'masks': None,
               'image_id': vid} for i, vid in enumerate(list(key))]

    dt = {
        "video":
            {
                "tensor": video_tensor,  # tensor,      (video_num, video_len, video_dim)
                "length": video_length,
                # tensor,      (video_num, 2), the first row is feature length, the second is time length
                "mask": video_mask,  # tensor,      (video_num, video_len,)
                "key": list(key),  # list,        (video_num)
                "target": target,
            },
        # "lnt":
        #     {
        #         "featstamps": timestamps,  # list,        (lnt_all_event_num, 2)
        #         'boxes': lnt_boxes_tensor,
        #         "boxes_mask": lnt_boxes_mask,
        #         "timestamp": list(raw_timestamp),  # list (len: video_num) of tensors (shape: (~lnt_event_num, 2))
        #         "gather_idx": proposal_gather_idx,  # tensor, (lnt_all_event_num)
        #         "gt_idx": gt_idx_tensor,  # tensor,      (lnt_all_event_num, 3)
        #     },
        # "query":
        #     {
        #         "gather_idx": query_gather_idx_tensor,
        #         # "gather_idx_for_loss": query_idx_for_loss,
        #     },
        "gt":
            {
                "featstamps": gt_timestamps,  # list,        (gt_all_event_num, 2)
                "timestamp": list(gt_raw_timestamp),  # list (len: video_num) of tensors (shape: (gt_event_num, 2))
                "gather_idx": caption_gather_idx,  # tensor,      (gt_all_event_num)
                "boxes": gt_boxes_tensor,
                "boxes_mask": gt_boxes_mask,
            },

        "cap":
            {
                "tensor": caption_tensor,  # tensor,      (gt_all_event_num, cap_len)
                "length": caption_length,  # tensor,      (gt_all_event_num)
                "mask": caption_mask,  # tensor,      (gt_all_event_num, cap_len, 1)
                "raw": list(raw_caption),  # list,        (video_num, ~gt_event_num, ~~caption_len)
            }
    }
    dt = {k1 + '_' + k2: v2 for k1, v1 in dt.items() for k2, v2 in v1.items()}
    return dt


class Translator(object):
    def __init__(self, translator_json, vocob_size):
        self.vocab_size = vocob_size
        self.vocab = json.load(open(translator_json, 'r'))
        assert self.vocab_size == len(self.vocab['word_to_ix'].keys())
        self.vocab['word_to_ix'] = defaultdict(lambda: self.vocab_size,
                                               self.vocab['word_to_ix'])
        self.vocab['ix_to_word'] = defaultdict(lambda: self.vocab_size,
                                               self.vocab['ix_to_word'])
        print('load translator, total_vocab: %d', len(self.vocab['ix_to_word']))

    def translate(self, sentence, max_len):
        tokens = [',', ':', '!', '_', ';', '-', '.', '?', '/', '"', '\\n', '\\', '.']
        for token in tokens:
            sentence = sentence.replace(token, ' ')
        sentence_split = sentence.replace('.', ' . ').replace(',', ' , ').lower().split()
        res = np.array(
            [0] + [self.vocab['word_to_ix'][word] for word in sentence_split][:max_len - 2] + [0])
        return res

    def rtranslate(self, sent_ids):
        for i in range(len(sent_ids)):
            if sent_ids[i] == 0:
                sent_ids = sent_ids[:i]
                break
        if len(sent_ids):
            return ' '.join([self.vocab['ix_to_word'][str(idx)] for idx in sent_ids]) + '.'
        else:
            return ''


class EDVCdataset(Dataset):

    def __init__(self, anno_file, feature_folder, translator_json, is_training, proposal_type, opt):

        super(EDVCdataset, self).__init__()
        self.anno = json.load(open(anno_file, 'r'))
        self.translator = Translator(translator_json, opt.vocab_size)
        self.max_caption_len = opt.max_caption_len
        self.keys = list(self.anno.keys())
        for json_path in opt.invalid_video_json:
            invalid_videos = json.load(open(json_path))
            self.keys = [k for k in self.keys if k[:13] not in invalid_videos]
        print('load captioning file, %d captioning loaded', len(self.keys))

        self.feature_folder = feature_folder
        self.feature_sample_rate = opt.feature_sample_rate
        self.opt = opt
        self.proposal_type = proposal_type
        self.is_training = is_training
        self.train_proposal_sample_num = opt.train_proposal_sample_num
        self.gt_proposal_sample_num = opt.gt_proposal_sample_num
        self.feature_dim = self.opt.feature_dim
        self.num_queries = opt.num_queries

    def __len__(self):
        return len(self.keys)

    def process_time_step(self, duration, timestamps_list, feature_length):
        duration = np.array(duration)
        timestamps = np.array(timestamps_list)
        feature_length = np.array(feature_length)
        featstamps = feature_length * timestamps / duration
        featstamps = np.minimum(featstamps, feature_length - 1).astype('int')
        featstamps = np.maximum(featstamps, 0).astype('int')
        return featstamps.tolist()

    def __getitem__(self, idx):
        raise NotImplementedError()


class PropSeqDataset(EDVCdataset):

    def __init__(self, anno_file, feature_folder, translator_pickle, is_training, proposal_type,

                 opt):
        super(PropSeqDataset, self).__init__(anno_file,
                                             feature_folder, translator_pickle, is_training, proposal_type,
                                             opt)

    def load_feats(self, key):
        vf_types = self.opt.visual_feature_type
        rescale_method = 'fix_length'
        if type(vf_types) == list:
            assert type(self.feature_folder) == list and len(vf_types) == len(self.feature_folder)
            feats_dict = {}
            all_padding = True
            for vf_type, vf_folder in zip(vf_types, self.feature_folder):
                feats, is_padding = get_feats(key, vf_type, vf_folder)
                all_padding = is_padding & all_padding
                feats_dict[vf_type] = feats
                if self.opt.data_rescale:
                    if rescale_method == 'fix_length':
                        rescale_len = self.opt.frame_embedding_num
                    elif rescale_method.startswith('follow'):
                        follow_type = rescale_method.split('_')[1]
                        assert follow_type in vf_types
                        rescale_len = len(feats_dict[follow_type])
                    else:
                        raise AssertionError('rescale_method must be \"fix_length\" or "follow_*"')
                    if feats.shape[0] != rescale_len:
                        feats = resizeFeature(feats, rescale_len, 'nearest')
                else:
                    feats = feats[::self.opt.feature_sample_rate]
                feats_dict[vf_type] = feats
            if all_padding:
                print('all feature files of video {} do not exist'.format(key))
            out = np.concatenate([feats_dict[type_] for type_ in vf_types], axis=-1)
        else:
            out, is_padding = get_feats(key, vf_types, self.feature_folder, data_norm=self.opt.data_norm)
            if self.opt.data_rescale:
                out = resizeFeature(out, self.opt.frame_embedding_num, 'nearest')
        assert out.shape[1] == self.feature_dim, 'wrong value of feature_dim'
        return out

    def __getitem__(self, idx):
        key = str(self.keys[idx])
        feats = self.load_feats(key)
        duration = self.anno[key]['duration']
        captions = self.anno[key]['sentences']
        gt_timestamps = self.anno[key]['timestamps']  # [gt_num, 2]
        action_labels = self.anno[key].get('action_labels', [0] * len(gt_timestamps))
        assert max(action_labels) <= self.opt.num_classes

        gt_sample_num = len(gt_timestamps) if (
                len(gt_timestamps) < self.gt_proposal_sample_num) else self.gt_proposal_sample_num
        random_ids = np.random.choice(list(range(len(gt_timestamps))), gt_sample_num, replace=False)

        captions = [captions[_] for _ in range(len(captions)) if _ in random_ids]
        gt_timestamps = [gt_timestamps[_] for _ in range(len(gt_timestamps)) if _ in random_ids]
        action_labels = [action_labels[_] for _ in range(len(action_labels)) if _ in random_ids]

        caption_label = [np.array(self.translator.translate(sent, self.max_caption_len)) for sent in captions]
        gt_featstamps = self.process_time_step(duration, gt_timestamps, feats.shape[0])

        # lnt_timestamps = gt_timestamps
        # lnt_featstamps = gt_featstamps
        # gt_idx = np.arange(len(gt_timestamps))
        # event_seq_idx = seq_gt_idx = np.expand_dims(gt_idx, 0)
        # lnt_scores = [1.] * len(lnt_featstamps)

        return feats, gt_featstamps, action_labels, caption_label, gt_timestamps, duration, captions, key


def iou(interval_1, interval_2):
    interval_1, interval_2 = map(np.array, (interval_1, interval_2))
    start, end = interval_2[None, :, 0], interval_2[None, :, 1]
    start_i, end_i = interval_1[:, None, 0], interval_1[:, None, 1]
    intersection = np.minimum(end, end_i) - np.maximum(start, start_i)
    union = np.minimum(np.maximum(end, end_i) - np.minimum(start, start_i), end - start + end_i - start_i)
    iou = intersection.clip(0) / (union + 1e-8)
    return iou


def sort_events(proposal_data):
    for vid in proposal_data.keys():
        v_data = proposal_data[vid]
        v_data = [p for p in v_data if p['score'] > 0]
        tmp = sorted(v_data, key=lambda x: x['segment'])
        proposal_data[vid] = tmp
    return proposal_data


def read_file(path, feat_dim, MEAN=0., VAR=1., data_norm=False):
    if os.path.exists(path) or 's3://' in path:
        ext = path.split('.')[-1]
        if ext == 'npy':
            feats = np.load(path)
        elif ext == 'csv':
            feats = pd.read_csv(path).values
        elif ext == 'pkl':
            feats = pickle.load(open(path))
        else:
            raise NotImplementedError
        padding = False
    else:
        print('{} not exists, use zero padding. '.format(path))
        feats = np.zeros((100, feat_dim))
        padding = True
    if data_norm:
        feats = (feats - MEAN) / np.sqrt(VAR)
    return feats, padding


def get_feats(key, vf_type, vf_folder, data_norm=False):
    MEAN = VAR = 0
    if vf_type == 'c3d':
        feat_dim = 500
        MEAN = -0.001915027447565527
        VAR = 1.9239444588254049
        path = os.path.join(vf_folder, key[0:13] + '.npy')

    elif vf_type == 'resnet':
        feat_dim = 2048
        MEAN = 0.41634243404998694
        VAR = 0.2569392081183313
        path = os.path.join(vf_folder, key[2:13] + '_resnet.npy')
    elif vf_type == 'bn':
        feat_dim = 1024
        MEAN = 0.8945046635916155
        VAR = 3.6579982046018844
        path = os.path.join(vf_folder, key[2:13] + '_bn.npy')
    elif vf_type == 'tsn_100':
        feat_dim = 400
        path = os.path.join(vf_folder, key[0:13] + '.csv')
    elif vf_type == 'i3d_rgb':
        feat_dim = 1024
        path = os.path.join(vf_folder, key[:13] + '_rgb.npy')
    elif vf_type == 'i3d_flow':
        feat_dim = 1024
        path = os.path.join(vf_folder, key[:13] + '_flow.npy')
    elif vf_type == 'tsp':
        feat_dim = 512
        path = os.path.join(vf_folder, key[0:13] + '.npy')
    elif vf_type == 'vggish':
        feat_dim = 128
        path = os.path.join(vf_folder, key[0:13] + '.npy')
    else:
        raise AssertionError('feature type error')

    feats, padding = read_file(path, feat_dim, MEAN, VAR, data_norm)

    if len(feats.shape) == 1:
        assert feats.shape[0] == feat_dim, 'load {} error, got shape {}'.format(path, feats.shape)

    assert feats.shape[1] == feat_dim, 'load {} error, got shape {}'.format(path, feats.shape)
    return feats, padding


def resizeFeature(inputData, newSize, sample_method):
    # inputX: (temporal_length,feature_dimension) #
    originalSize = len(inputData)
    # print originalSize
    if originalSize == 1:
        inputData = np.reshape(inputData, [-1])
        return np.stack([inputData] * newSize)
    x = np.array(range(originalSize))
    f = interp1d(x, inputData, axis=0, kind=sample_method)
    x_new = [i * float(originalSize - 1) / (newSize - 1) for i in range(newSize)]
    y_new = f(x_new)
    return y_new