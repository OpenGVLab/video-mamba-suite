# This file contains ShowAttendTell and AllImg model

# ShowAttendTell(Soft attention) is from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
# https://arxiv.org/abs/1502.03044

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

from pdvc.ops.modules import MSDeformAttnCap

class Captioner(nn.Module):
    def __init__(self, opt):
        super(Captioner, self).__init__()
        self.opt = opt

        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob
        self.max_caption_len = opt.max_caption_len

        self.ss_prob = 0.0 # Schedule sampling probability
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)

        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, batch_size, self.rnn_size).zero_(),
                weight.new(self.num_layers, batch_size, self.rnn_size).zero_())  # (h0, c0)

    def build_loss(self, input, target, mask):
        one_hot = torch.nn.functional.one_hot(target, self.opt.vocab_size+1)
        max_len = input.shape[1]
        output = - (one_hot[:, :max_len] * input * mask[:, :max_len, None]).sum(2).sum(1) / (mask.sum(1) + 1e-6)
        return output


    def forward(self,hs, reference, others, cap_tensor):

        seq = cap_tensor
        vid_num, query_num, _ = hs.shape
        assert vid_num == 1

        reference_points = reference
        input_flatten = others['memory']
        input_spatial_shapes = others['spatial_shapes']
        input_level_start_index = others['level_start_index']
        input_padding_mask = others['mask_flatten']
        if reference_points.shape[-1] == 2:
            reference_points = reference_points[:, :, None] \
                                     * torch.stack([others['valid_ratios']]*2, -1)[:, None]
        elif reference_points.shape[-1] == 1:
            reference_points = reference_points[:, :, None] * others['valid_ratios'][:, None, :, None]

        query = hs
        batch_size = query.shape[1]
        state = self.init_hidden(batch_size)
        outputs = []
        seq = seq.long()

        n_levels = self.core.n_levels
        if n_levels < self.core.opt.num_feature_levels:
            input_spatial_shapes = input_spatial_shapes[:n_levels]
            input_level_start_index = input_level_start_index[:n_levels]
            total_input_len = torch.prod(input_spatial_shapes, dim=1).sum()
            input_flatten = input_flatten[:, :total_input_len]
            input_padding_mask = input_padding_mask[:, :total_input_len]
            reference_points = reference_points[:, :, :n_levels]
            pass

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = hs.new_zeros(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()
                # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            output, state = self.get_logprobs_state(it, state, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask)
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)


    def get_logprobs_state(self, it, state, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, mask):
        xt = self.embed(it)
        output, state = self.core(xt, state, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, mask)
        logprobs = F.log_softmax(self.logit(self.dropout(output)), dim=1)
        return logprobs, state

    def sample(self,hs, reference, others, opt={}):

        vid_num, query_num, _ = hs.shape
        assert vid_num == 1
        batch_size = vid_num * query_num
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        reference_points = reference
        input_flatten = others['memory']
        input_spatial_shapes = others['spatial_shapes']
        input_level_start_index = others['level_start_index']
        input_padding_mask = others['mask_flatten']
        if reference_points.shape[-1] == 2:
            reference_points = reference_points[:, :, None] \
                                     * torch.stack([others['valid_ratios']]*2, -1)[:, None]
        elif reference_points.shape[-1] == 1:
            reference_points = reference_points[:, :, None] * others['valid_ratios'][:, None,:, None]
        query = hs

        n_levels = self.core.n_levels
        if n_levels < self.core.opt.num_feature_levels:
            input_spatial_shapes = input_spatial_shapes[:n_levels]
            input_level_start_index = input_level_start_index[:n_levels]
            total_input_len = torch.prod(input_spatial_shapes, dim=1).sum()
            input_flatten = input_flatten[:, :total_input_len]
            input_padding_mask = input_padding_mask[:, :total_input_len]
            reference_points = reference_points[:, :, :n_levels]
            pass

        state = self.init_hidden(batch_size)

        seq = []
        seqLogprobs = []

        for t in range(self.max_caption_len + 1):
            if t == 0: # input <bos>
                it = hs.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            logprobs, state = self.get_logprobs_state(it, state, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask)

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished & (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

        if seq==[] or len(seq)==0:
            return [],[]
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)


class ShowAttendTellCore(nn.Module):

    def __init__(self, opt):
        super(ShowAttendTellCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size

        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob
        #self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = int(opt.clip_context_dim / opt.cap_nheads)
        self.att_hid_size = opt.att_hid_size

        self.opt = opt
        self.wordRNN_input_feats_type = opt.wordRNN_input_feats_type
        self.input_dim = opt.hidden_dim * 2

        self.rnn = nn.LSTM(self.input_encoding_size + self.input_dim ,
                                                      self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)
        self.att_drop = nn.Dropout(0.5)

        d_model = opt.hidden_dim
        self.n_levels = opt.cap_num_feature_levels
        self.n_heads = opt.cap_nheads
        self.n_points = opt.cap_dec_n_points

        self.deformable_att = MSDeformAttnCap(d_model, self.n_levels, self.n_heads, self.n_points)

        if self.att_hid_size > 0:
            self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
            self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def get_input_feats(self, event, att_clip):
        input_feats = []
        if 'E' in self.wordRNN_input_feats_type:
            input_feats.append(event)
        if 'C' in self.wordRNN_input_feats_type:
            input_feats.append(att_clip)
        input_feats = torch.cat(input_feats,1)
        return input_feats

    def forward(self,xt, state, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask):

        joint_query = torch.cat((state[0][-1].unsqueeze(0), query), 2)
        # (N_, N_q, C)

        N_, Lq_, L_, _ = reference_points.shape

        # (N_ * M_, D_, Lq_, L_* P_)
        clip = self.deformable_att(joint_query, reference_points, input_flatten, input_spatial_shapes,
                                       input_level_start_index, input_padding_mask)
        clip = clip.reshape(N_, self.n_heads, -1, Lq_, self.n_levels * self.n_points).permute(0, 3, 1, 4, 2)
        clip = clip.reshape(N_ * Lq_, self.n_heads, self.n_levels * self.n_points, self.att_feat_size)
        att_size = self.n_levels * self.n_points

        att = self.ctx2att(clip)                             # (batch * att_size) * att_hid_size
        att = att.view(-1, self.n_heads, att_size, self.att_hid_size)     # batch * att_size * att_hid_size
        att_h = self.h2att(state[0][-1])                    # batch * att_hid_size
        att_h = att_h.unsqueeze(1).unsqueeze(1).expand_as(att)           # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = torch.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size

        weight = F.softmax(dot, dim=1)
        att_feats_ = clip.reshape(-1, att_size, self.att_feat_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size
        att_res = att_res.reshape(N_ * Lq_, self.n_heads, self.att_feat_size).flatten(1)
        input_feats = torch.cat((att_res.unsqueeze(0), query), 2)
        # print(xt.shape, input_feats.shape, query.shape, reference_points.shape)
        output, state = self.rnn(torch.cat([xt.unsqueeze(0), input_feats], 2), state)

        return output.squeeze(0), state


class LSTMDSACaptioner(Captioner):
    def __init__(self, opt):
        super(LSTMDSACaptioner, self).__init__(opt)
        self.core = ShowAttendTellCore(opt)

