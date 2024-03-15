# This file contains ShowAttendTell and AllImg model

# ShowAttendTell is from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
# https://arxiv.org/abs/1502.03044

# AllImg is a model where
# img feature is concatenated with word embedding at every time step as the input of lstm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *

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

    def forward(self, event, clip, clip_mask, seq):
        batch_size = clip.shape[0]

        state = self.init_hidden(batch_size)
        outputs = []
        seq = seq.long()

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = clip.data.new(batch_size).uniform_(0, 1)
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

            output, state = self.get_logprobs_state(it, event, clip, clip_mask, state)
            outputs.append(output)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)


    def get_logprobs_state(self, it, event , clip, clip_mask, state):
        xt = self.embed(it)
        output, state = self.core(xt, event , clip, clip_mask, state)
        logprobs = F.log_softmax(self.logit(self.dropout(output)), dim=1)
        return logprobs, state

    def sample(self, event , clip, clip_mask, opt={}):

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)

        batch_size = clip.shape[0]

        state = self.init_hidden(batch_size)

        seq = []
        seqLogprobs = []

        for t in range(self.max_caption_len + 1):
            if t == 0: # input <bos>
                it = clip.data.new(batch_size).long().zero_()
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

            logprobs, state = self.get_logprobs_state(it, event , clip, clip_mask, state)

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

class AllImgCore(nn.Module):
    def __init__(self, opt):
        super(AllImgCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob
        self.att_feat_size = opt.clip_context_dim

        self.opt = opt
        self.wordRNN_input_feats_type = opt.wordRNN_input_feats_type
        self.input_dim = self.decide_input_feats_dim()
        self.rnn = nn.LSTM(self.input_encoding_size + self.input_dim,
                           self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)
        assert self.wordRNN_input_feats_type == 'C'

    def decide_input_feats_dim(self):
        dim = 0
        if 'E' in self.wordRNN_input_feats_type:
            dim += self.opt.event_context_dim
        if 'C' in self.wordRNN_input_feats_type:
            dim += self.opt.clip_context_dim
        return dim

    def forward(self, xt, event, clip, clip_mask, state):
        input_feats = (clip * clip_mask.unsqueeze(2)).sum(1) / (clip_mask.sum(1, keepdims=True) + 1e-5)
        output, state = self.rnn(torch.cat([xt, input_feats], 1).unsqueeze(0), state)
        return output.squeeze(0), state


class LightCaptioner(Captioner):
    def __init__(self, opt):
        super(LightCaptioner, self).__init__(opt)
        self.core = AllImgCore(opt)
