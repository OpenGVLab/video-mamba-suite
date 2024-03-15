# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

import time

class DotProductAttention(nn.Module):

    def __init__(self, dropout=0.0, alpha=1.0, num_heads=8):
        super(DotProductAttention, self).__init__()

        self.dropout = dropout
        self.alpha = alpha
        self.num_heads = num_heads
        self.decay_mat = None

    def forward(self, q, k, v, attn_mask=None):
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_output_weights += attn_mask

        # attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        # attn_output_weights = F.dropout(attn_output_weights,
        #                                 p=self.dropout,
        #                                 training=self.training)
        if self.alpha != 1.0 and self.decay_mat is None:
            self.decay_mat = torch.vander(torch.tensor([self.alpha] * self.num_heads), N=v.shape[1])
            self.decay_mat = self.decay_mat.to(v.device)
        if self.alpha != 1.0:
            bs = attn_output_weights.shape[0] // self.num_heads
            decay_mat = self.decay_mat.unsqueeze(0).repeat(bs, 1, 1)
            decay_mat = decay_mat.view(-1, *decay_mat.shape[2:])
            attn_output_weights = torch.log(decay_mat[:, None, :] + 1e-10) + attn_output_weights
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights,
                                        p=self.dropout,
                                        training=self.training)
        attn_output = torch.bmm(attn_output_weights, v)

        return attn_output


class DotProductAttentionStream(DotProductAttention):

    def __init__(self, dropout=0.0, alpha=1.0):
        super(DotProductAttentionStream, self).__init__(dropout, alpha)

        ############################
        # Cache for stream inference
        ############################
        self.k_weights_cache = None
        self.k_pos_weights_cache = None
        self.QK_exp_cache = None
        self.recache_cnt = 0
        self.recache_freq = 99999
        self.alpha = alpha

    def clear_cache(self):
        self.k_weights_cache = None
        self.k_pos_weights_cache = None
        self.QK_exp_cache = None
        self.recache_cnt = 0


    def stream_inference(self, q, k, v, k_pos, v_pos, attn_mask=None, cache_num=1, cache_id=0):
        '''
        if self.k_weights_cache is not None:
            k_weights_new = torch.bmm(q, k[:, [-1]].transpose(1, 2))
            k_weights = torch.cat((self.k_weights_cache[:, :, 1:], k_weights_new), dim=-1)
            self.k_weights_cache = k_weights
            # k_pos_weights = self.k_pos_weights_cache
        else:
            k_weights = torch.bmm(q, k.transpose(1, 2))
            self.k_weights_cache = k_weights
            # k_pos_weights = torch.bmm(q, k_pos.transpose(1, 2))
            # self.k_pos_weights_cache = k_pos_weights
        # attn_output_weights = k_weights + k_pos_weights
        attn_output_weights = k_weights

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights,
                                        p=self.dropout,
                                        training=self.training)
        # attn_output = torch.bmm(attn_output_weights, (v + v_pos))
        attn_output = torch.bmm(attn_output_weights, v)
        return attn_output
        '''
        B = 1
        num = 1
        q = q.repeat(B, 1, 1)
        k = k.repeat(B, num, 1)
        v = v.repeat(B, num, 1)
        attn_mask = attn_mask.repeat(B, 1, num)
        if self.alpha <= 1.0:
            if self.QK_exp_cache is None:
                QK = torch.bmm(q, k.transpose(1, 2))  # q~(H, Tq, C), k~(H, Tk, C) => O(H*Tq*Tk*C)
                self.QK_max = QK.max(dim=-1, keepdim=True).values
                QK_exp = torch.exp(QK - self.QK_max + attn_mask)  # => O(H*Tq*Tk)
                self.QK_exp_cache = [QK_exp for _ in range(cache_num)]
                self.QKV_exp_cache = [torch.bmm(QK_exp, v) for _ in range(cache_num)]       # => O(H*Tq*Tk*C)
                self.Z = [QK_exp.sum(dim=-1, keepdim=True) for _ in range(cache_num)]       # => O(H*Tq*Tk)
                output = self.QKV_exp_cache[cache_id] / self.Z[cache_id]
                self.recache_cnt += 1
                return output
            else:
                QK_push = torch.bmm(q, k[:, -num:, :].transpose(1, 2)) # => O(H*Tq*C)
                QK_exp_push = torch.exp(QK_push - self.QK_max)        # => O(H*Tq*Tk)
                sum_push = QK_exp_push.sum(-1, keepdim=True)
                QKV_exp_push = torch.bmm(QK_exp_push, v[:, -num:, :])  # => O(H*Tq*Tk)
                self.QKV_exp_cache[cache_id] = self.alpha * self.QKV_exp_cache[cache_id] + QKV_exp_push # => O(H*Tq*Tk)
                self.Z[cache_id] = self.alpha * self.Z[cache_id] + sum_push

                output = self.QKV_exp_cache[cache_id] / self.Z[cache_id]
                self.recache_cnt += 1
                return output

        if self.QK_exp_cache is None:
            QK = torch.bmm(q, k.transpose(1, 2))  # q~(H, Tq, C), k~(H, Tk, C) => O(H*Tq*Tk*C)
            self.QK_max = QK.max(dim=-1, keepdim=True).values
            QK_exp = torch.exp(QK - self.QK_max + attn_mask)  # => O(H*Tq*Tk)
            self.QK_exp_cache = [QK_exp for _ in range(cache_num)]
            self.QKV_exp_cache = [torch.bmm(QK_exp, v) for _ in range(cache_num)]       # => O(H*Tq*Tk*C)

            self.Z = [QK_exp.sum(dim=-1, keepdim=True) for _ in range(cache_num)]       # => O(H*Tq*Tk)
            self.QKV_exp_pop = [torch.bmm(self.QK_exp_cache[cache_i][:, :, :num], v[:, :num, :]) for cache_i in range(cache_num)]
            self.sum_pop = [self.QK_exp_cache[cache_i][:, :, :num].sum(-1, keepdim=True) for cache_i in range(cache_num)]

            output = self.QKV_exp_cache[cache_id] / self.Z[cache_id]
            self.recache_cnt += 1
            return output
        elif (self.recache_cnt // cache_num) % self.recache_freq == 0:
            QK = torch.bmm(q, k.transpose(1, 2))  # q~(H, Tq, C), k~(H, Tk, C) => O(H*Tq*Tk*C)
            self.QK_max = QK.max(dim=-1, keepdim=True).values
            QK_exp = torch.exp(QK - self.QK_max + attn_mask)  # => O(H*Tq*Tk)
            self.QK_exp_cache[cache_id] = QK_exp
            self.QKV_exp_cache[cache_id] = torch.bmm(QK_exp, v)         # => O(H*Tq*Tk*C)

            self.Z[cache_id] = QK_exp.sum(dim=-1, keepdim=True)         # => O(H*Tq*Tk)

            self.QKV_exp_pop[cache_id] = torch.bmm(self.QK_exp_cache[cache_id][:, :, :num], v[:, :num, :])
            self.sum_pop[cache_id] = self.QK_exp_cache[cache_id][:, :, :num].sum(-1, keepdim=True)
            output = self.QKV_exp_cache[cache_id] / self.Z[cache_id]
            self.recache_cnt += 1
            return output
        else:
            QK_push = torch.bmm(q, k[:, -num:, :].transpose(1, 2)) # => O(H*Tq*C)
            QK_exp_push = torch.exp(QK_push - self.QK_max)        # => O(H*Tq*Tk)
            sum_push = QK_exp_push.sum(-1, keepdim=True)
            QKV_exp_push = torch.bmm(QK_exp_push, v[:, -num:, :])  # => O(H*Tq*Tk)

            # add with no decay
            self.QKV_exp_cache[cache_id] = self.QKV_exp_cache[cache_id] + QKV_exp_push # => O(H*Tq*Tk)

            # add with no decay
            self.Z[cache_id] = self.Z[cache_id] + sum_push  # => O(H*Tq*Tk)

            self.QK_exp_cache[cache_id] = torch.cat((self.QK_exp_cache[cache_id][:, :, num:], QK_exp_push), dim=-1)
            self.QKV_exp_pop[cache_id] = torch.bmm(self.QK_exp_cache[cache_id][:, :, :num], v[:, :num, :])
            self.sum_pop[cache_id] = self.QK_exp_cache[cache_id][:, :, :num].sum(-1, keepdim=True)
            output = self.QKV_exp_cache[cache_id] / self.Z[cache_id]
            self.recache_cnt += 1
            return output


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, kdim=None, vdim=None,
                 attention_type='dotproduct', decay_alpha=1.0):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        if self._qkv_same_embed_dim:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        else:
            raise RuntimeError('Do not support q, k, v have different dimensions')

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

        if attention_type == 'dotproduct':
            self.attention = DotProductAttention(dropout, decay_alpha, num_heads)
        elif attention_type == 'linear':
            self.attention = LinearAttention(dropout)
        else:
            raise RuntimeError('attention_type should be [dotproduct | linear]')

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, \
            'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5

        _b = self.in_proj_bias
        _start = None
        _end = embed_dim
        _w = self.in_proj_weight[:_end, :]
        if _b is not None:
            _b = _b[:_end]
        q = F.linear(q, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = F.linear(k, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v = F.linear(v, _w, _b)

        q = q * scaling

        q = q.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(bsz, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, *attn_mask.shape[2:])

        if key_padding_mask is not None:
            if key_padding_mask.ndim == 4:
                key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1, 1, 1)
                key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1, 1)
                key_padding_mask = key_padding_mask.reshape(*key_padding_mask.shape[:3], -1)
            else:
                key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1)
                key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            key_padding_mask = key_padding_mask.reshape(-1, *key_padding_mask.shape[2:])

        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask + key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        attn_output = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tsz, bsz,
                                                                    self.embed_dim)
        return self.out_proj(attn_output), None


class MultiheadAttentionStream(MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, kdim=None, vdim=None,
                 attention_type='dotproduct', decay_alpha=1.0):
        super(MultiheadAttentionStream, self).__init__(embed_dim, num_heads,
                                                       dropout, bias, kdim,
                                                       vdim, attention_type,
                                                       decay_alpha)

        if attention_type == 'dotproduct':
            self.attention = DotProductAttentionStream(dropout, decay_alpha)
        elif attention_type == 'linear':
            self.attention = LinearAttentionStream(dropout)
        else:
            raise RuntimeError('attention_type should be [dotproduct | linear]')

        ############################
        # Cache for stream inference
        ############################
        self.q_cache = None
        self.k_cache = None
        self.v_cache = None
        self.k_pos_cache = None
        self.v_pos_cache = None

    def clear_cache(self):
        self.q_cache = None
        self.k_cache = None
        self.v_cache = None
        self.k_pos_cache = None
        self.v_pos_cache = None
        self.attention.clear_cache()

    def stream_inference(self, q, k, v, pos, attn_mask=None, key_padding_mask=None,
                         cache_num=1, cache_id=0):
        tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, \
            'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5

        if self.q_cache is not None:
            q = self.q_cache
        else:
            _b = self.in_proj_bias
            _start = None
            _end = embed_dim
            _w = self.in_proj_weight[:_end, :]
            if _b is not None:
                _b = _b[:_end]
            q = F.linear(q, _w, _b)
            self.q_cache = q

        assert (self.k_cache is None) == (self.k_pos_cache is None)
        if self.k_cache is not None:
            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k_new = F.linear(k[[-1]], _w, None)
            k = torch.cat((self.k_cache[1:], k_new))
            self.k_cache = k
            k_pos = self.k_pos_cache
        else:
            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(k, _w, None)
            self.k_cache = k
            k_pos = F.linear(pos, _w, _b)
            self.k_pos_cache = k_pos

        assert (self.v_cache is None) == (self.v_pos_cache is None)
        if self.v_cache is not None:
            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v_new = F.linear(v[[-1]], _w, None)
            v = torch.cat((self.v_cache[1:], v_new))
            self.v_cache = v
            v_pos = self.v_pos_cache
        else:
            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(v, _w, None)
            self.v_cache = v
            v_pos = F.linear(pos, _w, _b)
            self.v_pos_cache = v_pos

        q = q * scaling

        q = q.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k_pos = k_pos.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v_pos = v_pos.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(bsz, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, *attn_mask.shape[2:])

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1)
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            key_padding_mask = key_padding_mask.reshape(-1, *key_padding_mask.shape[2:])

        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask + key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        attn_output = self.attention.stream_inference(q, k, v, k_pos, v_pos, mask,
                                                      cache_num=cache_num, cache_id=cache_id)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tsz, bsz,
                                                                    self.embed_dim)
        return self.out_proj(attn_output), None


class LinearAttention(nn.Module):

    def __init__(self, dropout=0.0, nonlinear=F.elu,
                 eps=1e-6):
        super(LinearAttention, self).__init__()

        self.dropout = dropout
        self.nonlinear = nonlinear
        self.eps = eps

    def forward(self, q, k, v, attn_mask=None):
        kk = self.nonlinear(k) + 1
        qq = self.nonlinear(q) + 1
        vv = self.nonlinear(v) + 1

        kv = torch.bmm(kk.transpose(1, 2), vv)
        
        ks = torch.sum(kk, dim=1, keepdim=True)  # (B 1 C)
        Z = 1 / (torch.bmm(qq, ks.transpose(1, 2)) + self.eps)
        attn_output = torch.bmm(qq, kv.transpose(1, 2)) * Z

        # if attn_mask is not None:
        #     print(attn_mask.shape)
        #     raise RuntimeError(("LinearAttention does not support arbitrary "
        #                         "attention masks"))

        return attn_output

class LinearAttentionStream(LinearAttention):

    def __init__(self, dropout=0.0):
        super(LinearAttentionStream, self).__init__(dropout)

        ############################
        # Cache for stream inference
        ############################
        self.k_weights_cache = None
        self.k_pos_weights_cache = None

    def stream_inference(self, q, k, v, k_pos, v_pos, attn_mask=None):
        raise NotImplementedError

