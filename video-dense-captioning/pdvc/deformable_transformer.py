# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from misc.detr_utils.misc import  inverse_sigmoid
from pdvc.ops.modules import MSDeformAttn

try:
    # from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.modules.mamba2 import Mamba
except:
    print('Please install mamba_ssm before using Mamba encoder')

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 encoder_type='deformable',
                ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.no_encoder = (num_encoder_layers == 0)
        self.num_feature_levels = num_feature_levels

        if encoder_type == 'deformable':
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
        else:
            encoder_layer = MambaEncoderLayer(d_model, dim_feedforward,
                                                dropout, activation,
                                                num_feature_levels, nhead, enc_n_points)
                                                
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.pos_trans = nn.Linear(d_model, d_model * 2)
        self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        self.reference_points = nn.Linear(d_model, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)


    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 256
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 2
        proposals = proposals.sigmoid() * scale
        # N, L, 2, 256
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 2, 128, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def get_valid_ratio(self, mask):
        valid_ratio_L = torch.sum(~mask, 1).float() / mask.shape[1]
        return valid_ratio_L

    def prepare_encoder_inputs(self, srcs, masks, pos_embeds):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        temporal_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            """
            lvl: (bs, )
            src: (bs, c, L )
            mask: (bs, L)
            pos_embed: (bs, d_m, L)
            """
            bs, c, L = src.shape
            temporal_shapes.append(L)
            src = src.transpose(1, 2)  # （bs, L, c）
            pos_embed = pos_embed.transpose(1, 2)  # #（bs, L, d_m）
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # (lvl_num, bs, wh, c)
        mask_flatten = torch.cat(mask_flatten, 1)  # (lvl_num, bs, wh)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # (lvl_num, bs, wh, d_m)
        temporal_shapes = torch.as_tensor(temporal_shapes, dtype=torch.long, device=src_flatten.device)  # (lvl_num, 2)
        level_start_index = torch.cat((temporal_shapes.new_zeros((1,)), temporal_shapes.cumsum(0)[
                                                                       :-1]))  # prod: [w0h0, w0h0+w1h1, w0h0+w1h1+w2h2, ...]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks],
                                   1)  # (bs, lvl_num, 2), where 2 means (h_rate, and w_rate)， all values <= 1

        return src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten

    def forward_encoder(self, src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                        mask_flatten):
        # encoder
        if self.no_encoder:
            memory = src_flatten
        else:
            memory = self.encoder(src_flatten, temporal_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                                  mask_flatten)

        return memory

    def prepare_decoder_input_query(self, memory, query_embed):
        bs, _, _ = memory.shape
        query_embed, tgt = torch.chunk(query_embed, 2, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_embed).sigmoid()
        init_reference_out = reference_points  # (bs, object_query, 2)
        return init_reference_out, tgt, reference_points, query_embed

    def prepare_decoder_input_proposal(self, gt_reference_points):
        topk_coords_unact = inverse_sigmoid(gt_reference_points)
        reference_points = gt_reference_points
        init_reference_out = reference_points
        pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        query_embed, tgt = torch.chunk(pos_trans_out, 2, dim=2)
        return init_reference_out, tgt, reference_points, query_embed

    def forward_decoder(self, *kargs):
        hs, inter_references_out = self.decoder(*kargs)
        return hs, inter_references_out


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, temporal_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, temporal_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(temporal_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (L_) in enumerate(temporal_shapes):
            ref = torch.linspace(0.5, L_ - 0.5, L_, dtype=torch.float32, device=device)
            ref = ref.reshape(-1)[None] / (valid_ratios[:, None, lvl] * L_)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        reference_points = reference_points[:,:,:,None]
        return reference_points

    def forward(self, src, temporal_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(temporal_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, temporal_shapes, level_start_index, padding_mask)

        return output



class MambaEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        
        # self attention
        self.self_attn = Mamba(d_model, d_conv=4, bimamba_type="v2", use_fast_path=True, expand=1)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, temporal_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos))
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.forward_ffn(src)
        return src


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_temporal_shapes, level_start_index,
                src_padding_mask=None, query_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), key_padding_mask=~query_mask)[
            0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_temporal_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_head = None

    def forward(self, tgt, reference_points, src, src_temporal_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, query_padding_mask=None, disable_iterative_refine=False):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        bs = tgt.shape[0]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 2:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.stack([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 1
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None, :, None]
            output = layer(output, query_pos, reference_points_input, src, src_temporal_shapes, src_level_start_index,
                           src_padding_mask, query_padding_mask)

            # hack implementation for iterative bounding box refinement
            if disable_iterative_refine:
                reference_points = reference_points
            else:
                if (self.bbox_head is not None):
                    tmp = self.bbox_head[lid](output)
                    if reference_points.shape[-1] == 2:
                        new_reference_points = tmp + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    else:
                        assert reference_points.shape[-1] == 1
                        new_reference_points = tmp
                        new_reference_points[..., :1] = tmp[..., :1] + inverse_sigmoid(reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()
                else:
                    reference_points = reference_points

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.transformer_ff_dim,
        dropout=args.transformer_dropout_prob,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        encoder_type=args.encoder_type,     ### added
    )