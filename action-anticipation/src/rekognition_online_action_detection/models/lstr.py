# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from mamba_ssm import Mamba
from mamba_ssm.modules.mamba_simple import Block
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from . import transformer as tr

from .models import META_ARCHITECTURES as registry
from .normalized_linear import NormalizedLinear
from .feature_head import build_feature_head
from ..utils.ek_utils import (action_to_noun_map, action_to_verb_map)

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class LSTR(nn.Module):

    def __init__(self, cfg):
        super(LSTR, self).__init__()

        # Build long feature heads
        self.cfg = cfg
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.long_enabled = self.long_memory_num_samples > 0
        if self.long_enabled:
            self.feature_head_long = build_feature_head(cfg)

        # Build work feature head
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.work_enabled = self.work_memory_num_samples > 0
        if self.work_enabled:
            self.feature_head_work = build_feature_head(cfg)

        self.d_model = self.feature_head_work.d_model
        self.num_heads = cfg.MODEL.LSTR.NUM_HEADS
        self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.LSTR.DROPOUT
        self.activation = cfg.MODEL.LSTR.ACTIVATION
        self.num_classes = cfg.DATA.NUM_CLASSES

        self.use_v_n = cfg.MODEL.LSTR.V_N_CLASSIFIER

        self.encoder_attention_type = cfg.MODEL.LSTR.ENC_ATTENTION_TYPE
        self.decay_alpha = cfg.MODEL.LSTR.ENC_ATTENTION_DECAY

        self.anticipation_num_samples = cfg.MODEL.LSTR.ANTICIPATION_NUM_SAMPLES

        # Build position encoding
        self.pos_encoding = tr.PositionalEncoding(self.d_model, self.dropout)

        self.long_memory_use_pe = cfg.MODEL.LSTR.LONG_MEMORY_USE_PE
        self.work_memory_use_pe = cfg.MODEL.LSTR.WORK_MEMORY_USE_PE
        self.include_work = cfg.MODEL.LSTR.LONG_MEMORY_INCLUDE_WORK
        self.include_work2 = cfg.MODEL.LSTR.LONG_MEMORY_INCLUDE_WORK2
        if self.include_work and self.include_work2:
            raise ValueError('cfg.MODEL.LSTR.LONG_MEMORY_INCLUDE_WORK and'
                             'cfg.MODEL.LSTR.LONG_MEMORY_INCLUDE_WORK2 should'
                             'not be both True')

        # Build LSTR encoder
        if self.long_enabled and False:
            self.enc_queries = nn.ModuleList()
            self.enc_modules = nn.ModuleList()
            for i, param in enumerate(cfg.MODEL.LSTR.ENC_MODULE):
                if param[0] != -1:
                    self.enc_queries.append(nn.Embedding(param[0], self.d_model))
                    enc_layer = tr.TransformerDecoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation,
                        attention_type=self.encoder_attention_type if i == 0 else 'dotproduct',
                        decay_alpha=self.decay_alpha if i == 0 else 1.0)
                    self.enc_modules.append(tr.TransformerDecoder(
                        enc_layer, param[1], tr.layer_norm(self.d_model, param[2])))
                else:
                    self.enc_queries.append(None)
                    enc_layer = tr.TransformerEncoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation,
                        attention_type=self.encoder_attention_type)
                    self.enc_modules.append(tr.TransformerEncoder(
                        enc_layer, param[1], tr.layer_norm(self.d_model, param[2])))
        else:
            self.register_parameter('enc_queries', None)
            self.register_parameter('enc_modules', None)

        # Build LSTR decoder
        if self.long_enabled and False:
            param = cfg.MODEL.LSTR.DEC_MODULE
            dec_layer = tr.TransformerDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation)
            self.dec_modules = tr.TransformerDecoder(
                dec_layer, param[1], tr.layer_norm(self.d_model, param[2]))
        else:
            if cfg.MODEL.LSTR.MAMBA_LAYER == 0:
                param = cfg.MODEL.LSTR.DEC_MODULE
                dec_layer = tr.TransformerEncoderLayer(
                    self.d_model, self.num_heads, self.dim_feedforward,
                    self.dropout, self.activation)
                self.dec_modules = tr.TransformerEncoder(
                    dec_layer, param[1], tr.layer_norm(self.d_model, param[2]))
                self.is_mamba = False
            else:
                print(f"Init Mamba Encoder!", flush=True)
                self.mamba_modules = nn.ModuleList()
                for _ in range(cfg.MODEL.LSTR.MAMBA_LAYER):
                    self.mamba_modules.append(create_block(self.d_model))
                self.norm_f = RMSNorm(
                    self.d_model, eps=1e-5
                )
                self.is_mamba = True

        # Build decode token (for anticipation)
        if self.anticipation_num_samples > 0:
            self.dec_query = nn.Embedding(self.anticipation_num_samples, self.d_model)
        else:
            self.register_parameter('dec_query', None)

        # Build classifier
        if cfg.MODEL.LSTR.DROPOUT_CLS > 0:
            self.dropout_cls = nn.Dropout(cfg.MODEL.LSTR.DROPOUT_CLS)
            if self.use_v_n:
                self.dropout_cls_v = nn.Dropout(cfg.MODEL.LSTR.DROPOUT_CLS)
                self.dropout_cls_n = nn.Dropout(cfg.MODEL.LSTR.DROPOUT_CLS)
        else:
            self.dropout_cls = None
        if cfg.MODEL.LSTR.FC_NORM:
            self.classifier = NormalizedLinear(self.d_model, self.num_classes)
        else:
            self.classifier = nn.Linear(self.d_model, self.num_classes)

        if self.use_v_n:
            a_to_v = action_to_verb_map(cfg.DATA.EK_EXT_PATH,
                                        action_offset=True,
                                        verb_offset=True)
            a_to_n = action_to_noun_map(cfg.DATA.EK_EXT_PATH,
                                        action_offset=True,
                                        noun_offset=True)
            num_verbs = max(set(a_to_v.values())) + 1
            num_nouns = max(set(a_to_n.values())) + 1
            print('Number of verbs (inc. 0):', num_verbs)
            print('Number of nouns (inc. 0):', num_nouns)
            self.classifier_verb = nn.Linear(self.d_model, num_verbs)
            self.classifier_noun = nn.Linear(self.d_model, num_nouns)

        self.pred_future = 'PRED_FUTURE' in list(zip(*cfg.MODEL.CRITERIONS))[0]

        total_param = sum([p.numel() for p in self.parameters()])
        print(f"Total parameters: {total_param / 10**6:.2f} M", flush=True)

    def forward(self, visual_inputs, motion_inputs, object_inputs, memory_key_padding_mask=None):
        if self.long_enabled:
            # Compute long memories
            if self.long_memory_use_pe:
                long_memories = self.pos_encoding(self.feature_head_long(
                    visual_inputs[:, :self.long_memory_num_samples if not self.include_work2 else None],
                    motion_inputs[:, :self.long_memory_num_samples if not self.include_work2 else None],
                    object_inputs[:, :self.long_memory_num_samples if not self.include_work2 else None],
                ).transpose(0, 1))
                long_memories = long_memories.transpose(0, 1) ## transpose back
            else:
                long_memories = self.feature_head_long(
                    visual_inputs[:, :self.long_memory_num_samples if not self.include_work2 else None],
                    motion_inputs[:, :self.long_memory_num_samples if not self.include_work2 else None],
                    object_inputs[:, :self.long_memory_num_samples if not self.include_work2 else None],
                )
            batch_size = long_memories.shape[0]
            # if long_memories.ndim == 5:
            #     long_memories = self.pos_encoding_3d_spatial(long_memories)
            long_memories = long_memories.view(batch_size, -1, self.d_model).transpose(0, 1)
            # print(long_memories.shape)   # T(*H*W), B, C

            if False and len(self.enc_modules) > 0:
                enc_queries = [
                    enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                    if enc_query is not None else None
                    for enc_query in self.enc_queries
                ]

                if self.include_work2:
                    memory_key_padding_mask = F.pad(memory_key_padding_mask,
                                                    (0, self.work_memory_num_samples),
                                                    'constant', 0)
                # Encode long memories
                if enc_queries[0] is not None:
                    long_memories = self.enc_modules[0](enc_queries[0], long_memories,
                                                        memory_key_padding_mask=memory_key_padding_mask)
                else:
                    long_memories = self.enc_modules[0](long_memories)
                for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                    if enc_query is not None:
                        long_memories = enc_module(enc_query, long_memories)
                    else:
                        long_memories = enc_module(long_memories)

        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        if self.work_enabled:
            # Compute work memories
            if visual_inputs.ndim == 5:
                visual_inputs_avg = visual_inputs[:, self.long_memory_num_samples:].mean((2, 3))
                motion_inputs_avg = motion_inputs[:, self.long_memory_num_samples:].mean((2, 3))
            else:
                visual_inputs_avg = visual_inputs[:, self.long_memory_num_samples:]
                motion_inputs_avg = motion_inputs[:, self.long_memory_num_samples:]
            work_memories_no_pe = self.feature_head_work(
                visual_inputs_avg,
                motion_inputs_avg,
                object_inputs[:, self.long_memory_num_samples:],
            ).transpose(0, 1)
            if self.work_memory_use_pe:
                work_memories = self.pos_encoding(work_memories_no_pe,
                    padding=self.long_memory_num_samples if self.long_memory_use_pe else 0)
            else:
                work_memories = work_memories_no_pe
            if self.dec_query is not None:
                anticipate_memories = self.pos_encoding(
                    self.dec_query.weight.unsqueeze(1).repeat(1, work_memories.shape[1], 1),
                    padding=(self.long_memory_num_samples + self.work_memory_num_samples
                             if self.long_memory_use_pe else self.work_memory_num_samples))
                work_memories = torch.cat((work_memories, anticipate_memories), dim=0)

            # Build mask
            mask = tr.generate_square_subsequent_mask(
                work_memories.shape[0])
            mask = mask.to(work_memories.device)
            # Compute output
            if self.long_enabled:
                if self.include_work:
                    extended_memory = torch.cat((memory, work_memories), dim=0)
                    extended_mask = F.pad(mask,
                                          (memory.shape[0], 0),
                                          'constant', 0)
                    extended_mask = extended_mask.to(extended_memory.device)
                    output = self.dec_modules(
                        work_memories,
                        memory=extended_memory,
                        tgt_mask=mask,
                        memory_mask=extended_mask,
                    )
                else:
                    output = self.dec_modules(
                        work_memories,
                        memory=memory,
                        tgt_mask=mask,
                    )
            else:
                if self.is_mamba:
                    if self.long_enabled:
                        work_memories = torch.cat([long_memories, work_memories], dim=0)
                    output = work_memories.transpose(0, 1)
                    for layer in self.mamba_modules:
                        output, residual = layer(
                            output
                        )
                    output = rms_norm_fn(
                        output,
                        self.norm_f.weight,
                        self.norm_f.bias,
                        eps=self.norm_f.eps,
                        residual=residual,
                        prenorm=False,
                        residual_in_fp32=True,
                    )
                    if self.long_enabled:
                        output = output[:, self.long_memory_num_samples:, :]
                    output = output.transpose(0, 1)
                else::
                    output = self.dec_modules(
                        work_memories,
                        src_mask=mask,
                    )
                    
        full_orig_feats = self.feature_head_work(
                visual_inputs_avg,
                motion_inputs_avg,
                object_inputs[:, self.long_memory_num_samples:]
        ).transpose(0, 1)
        all_outputs_decoded = output

        # Compute classification score
        if self.dropout_cls is not None:
            output_a = self.dropout_cls(output)
        else:
            output_a = output
        score = self.classifier(output_a)
        score = score.transpose(0, 1)

        if self.use_v_n:
            output_v = self.dropout_cls_v(output)
            output_n = self.dropout_cls_n(output)
            score_verb = self.classifier_verb(output_v).transpose(0, 1)
            score_noun = self.classifier_noun(output_n).transpose(0, 1)
            score = (score, score_verb, score_noun)
        if self.pred_future:
            return (score,
                    all_outputs_decoded[1: self.work_memory_num_samples + 1, ...],
                    full_orig_feats)
        else:
            return score


@registry.register('LSTR')
class LSTRStream(LSTR):

    def __init__(self, cfg):
        super(LSTRStream, self).__init__(cfg)

        ############################
        # Cache for stream inference
        ############################
        self.long_memories_cache = None
        self.compressed_long_memories_cache = None

    def stream_inference(self,
                         long_visual_inputs,
                         long_motion_inputs,
                         long_object_inputs,
                         work_visual_inputs,
                         work_motion_inputs,
                         work_object_inputs,
                         memory_key_padding_mask=None,
                         cache_num=1,
                         cache_id=0):
        assert self.long_enabled, 'Long-term memory cannot be empty for stream inference'
        assert len(self.enc_modules) > 0, 'LSTR encoder cannot be disabled for stream inference'

        if ((long_visual_inputs is not None) and
            (long_motion_inputs is not None) and
            (long_object_inputs is not None)):
            # Compute long memories
            long_memories = self.feature_head_long(
                long_visual_inputs if not self.include_work2 else torch.cat((long_visual_inputs, work_visual_inputs), dim=1),
                long_motion_inputs if not self.include_work2 else torch.cat((long_motion_inputs, work_motion_inputs), dim=1),
                long_object_inputs if not self.include_work2 else torch.cat((long_object_inputs, work_object_inputs), dim=1),
            ).transpose(0, 1)

            if self.long_memories_cache is None:
                self.long_memories_cache = [long_memories for _ in range(cache_num)]
            else:
                self.long_memories_cache[cache_id] = torch.cat((
                    self.long_memories_cache[cache_id][1:], long_memories
                ))

            long_memories = self.long_memories_cache[cache_id]
            pos = self.pos_encoding.pe[:self.long_memory_num_samples, :]

            enc_queries = [
                enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                if enc_query is not None else None
                for enc_query in self.enc_queries
            ]

            if self.include_work2:
                memory_key_padding_mask = F.pad(memory_key_padding_mask,
                                                (0, self.work_memory_num_samples),
                                                'constant', 0)
            # Encode long memories
            long_memories = self.enc_modules[0].stream_inference(enc_queries[0], long_memories, pos,
                                                                 memory_key_padding_mask=memory_key_padding_mask,
                                                                 cache_num=cache_num, cache_id=cache_id)
            self.compressed_long_memories_cache  = long_memories
            for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                if enc_query is not None:
                    long_memories = enc_module(enc_query, long_memories)
                else:
                    long_memories = enc_module(long_memories)
        else:
            long_memories = self.compressed_long_memories_cache

            enc_queries = [
                enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                if enc_query is not None else None
                for enc_query in self.enc_queries
            ]

            # Encode long memories
            for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                if enc_query is not None:
                    long_memories = enc_module(enc_query, long_memories)
                else:
                    long_memories = enc_module(long_memories)

        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        if self.work_enabled:
            # Compute work memories
            work_memories = self.pos_encoding(self.feature_head_work(
                work_visual_inputs,
                work_motion_inputs,
                work_object_inputs,
            ).transpose(0, 1),
            padding=self.long_memory_num_samples if self.long_memory_use_pe else 0)
            if self.dec_query is not None:
                anticipate_memories = self.pos_encoding(
                    self.dec_query.weight.unsqueeze(1).repeat(1, work_memories.shape[1], 1),
                    padding=(self.long_memory_num_samples + self.work_memory_num_samples
                             if self.long_memory_use_pe else  self.work_memory_num_samples))
                work_memories = torch.cat((work_memories, anticipate_memories))

            # Build mask
            mask = tr.generate_square_subsequent_mask(
                work_memories.shape[0])
            mask = mask.to(work_memories.device)

            # Compute output
            if self.long_enabled:
                if self.include_work:
                    extended_memory = torch.cat((memory, work_memories), dim=0)
                    extended_mask = F.pad(mask,
                                          (memory.shape[0], 0),
                                          'constant', 0)
                    extended_mask = extended_mask.to(extended_memory.device)
                    output = self.dec_modules(
                        work_memories,
                        memory=extended_memory,
                        tgt_mask=mask,
                        memory_mask=extended_mask,
                    )
                else:
                    output = self.dec_modules(
                        work_memories,
                        memory=memory,
                        tgt_mask=mask,
                    )
            else:
                output = self.dec_modules(
                    work_memories,
                    src_mask=mask,
                )

        # Compute classification score
        score = self.classifier(output)

        return score.transpose(0, 1)

    def clear_cache(self):
        self.long_memories_cache = None
        self.compressed_long_memories_cache = None
        self.enc_modules[0].clear_cache()
