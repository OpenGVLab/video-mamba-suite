# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of the code is from https://github.com/m-bain/frozen-in-time/blob/main/model/video_transformer.py
# Modified by Yue Zhao
# The original code is under MIT License

"""
Implementations of Video Transformers in PyTorch
A PyTorch implementation of space-time transformer as described in
'Frozen in Time: A Joint Image and Video Encoder for End-to-End Retrieval' - https://arxiv.org/abs/2104.00650
A PyTorch implementation of timesformer as described in
'Is Space-Time Attention All You Need for Video Understanding?' - https://arxiv.org/abs/2102.05095
Acknowledgments:
- This code builds on Ross Wightman's vision_transformer code in pytorch-image-models:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
- It is also inspired by lucidrains timesformer implementation:
https://github.com/lucidrains/TimeSformer-pytorch
Hacked together by Max Bain
"""

from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from typing import Callable, List, Optional
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum, nn

from flash_attn.modules.mha import MHA as FlashMHA
from flash_attn.modules.mha import FlashSelfAttention, SelfAttention
from flash_attn.modules.mlp import Mlp as FlashMlp

from mamba_ssm.modules.mamba_simple import Mamba



def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VideoPatchEmbed(nn.Module):
    """ Video to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 num_frames=8, ln_pre=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        # ln_pre is inserted to be compatible with CLIP-style model
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=not ln_pre)

    def forward(self, x):
        B, F, C, H, W = x.shape
        assert F <= self.num_frames
        x = x.view(-1, C, H, W)
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, F, W


class SpaceTimeBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, time_init='zeros',
                 attention_style='frozen-in-time', is_tanh_gating=False, use_flash_attn=False, use_light_mamba=False):
        super().__init__()



        self.norm1 = norm_layer(dim)
        self.use_flash_attn = use_flash_attn
        if not use_flash_attn:
            self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop)
        else:
            self.attn = FlashMHA(dim, num_heads, cross_attn=False, qkv_proj_bias=True,
                                 out_proj_bias=True, dropout=attn_drop, use_flash_attn=True)
        if use_light_mamba:
            from mamba_ssm.modules.mamba_new import Mamba as MambaNew
            self.time_mamba = MambaNew(dim, expand=1)
        else:
            self.time_mamba = Mamba(dim, d_conv=4, bimamba_type="v2", use_fast_path=True, expand=1)

        if is_tanh_gating:
            self.alpha_timeattn = nn.Parameter(torch.zeros([]))

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)

        self.attention_style = attention_style


    def forward(self, x, time_n, space_f, use_checkpoint=False):

        init_cls_token = x[:, :1]
        res_x = x
        x = x[:, 1:]
        
        if self.attention_style != "frozen-joint":
            x = rearrange(x, 'b (n t) d -> (b n) t d', n=time_n, t=space_f)
            time_output = self.time_mamba(self.norm3(x))
            if hasattr(self, "alpha_timeattn"):
                time_output = torch.tanh(self.alpha_timeattn) * time_output
            time_residual = x + time_output
            time_residual = rearrange(time_residual, '(b n) t d -> b (n t) d', n=time_n, t=space_f)
        else:
            time_output = self.time_mamba(self.norm3(x))
            if hasattr(self, "alpha_timeattn"):
                    time_output = torch.tanh(self.alpha_timeattn) * time_output
            time_residual = x + time_output
            

        cls_token = init_cls_token.repeat(1, space_f, 1)
        cls_token = rearrange(cls_token, 'b t d -> (b t) d',t=space_f).unsqueeze(1)
        xs = time_residual
        xs = rearrange(xs, 'b (n t) d -> (b t) n d', t=space_f)
        xs = torch.cat((cls_token, xs), dim=1)


        if not self.use_flash_attn:
            x_ = self.norm1(xs)
            space_output = self.attn(x_, x_, x_, need_weights=False)[0]
        else:
            space_output = self.attn(self.norm1(xs))
        
        cls_token = space_output[:, 0]
        cls_token = rearrange(cls_token, '(b t) d -> b t d', t=space_f)
        cls_token = cls_token.mean(1, keepdim=True)
        space_output = rearrange(space_output[:, 1:], '(b t) n d -> b (n t) d',t=space_f)

        if self.attention_style in ['frozen-in-time', 'frozen-joint']:
            x = res_x + torch.cat((cls_token, space_output), 1)
        elif self.attention_style == 'timesformer-div':
            x = torch.cat((init_cls_token, time_residual), 1) + torch.cat((cls_token, space_output), 1)
        else:
            raise NotImplementedError

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # exit()

        return x

class TimeMamba(nn.Module):
    """ Time Mamba 
    A PyTorch impl of : `Space-Time Transformer` from Frozen-in-time  - by Max Bain.
        https://arxiv.org/abs/2104.00650
    Based off:
     - ViT implementation from the timm library [https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py]
    lucidrains timesformer implementation [https://github.com/lucidrains/TimeSformer-pytorch].
    Notable differences:
     - allows for variable length input frames (<= num_frames)
     - allows for variable length input resolution  (<= (img_size, img_size)) [UNTESTED]
     - different attention block mechanism
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None,
                 num_frames=8, time_init='rand', attention_style='frozen-in-time', ln_pre=False,
                 act_layer=nn.GELU, is_tanh_gating=False, use_flash_attn=False, use_light_mamba=False, output_dim=512):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
            num_frames: (int) maximum number of frames expected as input
            time_init: (str) how to initialise the time attention layer, 'zeros' allows for the timesformer to start off
                        as ViT.
            attention_style: (str) how to attend to space and time.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        scale = embed_dim ** -0.5
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        print("######USING ATTENTION STYLE: ", attention_style)
        assert attention_style in ['frozen-in-time', 'timesformer-div', 'frozen-joint']
        if hybrid_backbone is not None:
            raise NotImplementedError('hybrid backbone not implemented')
        else:
            self.patch_embed = VideoPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=num_frames, ln_pre=ln_pre)
        num_patches = self.patch_embed.num_patches
        self.patches_per_frame = num_patches // num_frames

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patches_per_frame + 1,
                        embed_dim))  # remember to take pos_embed[1:] for tiling over time

        if ln_pre:
            self.ln_pre = nn.LayerNorm(embed_dim)
        else:
            self.ln_pre = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, time_init=time_init,
                attention_style=attention_style, act_layer=act_layer, is_tanh_gating=is_tanh_gating, 
                use_flash_attn=use_flash_attn, use_light_mamba=use_light_mamba)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        if output_dim is None:
            self.image_projection = None
        else:
            self.image_projection = nn.Parameter(scale * torch.randn(embed_dim, output_dim))

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        # if num_frames > 1, then we perform ViT inflation and initialise time attention to zero so not necessary.
        if num_frames == 1:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def freeze_spatial_weights(self):
        freeze_list = []
        for n, p in self.named_parameters():
            if 'temporal_embed' in n or 'timeattn' in n or 'norm3' in n:
                pass
            else:
                p.requires_grad = False
                freeze_list.append(n)
        print("Freeze the pretrained parts in vision model: {}".format(freeze_list))

    def freeze_temporal_weights(self):
        freeze_list = []
        for n, p in self.named_parameters():
            if 'temporal_embed' in n or 'timeattn' in n or 'norm3' in n:
                p.requires_grad = False
                freeze_list.append(n)
            else:
                pass
        print("Freeze the pretrained parts in vision model: {}".format(freeze_list))

    def forward_features(self, x, use_checkpoint=False, cls_at_last=True):
        # print(x.shape)
        B, curr_frames, channels, _, _ = x.shape
        x, T, W = self.patch_embed(x)
        # print(x.shape)
        # exit()

        BF = x.shape[0]
        cls_tokens = self.cls_token.expand(BF, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed

        ## reshape
        cls_tokens = x[:B, 0, :].unsqueeze(1)
        x = x[:,1:]
        x = rearrange(x, '(b t) n m -> b (n t) m',b=B,t=T)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.ln_pre is not None:
            x = self.ln_pre(x)
        x = self.pos_drop(x)
        n = self.patches_per_frame
        f = curr_frames
        for blk in self.blocks:
            if use_checkpoint:
                x = checkpoint.checkpoint(blk, x, n, f)
            else:
                x = blk(x, time_n=n, space_f=f)

        if cls_at_last:
            x = self.norm(x)[:, 0]
            x = self.pre_logits(x)
            return x
        else:
            return self.norm(x)

    def forward(self, x, use_checkpoint=False):
        # Note:  B C T H W => B T C H W
        # The default input order is different from the one in Frozen-in-Time
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = self.forward_features(x, use_checkpoint=use_checkpoint)
        
        if self.image_projection is not None:
            x = x @ self.image_projection
        
        return x


if __name__ == "__main__":
    import time
    num_frames = 5000
    hw = 96
    use_flash_attn = True
    # attn_mode = "frozen-in-time"
    attn_mode = "timesformer-div"
    model = TimeMamba(
        num_frames=num_frames,
        embed_dim=768,
        num_heads=12,
        img_size=hw,
        attention_style=attn_mode,
        use_flash_attn=use_flash_attn
    ).cuda()
    print("Model loaded")
    x = torch.randn(2, 3, num_frames, hw, hw).cuda()
    print("Data loaded")

    if use_flash_attn:
        model = model.half()
        x = x.half()

    record_i = 100

    for i in range(400):
        if i == record_i or True:
            fwd_begin = time.time()
            y = model(x, use_checkpoint=True)
            loss = y.mean()
            fwd_end = time.time()
            fwd_time = fwd_end - fwd_begin
            bwd_begin = time.time()
            loss.backward()
            bwd_end = time.time()
            bwd_time = bwd_end - bwd_begin
            max_mem = torch.cuda.max_memory_allocated()
            # print(y.shape,)
            print(fwd_time, bwd_time, max_mem/ 1024/ 1024,"MB",flush=True)
        else:
            y = model(x)
            loss = y.mean()
            loss.backward()

