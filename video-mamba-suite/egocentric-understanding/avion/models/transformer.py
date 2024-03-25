# adapted from: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py

from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.layers import trunc_normal_
from timm.models.layers import to_2tuple
from timm.models.layers import DropPath

from flash_attn.modules.mha import MHA as FlashMHA
from flash_attn.modules.mlp import Mlp as FlashMlp


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            use_flash_attn: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.use_flash_attn = use_flash_attn
        if not use_flash_attn:
            self.attn = nn.MultiheadAttention(d_model, n_head, dropout=attn_drop)
        else:
            self.attn = FlashMHA(d_model, n_head, cross_attn=False, qkv_proj_bias=True,
                                 out_proj_bias=True, dropout=attn_drop, use_flash_attn=True)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        if not use_flash_attn:
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("drop1", nn.Dropout(drop)),
                ("c_proj", nn.Linear(mlp_width, d_model)),
                ("drop2", nn.Dropout(drop)),
            ]))
        else:
            self.mlp = FlashMlp(d_model, hidden_features=mlp_width, activation=act_layer())
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.use_flash_attn:
            x = x + self.drop_path(self.ls_1(self.attention(self.ln_1(x), attn_mask=attn_mask)))
        else:
            x = x + self.drop_path(self.ls_1(self.attn(self.ln_1(x))))
        x = x + self.drop_path(self.ls_2(self.mlp(self.ln_2(x))))
        return x



class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            use_flash_attn: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                act_layer=act_layer, norm_layer=norm_layer,
                use_flash_attn=use_flash_attn)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            num_frames: int = 1,
            ls_init_value: float = None,
            global_average_pool: bool = False,
            output_dim: int = None,
            patch_dropout: float = 0.,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            ln_pre: bool = True,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
            use_fast_conv1: bool = False,
            use_flash_attn: bool = False,
    ):
        super().__init__()
        self.use_fast_conv1 = use_fast_conv1
        self.use_flash_attn = use_flash_attn
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.width = width
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.patches_per_frame = self.grid_size[0] * self.grid_size[1]
        self.output_dim = output_dim
        if use_fast_conv1:
            self.conv1 = nn.Linear(in_features=3 * patch_size ** 2, out_features=width, bias=not ln_pre)
        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=not ln_pre)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        assert num_frames >= 1
        self.num_frames = num_frames
        if num_frames > 1:
            self.temporal_embedding = nn.Parameter(torch.zeros(num_frames, width))

        assert not (patch_dropout > 0. and drop_rate > 0.)
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0. else nn.Identity()

        if ln_pre:
            self.ln_pre = norm_layer(width)
        else:
            self.ln_pre = nn.Identity()

        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_flash_attn=use_flash_attn,
        )

        self.global_average_pool = global_average_pool
        self.ln_post = norm_layer(width)
        if output_dim is None:
            self.image_projection = None
        else:
            self.image_projection = nn.Parameter(scale * torch.randn(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        # TODO: compare the two styles
        # Mimicking timm's initialization
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)

        for block in self.transformer.resblocks:
            for n, p in block.named_parameters():
                if 'weight' in n:
                    trunc_normal_(p, std=0.02)
                elif 'bias' in n:
                    nn.init.zeros_(p)
                else:
                    raise NotImplementedError('Unknown parameters named {}'.format(n)) 
        if self.image_projection is not None:
            nn.init.normal_(self.image_projection, std=self.width ** -0.5)

        # Same init as TextTransformer
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        # nn.init.normal_(self.positional_embedding, std=0.01)

        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # if self.image_projection is not None:
        #     nn.init.normal_(self.image_projection, std=self.output_dim ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor):
        if self.use_fast_conv1:
            if self.num_frames == 1:
                x = rearrange(x, "b c (hh sh) (ww sw) -> b (hh ww) (c sh sw)", sh=self.patch_size[0], sw=self.patch_size[1])
                x = self.conv1(x)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)
            else:
                x = rearrange(x, "b c t (hh sh) (ww sw) -> b (t hh ww) (c sh sw)", sh=self.patch_size[0], sw=self.patch_size[1])
                x = self.conv1(x)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                cls_embed = self.positional_embedding[0, :].unsqueeze(0)
                tile_pos_embed = self.positional_embedding[1:, :].repeat(self.num_frames, 1)
                tile_temporal_embed = self.temporal_embedding.repeat_interleave(self.patches_per_frame, 0)
                total_pos_embed = tile_pos_embed + tile_temporal_embed
                total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=0)
                x = x + total_pos_embed.to(x.dtype).unsqueeze(0)
        else:
            if self.num_frames == 1:
                x = self.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)
            else:
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # B, C, T, H, W =>  B, T, C, H, W
                B, F, C, H, W = x.shape
                x = x.view(-1, C, H, W)
                x = self.conv1(x)
                x = x.flatten(2).transpose(2, 1)    # BT, C', H, W => BT, HW, C'
                x = x.reshape(B, -1, self.width)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                cls_embed = self.positional_embedding[0, :].unsqueeze(0)
                tile_pos_embed = self.positional_embedding[1:, :].repeat(self.num_frames, 1)
                tile_temporal_embed = self.temporal_embedding.repeat_interleave(self.patches_per_frame, 0)
                total_pos_embed = tile_pos_embed + tile_temporal_embed
                total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=0)
                x = x + total_pos_embed.to(x.dtype).unsqueeze(0)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.pos_drop(x)

        if not self.use_flash_attn:
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        else:
            x = self.transformer(x)

        if self.global_average_pool:
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        x = self.ln_post(x)

        if self.image_projection is not None:
            x = x @ self.image_projection

        return x


    
class TextTransformer(nn.Module):

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            ls_init_value: float = None,
            output_dim: int = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            causal_mask: float = True,
            flash_attn: bool = False,
            flash_mlp: bool = False,
            fused_bias_fc: bool = False,
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)
        if output_dim is None:
            self.text_projection = None
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.causal_mask = causal_mask

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.width ** -0.5)
            # trunc_normal_(self.text_projection, std=0.001)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, cast_dtype=None, return_embedding=False):
        if cast_dtype is None:
            cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask if self.causal_mask else None)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if return_embedding:
            max_seq_len = text.argmax(dim=-1).max()
            txt_embed = x[:, :max_seq_len+1]
            txt_mask = torch.arange(max_seq_len+1).unsqueeze(0).expand(x.shape[0], -1).to(x.device)
            txt_mask = txt_mask <= text.argmax(dim=-1).unsqueeze(1)
            txt_mask = txt_mask.to(x.dtype)
            
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection.to(cast_dtype)
            return txt_embed, txt_mask, x
        
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection.to(cast_dtype)

        return x
