import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from avion.models.transformer import TextTransformer, VisionTransformer
from avion.models.timesformer import SpaceTimeTransformer
from avion.models.timemamba import TimeMamba
from avion.models.vimamba import ViViM
from avion.models.utils import enable_grad_checkpointing, remap_keys_from_open_clip_to_vit, remap_keys_from_open_clip_to_timesformer, overwrite_deit_model_to_clip_vision

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class VideoClassifier(nn.Module):
    def __init__(self,
                 vision_model: nn.Module,
                 dropout: float,
                 num_classes: int,
                 **kwargs,):
        super().__init__()
        self.visual = vision_model
        self.dropout = nn.Dropout(dropout)
        if hasattr(self.visual, "image_projection"):
            self.visual.image_projection = None
        self.fc_cls = nn.Linear(vision_model.width, num_classes, bias=True)
        self.fc_cls.weight.data.normal_(mean=0.0, std=0.01)
        self.fc_cls.bias.data.zero_()

    def get_num_layers(self):
        return self.visual.get_num_layers()

    def forward(self, image):
        image_embed = self.visual(image)
        if isinstance(image_embed, list):
            assert len(image_embed) == 1
            image_embed = image_embed[0]
        logit = self.fc_cls(self.dropout(image_embed))
        return logit
    

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 vision_model: nn.Module,
                 text_model: nn.Module,
                 vision_width: int = None,
                 text_width: int = None,
                 freeze_temperature=False,
                 **kwargs
    ):
        super().__init__()

        self.visual = vision_model
        self.textual = text_model

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if freeze_temperature:
            self.logit_scale.requires_grad_(False)
        if vision_width is not None:
            self.vision_width = vision_width
            self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        else:
            self.image_projection = None
        if text_width is not None:
            self.text_width = text_width
            self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim))
        else:
            self.text_projection = None

        self.init_parameters()

    def init_parameters(self):
        if self.image_projection is not None:
            trunc_normal_(self.image_projection, std=self.vision_width ** -0.5)
        if self.text_projection is not None:
            trunc_normal_(self.text_projection, std=self.text_width ** -0.5)

    def encode_image(self, image):
        x = self.visual(image)
        if self.image_projection is not None:
            x = x @ self.image_projection.to(x.dtype)
        return x

    def encode_text(self, text, cast_dtype=None, return_embedding=False):
        if return_embedding:
            txt_embed, txt_mask, x = self.textual(text, cast_dtype=cast_dtype, return_embedding=return_embedding)
            if self.text_projection is not None:
                x = x @ self.text_projection.to(x.dtype)
            return txt_embed, txt_mask, x
        else:
            x = self.textual(text, cast_dtype=cast_dtype)
            if self.text_projection is not None:
                x = x @ self.text_projection.to(x.dtype)
            return x

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text, cast_dtype=image_embed.dtype)

        return F.normalize(image_embed, dim=-1), F.normalize(text_embed, dim=-1), self.logit_scale.exp()


def CLIP_VITT16(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    skip_load_proj_keys=True,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        224, 16, 192, 12, 3, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)


    if pretrain_zoo == "openai":
        assert False
    elif pretrain_zoo == "open_clip":
        assert False
    elif pretrain_zoo == "deit":
        print("=> loading deit model")
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        state_dict = checkpoint["model"]
        
        print("=> overwrite deit vision model to clip vision model")
        overwrited_clip_state_dict = overwrite_deit_model_to_clip_vision(state_dict, clip_model.state_dict())
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            overwrited_clip_state_dict,
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn, 
            skip_load_proj_keys=skip_load_proj_keys
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError

    return model

def CLIP_VITS16(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    skip_load_proj_keys=True,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        224, 16, 384, 12, 6, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)


    if pretrain_zoo == "openai":
        assert False
    elif pretrain_zoo == "open_clip":
        assert False
    elif pretrain_zoo == "deit":
        print("=> loading deit model")
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        state_dict = checkpoint["model"]
        
        print("=> overwrite deit vision model to clip vision model")
        overwrited_clip_state_dict = overwrite_deit_model_to_clip_vision(state_dict, clip_model.state_dict())
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            overwrited_clip_state_dict,
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn, 
            skip_load_proj_keys=skip_load_proj_keys
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError

    return model

def CLIP_VITB32(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    skip_load_proj_keys=True,
    use_quick_gelu=True,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        224, 32, 768, 12, 12, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
        act_layer=nn.GELU if not use_quick_gelu else QuickGELU
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, 
                                 width=512, heads=8, layers=12, output_dim=project_embed_dim, 
                                 causal_mask=not use_bidirectional_lm, 
                                 act_layer=nn.GELU if not use_quick_gelu else QuickGELU)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)


    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/32', device='cpu')
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(),
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn, 
            skip_load_proj_keys=skip_load_proj_keys
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "open_clip":
        assert pretrain_path is not None
        state_dict = torch.load(pretrain_path)
        print("=> loading open_clip model")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            state_dict, 
            use_fast_conv1=use_fast_conv1, 
            use_flash_attn=use_flash_attn, 
            skip_load_proj_keys=skip_load_proj_keys
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    

    return model



def CLIP_VITB16(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    skip_load_proj_keys=True,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        224, 16, 768, 12, 12, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)


    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(),
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn, 
            skip_load_proj_keys=skip_load_proj_keys
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "open_clip":
        assert pretrain_path is not None
        state_dict = torch.load(pretrain_path)
        print("=> loading open_clip model")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            state_dict, 
            use_fast_conv1=use_fast_conv1, 
            use_flash_attn=use_flash_attn, 
            skip_load_proj_keys=skip_load_proj_keys
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "deit":
        print("=> loading deit model")
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        state_dict = checkpoint["model"]
        
        print("=> overwrite deit vision model to clip vision model")
        overwrited_clip_state_dict = overwrite_deit_model_to_clip_vision(state_dict, clip_model.state_dict())
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            overwrited_clip_state_dict,
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn, 
            skip_load_proj_keys=skip_load_proj_keys
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError

    return model



def CLIP_VITL14(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    vocab_size=49408,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        224, 14, 1024, 24, 16, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=vocab_size, width=768, heads=12, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)

    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-L/14', device='cpu')
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(), 24,
            context_length=context_length,
            vocab_size=vocab_size,
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "open_clip":
        assert pretrain_path is not None
        state_dict = torch.load(pretrain_path)
        print("=> loading open_clip model")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(state_dict, use_flash_attn=use_flash_attn)
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model


def CLIP_VITL14_336PX(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    vocab_size=49408,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        336, 14, 1024, 24, 16, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=vocab_size, width=768, heads=12, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)

    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-L/14@336px', device='cpu')
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(), 24,
            context_length=context_length,
            vocab_size=vocab_size,
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "open_clip":
        assert pretrain_path is not None
        state_dict = torch.load(pretrain_path)
        print("=> loading open_clip model")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(state_dict, use_flash_attn=use_flash_attn)
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model


def CLIP_TimeSformerT16(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = SpaceTimeTransformer(
        224, 16, 3, 0, 192, 12, 3, 4, 
        num_frames=num_frames,
        is_tanh_gating=True,
        attention_style="timesformer-div",
        use_flash_attn=use_flash_attn,
        ln_pre=True,
        output_dim=project_embed_dim
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    # enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)
    
    # print(vision_model.state_dict().keys())
    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        # print(clip_model.state_dict().keys())
        remapped_state_dict = remap_keys_from_open_clip_to_timesformer(
            clip_model.state_dict(),
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "deit":
        print("=> loading deit model")
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        state_dict = checkpoint["model"]
        
        print("=> overwrite deit vision model to clip vision model")
        overwrited_clip_state_dict = overwrite_deit_model_to_clip_vision(state_dict, clip_model.state_dict())
        remapped_state_dict = remap_keys_from_open_clip_to_timesformer(
            overwrited_clip_state_dict,
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
        
    else:
        raise NotImplementedError
    return model

def CLIP_TimeSformerB16(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = SpaceTimeTransformer(
        224, 16, 3, 0, 768, 12, 12, 4, 
        num_frames=num_frames,
        is_tanh_gating=True,
        attention_style="timesformer-div",
        use_flash_attn=use_flash_attn,
        ln_pre=True,
        output_dim=project_embed_dim
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    # enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)
    
    # print(vision_model.state_dict().keys())
    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        # print(clip_model.state_dict().keys())
        remapped_state_dict = remap_keys_from_open_clip_to_timesformer(
            clip_model.state_dict(),
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "deit":
        print("=> loading deit model")
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        state_dict = checkpoint["model"]
        
        print("=> overwrite deit vision model to clip vision model")
        overwrited_clip_state_dict = overwrite_deit_model_to_clip_vision(state_dict, clip_model.state_dict())
        remapped_state_dict = remap_keys_from_open_clip_to_timesformer(
            overwrited_clip_state_dict,
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
        
    else:
        raise NotImplementedError
    return model

def CLIP_TimeSformerL14(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = SpaceTimeTransformer(
        224, 14, 3, 0, 1024, 24, 16, 4, 
        num_frames=num_frames,
        is_tanh_gating=True,
        attention_style="timesformer-div",
        use_flash_attn=use_flash_attn,
        ln_pre=True,
        output_dim=project_embed_dim
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=768, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)
    
    # print(vision_model.state_dict().keys())
    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-L/14', device='cpu')
        # print(clip_model.state_dict().keys())
        remapped_state_dict = remap_keys_from_open_clip_to_timesformer(
            clip_model.state_dict(),
            use_flash_attn=use_flash_attn,
            visual_transformer_layers=24,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model


def CLIP_FrozenInTime_TimeSformerB16(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = SpaceTimeTransformer(
        224, 16, 3, 0, 768, 12, 12, 4, 
        num_frames=num_frames,
        is_tanh_gating=True,
        attention_style="frozen-in-time",
        use_flash_attn=use_flash_attn,
        ln_pre=True,
        output_dim=project_embed_dim
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)
    
    # print(vision_model.state_dict().keys())
    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        # print(clip_model.state_dict().keys())
        remapped_state_dict = remap_keys_from_open_clip_to_timesformer(
            clip_model.state_dict(),
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model

def CLIP_TimeMamba_like_timesformer(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = TimeMamba(
        224, 16, 3, 0, 768, 12, 12, 4, 
        num_frames=num_frames,
        is_tanh_gating=True,
        attention_style="timesformer-div",
        use_flash_attn=use_flash_attn,
        ln_pre=True,
        output_dim=project_embed_dim
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    # enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)
    
    # print(vision_model.state_dict().keys())
    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        # print(clip_model.state_dict().keys())
        remapped_state_dict = remap_keys_from_open_clip_to_timesformer(
            clip_model.state_dict(),
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model

def CLIP_TimeMamba_like_frozen(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = TimeMamba(
        224, 16, 3, 0, 768, 12, 12, 4, 
        num_frames=num_frames,
        is_tanh_gating=True,
        attention_style="frozen-in-time",
        use_flash_attn=use_flash_attn,
        ln_pre=True,
        output_dim=project_embed_dim
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    # enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)
    
    # print(vision_model.state_dict().keys())
    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        # print(clip_model.state_dict().keys())
        remapped_state_dict = remap_keys_from_open_clip_to_timesformer(
            clip_model.state_dict(),
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model


def CLIP_TimeMamba_like_frozen_joint(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = TimeMamba(
        224, 16, 3, 0, 768, 12, 12, 4, 
        num_frames=num_frames,
        is_tanh_gating=True,
        attention_style="frozen-joint",
        use_flash_attn=use_flash_attn,
        ln_pre=True,
        output_dim=project_embed_dim
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    # enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)
    
    # print(vision_model.state_dict().keys())
    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        # print(clip_model.state_dict().keys())
        remapped_state_dict = remap_keys_from_open_clip_to_timesformer(
            clip_model.state_dict(),
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model


def CLIP_ViViM_tiny(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = ViViM(
        patch_size=16, 
        embed_dim=192, 
        depth=24, 
        num_frames=num_frames,
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        final_pool_type='mean', 
        if_abs_pos_embed=True, 
        if_rope=False, 
        if_rope_residual=False, 
        bimamba_type="v2", 
        if_cls_token=True, 
        if_devide_out=True, 
        use_middle_cls_token=True, 
        output_dim=project_embed_dim,
        drop_path_rate=drop_path_rate,
        **kwargs)

    
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    # enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)
    
    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        # print(clip_model.state_dict().keys())
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(),
            use_flash_attn=use_flash_attn,
        )
        textual_state_dict = {k:v for k,v in remapped_state_dict.items() if "textual" in k}
        missing_keys, unexpected_keys = model.load_state_dict(textual_state_dict, strict=False)
        print("load textual missing_keys: ", missing_keys)
        print("load textual unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    
    
    # overwrite visual model state
    print("=> overwrite visual model state")
    checkpoint = torch.load("/mnt/petrelfs/chenguo/pretrained_models/vim/Vim-tiny-midclstok/vim_t_midclstok_76p1acc.pth", map_location="cpu")
    missing_keys, unexpected_keys = model.visual.load_state_dict(checkpoint["model"], strict=False)
    print("load visual missing_keys: ", missing_keys)
    print("load visual unexpected_keys: ", unexpected_keys)
    
    return model


def CLIP_ViViM_small(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = ViViM(
        patch_size=16, 
        embed_dim=384, 
        depth=24, 
        num_frames=num_frames,
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        final_pool_type='mean', 
        if_abs_pos_embed=True, 
        if_rope=False, 
        if_rope_residual=False, 
        bimamba_type="v2", 
        if_cls_token=True, 
        if_devide_out=True, 
        use_middle_cls_token=True, 
        output_dim=project_embed_dim,
        drop_path_rate=drop_path_rate,
        **kwargs)

    
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    # enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature)
    
    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load('ViT-B/16', device='cpu')
        # print(clip_model.state_dict().keys())
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(),
            use_flash_attn=use_flash_attn,
        )
        textual_state_dict = {k:v for k,v in remapped_state_dict.items() if "textual" in k}
        missing_keys, unexpected_keys = model.load_state_dict(textual_state_dict, strict=False)
        print("load textual missing_keys: ", missing_keys)
        print("load textual unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    
    
    # overwrite visual model state
    print("=> overwrite visual model state")
    checkpoint = torch.load("/mnt/petrelfs/chenguo/pretrained_models/vim/Vim-small-midclstok/vim_s_midclstok_80p5acc.pth", map_location="cpu")
    missing_keys, unexpected_keys = model.visual.load_state_dict(checkpoint["model"], strict=False)
    print("load visual missing_keys: ", missing_keys)
    print("load visual unexpected_keys: ", unexpected_keys)
    
    return model
