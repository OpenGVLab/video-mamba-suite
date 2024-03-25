from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


def enable_grad_checkpointing(model: nn.Module, enable: bool):
    if hasattr(model, 'set_grad_checkpointing'):
        model.set_grad_checkpointing(enable=enable)
    else:
        print("{} has no attribute named 'set_grad_checkpointing'".format(model._get_name()))



def overwrite_deit_model_to_clip_vision(deit_state_dict,
                                        clip_state_dict,
                                        visual_transformer_layers=12,):
    print(deit_state_dict.keys())
    print(clip_state_dict.keys())
    # print("cls token:", deit_state_dict["cls_token"].shape,clip_state_dict["visual.class_embedding"].shape)
    # print("pos_embed:", deit_state_dict["pos_embed"].shape,clip_state_dict["visual.positional_embedding"].shape)
    # print("patch_embed.proj.weight:", deit_state_dict["patch_embed.proj.weight"].shape, clip_state_dict["visual.conv1.weight"].shape)
    
    new_embed_dim = deit_state_dict["cls_token"].shape[-1]
    
    clip_state_dict["visual.class_embedding"] = deit_state_dict["cls_token"].squeeze(0).squeeze(0)
    clip_state_dict["visual.positional_embedding"] = deit_state_dict["pos_embed"].squeeze(0)
    clip_state_dict["visual.conv1.weight"] = deit_state_dict["patch_embed.proj.weight"]
    clip_state_dict["visual.ln_post.weight"] = deit_state_dict["norm.weight"]
    clip_state_dict["visual.ln_post.bias"] = deit_state_dict["norm.bias"]
    
    if clip_state_dict["visual.ln_pre.weight"].shape[0]!=new_embed_dim:
        print("ln_pre dos not match!")
        clip_state_dict.pop("visual.ln_pre.weight")
        clip_state_dict.pop("visual.ln_pre.bias")
    
    for i in range(visual_transformer_layers):
        clip_state_dict[f"visual.transformer.resblocks.{i}.attn.in_proj_weight"] = deit_state_dict[f"blocks.{i}.attn.qkv.weight"]
        clip_state_dict[f"visual.transformer.resblocks.{i}.attn.in_proj_bias"] = deit_state_dict[f"blocks.{i}.attn.qkv.bias"]
        clip_state_dict[f"visual.transformer.resblocks.{i}.attn.out_proj.weight"] = deit_state_dict[f"blocks.{i}.attn.proj.weight"]
        clip_state_dict[f"visual.transformer.resblocks.{i}.attn.out_proj.bias"] = deit_state_dict[f"blocks.{i}.attn.proj.bias"]
        clip_state_dict[f"visual.transformer.resblocks.{i}.ln_1.weight"] = deit_state_dict[f"blocks.{i}.norm1.weight"]
        clip_state_dict[f"visual.transformer.resblocks.{i}.ln_1.bias"] = deit_state_dict[f"blocks.{i}.norm1.bias"]
        clip_state_dict[f"visual.transformer.resblocks.{i}.mlp.c_fc.weight"] = deit_state_dict[f"blocks.{i}.mlp.fc1.weight"]
        clip_state_dict[f"visual.transformer.resblocks.{i}.mlp.c_fc.bias"] = deit_state_dict[f"blocks.{i}.mlp.fc1.bias"]
        clip_state_dict[f"visual.transformer.resblocks.{i}.mlp.c_proj.weight"] = deit_state_dict[f"blocks.{i}.mlp.fc2.weight"]
        clip_state_dict[f"visual.transformer.resblocks.{i}.mlp.c_proj.bias"] = deit_state_dict[f"blocks.{i}.mlp.fc2.bias"]
        clip_state_dict[f"visual.transformer.resblocks.{i}.ln_2.weight"] = deit_state_dict[f"blocks.{i}.norm2.weight"]
        clip_state_dict[f"visual.transformer.resblocks.{i}.ln_2.bias"] = deit_state_dict[f"blocks.{i}.norm2.bias"]
    
    return clip_state_dict


def remap_keys_from_open_clip_to_timesformer(
    clip_state_dict,
    visual_transformer_layers=12,
    textual_transformer_layers=12,
    context_length=77,
    vocab_size=49408,
    use_fast_conv1=False,
    use_flash_attn=False,
    img_size=224,
):
    if 'state_dict' in clip_state_dict:
        clip_state_dict = clip_state_dict['state_dict']
    if list(clip_state_dict.keys())[0].startswith('module.'):
        clip_state_dict = OrderedDict({
            k.replace('module.', ''): v for k, v in clip_state_dict.items()
        })
    remapped_state_dict = OrderedDict()
    key_mapping = {
        "visual.class_embedding" : "visual.cls_token",
        "visual.positional_embedding": "visual.pos_embed",
        "visual.conv1.weight": "visual.patch_embed.proj.weight",
        "visual.ln_pre.weight": "visual.ln_pre.weight",
        "visual.ln_pre.bias": "visual.ln_pre.bias",
        "visual.ln_post.weight": "visual.norm.weight",
        "visual.ln_post.bias": "visual.norm.bias",
        "visual.image_projection": "visual.image_projection",
        "logit_scale": "logit_scale",
        "positional_embedding": "textual.positional_embedding",
        "text_projection": "textual.text_projection",
        "token_embedding.weight": "textual.token_embedding.weight",
        "ln_final.weight": "textual.ln_final.weight",
        "ln_final.bias": "textual.ln_final.bias"
    }
    for layer in range(visual_transformer_layers):
        if use_flash_attn:
            for src_name, tgt_name in {
                'attn.in_proj_weight': 'attn.Wqkv.weight', 'attn.in_proj_bias': 'attn.Wqkv.bias',
                'attn.out_proj.weight': 'attn.out_proj.weight', 'attn.out_proj.bias': 'attn.out_proj.bias',
                'mlp.c_fc.weight': 'mlp.fc1.weight', 'mlp.c_fc.bias': 'mlp.fc1.bias',
                'mlp.c_proj.weight': 'mlp.fc2.weight', 'mlp.c_proj.bias': 'mlp.fc2.bias',
                'ln_1.weight': 'norm1.weight', 'ln_1.bias': 'norm1.bias', 
                'ln_2.weight': 'norm2.weight', 'ln_2.bias': 'norm2.bias', 
            }.items():
                key_mapping[f"visual.transformer.resblocks.{layer}.{src_name}"] = f"visual.blocks.{layer}.{tgt_name}"
        else:
            raise NotImplementedError
        
        
    for layer in range(textual_transformer_layers):
        for name in [
            'attn.in_proj_weight', 'attn.in_proj_bias', 'attn.out_proj.weight', 'attn.out_proj.bias',
            'ln_1.weight', 'ln_1.bias', 'ln_2.weight', 'ln_2.bias',
             'mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias',
        ]:
            key_mapping[f"transformer.resblocks.{layer}.{name}"] = f"textual.transformer.resblocks.{layer}.{name}"

    for key in clip_state_dict:
        if key in ["visual.proj", "text_projection", "logit_scale"]:
            continue  # due to possible dim mismatch, we load this later
        elif key == "visual.class_embedding":
            clip_state_dict[key] = clip_state_dict[key].unsqueeze(0).unsqueeze(0)
        elif key == "visual.positional_embedding":
            print(clip_state_dict[key].shape)
            clip_state_dict[key] = clip_state_dict[key].unsqueeze(0)
        elif key == 'token_embedding.weight':
            old_vocab_size, dim = clip_state_dict[key].shape
            old_dtype = clip_state_dict[key].dtype
            assert vocab_size >= old_vocab_size
            remapped_state_dict[key_mapping[key]] = torch.cat(
                (clip_state_dict[key], torch.zeros((vocab_size - old_vocab_size, dim), dtype=old_dtype)), dim=0
            )
        remapped_state_dict[key_mapping[key]] = clip_state_dict[key]

    return remapped_state_dict
    

# util functions to convert OpenCLIP-style model keys to ViT-style
def remap_keys_from_open_clip_to_vit(
    clip_state_dict,
    visual_transformer_layers=12,
    textual_transformer_layers=12,
    context_length=77,
    vocab_size=49408,
    use_fast_conv1=False,
    use_flash_attn=False,
    skip_load_proj_keys=True,
):
    if 'state_dict' in clip_state_dict:
        clip_state_dict = clip_state_dict['state_dict']
    if list(clip_state_dict.keys())[0].startswith('module.'):
        clip_state_dict = OrderedDict({
            k.replace('module.', ''): v for k, v in clip_state_dict.items()
        })
    remapped_state_dict = OrderedDict()
    key_mapping = {
        "logit_scale": "logit_scale",
        "visual.proj": "visual.image_projection",
        "positional_embedding": "textual.positional_embedding",
        "text_projection": "textual.text_projection",
        "token_embedding.weight": "textual.token_embedding.weight",
        "ln_final.weight": "textual.ln_final.weight",
        "ln_final.bias": "textual.ln_final.bias"
    }

    for layer in range(visual_transformer_layers):
        if use_flash_attn:
            for src_name, tgt_name in {
                'attn.in_proj_weight': 'attn.Wqkv.weight', 'attn.in_proj_bias': 'attn.Wqkv.bias',
                'attn.out_proj.weight': 'attn.out_proj.weight', 'attn.out_proj.bias': 'attn.out_proj.bias',
                'mlp.c_fc.weight': 'mlp.fc1.weight', 'mlp.c_fc.bias': 'mlp.fc1.bias',
                'mlp.c_proj.weight': 'mlp.fc2.weight', 'mlp.c_proj.bias': 'mlp.fc2.bias',
            }.items():
                key_mapping[f"visual.transformer.resblocks.{layer}.{src_name}"] = f"visual.transformer.resblocks.{layer}.{tgt_name}"


    for layer in range(textual_transformer_layers):
        for name in [
            'attn.in_proj_weight', 'attn.in_proj_bias', 'attn.out_proj.weight', 'attn.out_proj.bias',
            'ln_1.weight', 'ln_1.bias', 'ln_2.weight', 'ln_2.bias',
            'mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias',
        ]:
            key_mapping[f"transformer.resblocks.{layer}.{name}"] = f"textual.transformer.resblocks.{layer}.{name}"

    # some keys may need to skip
    skip_keys = ["logit_scale"]
    if skip_load_proj_keys:
        print("Skipping loading proj keys")
        skip_keys += ["visual.proj", "text_projection"]
        
    for key in clip_state_dict:
        if key in skip_keys:
            continue
        if use_fast_conv1 and key == 'visual.conv1.weight':
            remapped_state_dict['visual.conv1.weight'] = clip_state_dict[key].flatten(1)
            # assert mean is not None and std is not None
            # W_2 = clip_state_dict[key].flatten(1)
            # std = torch.tensor(std).float()
            # std = std.repeat_interleave(clip_state_dict[key].shape[-1] * clip_state_dict[key].shape[-2])
            # W_1 = torch.diag(1 / std)
            # W_fused = W_2 @ W_1
            # mean = torch.tensor(mean).float().repeat_interleave(clip_state_dict[key].shape[-1] * clip_state_dict[key].shape[-2])
            # b_1 = mean / std
            # b_fused = W_2 @ (-b_1)
            # remapped_state_dict['visual.conv1.weight'] = W_fused
            # remapped_state_dict['visual.conv1.bias'] = b_fused
        elif key not in key_mapping:
            remapped_state_dict[key] = clip_state_dict[key]
        else:
            if key == 'positional_embedding':
                old_context_length, dim = clip_state_dict[key].shape
                old_dtype = clip_state_dict[key].dtype
                if context_length <= old_context_length:
                    remapped_state_dict[key_mapping[key]] = clip_state_dict[key][:context_length, :]
                else:
                    remapped_state_dict[key_mapping[key]] = torch.cat(
                        (clip_state_dict[key], torch.zeros((context_length - old_context_length, dim), dtype=old_dtype)), dim=0
                    )
            elif key == 'token_embedding.weight':
                old_vocab_size, dim = clip_state_dict[key].shape
                old_dtype = clip_state_dict[key].dtype
                assert vocab_size >= old_vocab_size
                remapped_state_dict[key_mapping[key]] = torch.cat(
                    (clip_state_dict[key], torch.zeros((vocab_size - old_vocab_size, dim), dtype=old_dtype)), dim=0
                )
            else:
                remapped_state_dict[key_mapping[key]] = clip_state_dict[key]

    return remapped_state_dict


def inflate_positional_embeds(
    current_model_state_dict, new_state_dict,
    num_frames=4,
    load_temporal_fix='bilinear',
):
    # allow loading of timesformer with fewer num_frames
    curr_keys = list(current_model_state_dict.keys())
    if 'visual.temporal_embedding' in new_state_dict and 'visual.temporal_embedding' in curr_keys:
        load_temporal_embed = new_state_dict['visual.temporal_embedding']
        load_num_frames = load_temporal_embed.shape[0]
        curr_num_frames = num_frames
        embed_dim = load_temporal_embed.shape[1]

        if load_num_frames != curr_num_frames:
            if load_num_frames > curr_num_frames:
                print(f'### loaded SpaceTimeTransformer model has MORE frames than current...'
                      f'### loading weights, filling in the extras via {load_temporal_fix}')
                new_temporal_embed = load_temporal_embed[:curr_num_frames, :]
            else:
                print(f'### loaded SpaceTimeTransformer model has FEWER frames than current...'
                      f'### loading weights, filling in the extras via {load_temporal_fix}')
                if load_temporal_fix == 'zeros':
                    new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                    new_temporal_embed[:load_num_frames] = load_temporal_embed
                elif load_temporal_fix in ['interp', 'bilinear']:
                    # interpolate
                    # unsqueeze so pytorch thinks its an image
                    mode = 'nearest'
                    if load_temporal_fix == 'bilinear':
                        mode = 'bilinear'
                    load_temporal_embed = load_temporal_embed.unsqueeze(0).unsqueeze(0)
                    new_temporal_embed = F.interpolate(load_temporal_embed,
                                                       (curr_num_frames, embed_dim), mode=mode).squeeze(0).squeeze(0)
                else:
                    raise NotImplementedError
            new_state_dict['visual.temporal_embedding'] = new_temporal_embed
    # allow loading with smaller spatial patches. assumes custom border crop, to append the
    # border patches to the input sequence
    if 'visual.positional_embedding' in new_state_dict and 'visual.positional_embedding' in curr_keys:
        load_pos_embed = new_state_dict['visual.positional_embedding']
        load_num_patches = load_pos_embed.shape[0]
        curr_pos_embed = current_model_state_dict['visual.positional_embedding']
        if load_num_patches != curr_pos_embed.shape[0]:
            raise NotImplementedError(
                'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

    return new_state_dict
