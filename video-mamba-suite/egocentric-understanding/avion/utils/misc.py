import csv
import math
import sys
import torch


def check_loss_nan(loss):
    if not math.isfinite(loss.item()):
        print("Loss is {}, stopping training".format(loss.item()))
        sys.exit(1)


def interpolate_pos_embed(old_pos_embed, model, num_frames):
    embedding_size = old_pos_embed.shape[-1] # channel dim
    num_patches = model.patch_embed.num_patches #
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

    # height (== width) for the checkpoint position embedding
    orig_size = int(((old_pos_embed.shape[-2] - num_extra_tokens)//(num_frames // model.patch_embed.tubelet_size)) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int((num_patches // (num_frames // model.patch_embed.tubelet_size) )** 0.5)
    # class_token and dist_token are kept unchanged
    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        extra_tokens = old_pos_embed[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = old_pos_embed[:, num_extra_tokens:]
        # B, L, C -> BT, H, W, C -> BT, C, H, W
        pos_tokens = pos_tokens.reshape(-1, num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size)
        pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        return new_pos_embed
    else:
        print('Skipping interpolation')
        return old_pos_embed


def generate_label_map(dataset):
    if dataset == 'ek100_cls':
        print("Preprocess ek100 action label space")
        vn_list = []
        mapping_vn2narration = {}
        for f in [
            'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv',
            'datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv',
        ]:
            csv_reader = csv.reader(open(f))
            _ = next(csv_reader)  # skip the header
            for row in csv_reader:
                vn = '{}:{}'.format(int(row[10]), int(row[12]))
                narration = row[8]
                if vn not in vn_list:
                    vn_list.append(vn)
                if vn not in mapping_vn2narration:
                    mapping_vn2narration[vn] = [narration]
                else:
                    mapping_vn2narration[vn].append(narration)
                # mapping_vn2narration[vn] = [narration]
        vn_list = sorted(vn_list)
        print('# of action= {}'.format(len(vn_list)))
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))]
        print(labels[:5])
    elif dataset == 'charades_ego':
        print("=> preprocessing charades_ego action label space")
        vn_list = []
        labels = []
        with open('datasets/CharadesEgo/CharadesEgo/Charades_v1_classes.txt') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                vn = row[0][:4]
                vn_list.append(vn)
                narration = row[0][5:]
                labels.append(narration)
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        print(labels[:5])
    elif dataset == 'egtea':
        print("=> preprocessing egtea action label space")
        labels = []
        with open('datasets/EGTEA/action_idx.txt') as f:
            for row in f:
                row = row.strip()
                narration = ' '.join(row.split(' ')[:-1])
                labels.append(narration.replace('_', ' ').lower())
                # labels.append(narration)
        mapping_vn2act = {label: i for i, label in enumerate(labels)}
        print(len(labels), labels[:5])
    else:
        raise NotImplementedError
    return labels, mapping_vn2act
