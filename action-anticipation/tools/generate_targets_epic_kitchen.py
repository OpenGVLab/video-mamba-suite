import argparse
import numpy as np
import pandas as pd
import os.path as osp


def str2sec(string):
    hh, mm, ss = string.split(':')
    return float(hh) * 3600 + float(mm) * 60 + float(ss)


def main():
    parser = argparse.ArgumentParser(description='Generate targets')
    parser.add_argument('--dataset', type=str,
                        choices=['EK55', 'EK100'], default='EK55')
    parser.add_argument('--data-path', type=str,
                        default='/scratch/cluster/yzhao/Datasets/EPIC-KITCHENS/')
    parser.add_argument('--targets-path', type=str,
                        default='/scratch/cluster/yzhao/Datasets/'
                                'EPIC-KITCHENS/features/target_perframe/')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--tick', type=float, default=7.5)
    parser.add_argument('--anno-path', type=str,
                        default='external/rulstm/RULSTM/data/ek55/')
    parser.add_argument('--class-type', type=str, default='action',
                        choices=['verb', 'noun', 'action'])
    args = parser.parse_args()

    # initialize the targets
    targets = {}
    if args.class_type == 'action':
        num_classes = 2513 + 1
    elif args.class_type == 'verb':
        actions = pd.read_csv(osp.join(args.anno_path, 'actions.csv'))
        a_to_v = {a[1]['id'] + 1: a[1]['verb'] + 1
                  for a in actions.iterrows()}
        num_classes = max(a_to_v.values()) + 1
    elif args.class_type == 'noun':
        actions = pd.read_csv(osp.join(args.anno_path, 'actions.csv'))
        a_to_n = {a[1]['id'] + 1: a[1]['noun'] + 1
                  for a in actions.iterrows()}
        num_classes = max(a_to_n.values()) + 1

    for subset in ['training', 'validation']:
        vids = [line.strip() for line in open('{}/{}_videos.csv'.format(args.anno_path, subset))]
        anno_per_video = {}
        for line in open('{}/{}.csv'.format(args.anno_path, subset)):
            idx, vid, f_start, f_end, c_verb, c_noun, c_verbnoun = line.strip().split(',')
            vid = vid.strip()
            if vid not in anno_per_video:
                anno_per_video[vid] = [(int(f_start), int(f_end), int(c_verb), int(c_noun), int(c_verbnoun))]
            else:
                anno_per_video[vid].append((int(f_start), int(f_end), int(c_verb), int(c_noun), int(c_verbnoun)))
        for vid in vids:
            feature = np.load('{}/features/rgb_kinetics_resnet50/{}.npy'.format(args.data_path, vid))
            feature_len = feature.shape[0]
            targets[vid] = np.zeros((feature_len,
                                     num_classes))
            targets[vid][:, 0] = 1
            for anno in anno_per_video[vid]:
                print(vid, anno)
                ss = int(np.floor(anno[0] / args.tick))
                ee = min(int(np.ceil(anno[1] / args.tick)), feature_len)
                if args.class_type == 'action':
                    ci = anno[4]
                elif args.class_type == 'verb':
                    ci = anno[2]
                elif args.class_type == 'noun':
                    ci = anno[3]
                for si in range(ss, ee):
                    targets[vid][si, ci + 1] = 1
                    targets[vid][si, 0] = 0

    print('Dumping npy files..')
    for vid in targets:
        np.save(args.targets_path + '{}.npy'.format(vid), targets[vid])


if __name__ == '__main__':
    main()
