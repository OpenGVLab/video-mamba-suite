import argparse
from collections import OrderedDict
from functools import partial
import json
import os
import pickle
import time
import numpy as np
import pandas as pd

import kornia as K
import scipy
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
import torchvision
import torchvision.transforms._transforms_video as transforms_video
from timm.data.loader import MultiEpochsDataLoader
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy

from avion.utils.train_utils import init_dist_slurm
from avion.data.clip_dataset import get_downstream_dataset
from avion.data.tokenizer import tokenize
from avion.data.transforms import Permute

import avion.models.model_clip as model_clip
from avion.models.utils import inflate_positional_embeds
from avion.optim.schedulers import cosine_scheduler
import avion.utils.distributed as dist_utils
from avion.utils.evaluation_ek100cls import get_marginal_indexes, get_mean_accuracy, marginalize
from avion.utils.meters import AverageMeter, ProgressMeter
from avion.utils.misc import check_loss_nan, generate_label_map


def get_args_parser():
    parser = argparse.ArgumentParser(description='AVION finetune ek100 cls', add_help=False)
    parser.add_argument('--dataset', default='ek100_cls', type=str, choices=['ek100_mir'])
    parser.add_argument('--root', default='datasets/EK100/EK100_256p_15sec/', type=str, help='path to train dataset root')
    parser.add_argument('--train-metadata', type=str,
                        default='datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv')
    parser.add_argument('--val-metadata', type=str,
                        default='datasets/EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--num-crops', default=1, type=int, help='number of crops in transforms for testing')
    parser.add_argument('--num-clips', default=1, type=int, help='number of clips for testing')
    parser.add_argument('--video-chunk-length', default=15, type=int)
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=2, type=int, help='clip stride')
    parser.add_argument('--norm-style', default='openai', type=str, choices=['openai', 'timm'])
    parser.add_argument('--fused-decode-crop', action='store_true', dest='fused_decode_crop')
    parser.add_argument('--no-fused-decode-crop', action='store_false', dest='fused_decode_crop')
    parser.set_defaults(fused_decode_crop=False)
    parser.add_argument('--decode-threads', default=1, type=int)
    parser.add_argument('--use-multi-epochs-loader', action='store_true')
    # model
    parser.add_argument('--grad-checkpointing', action='store_true', dest='use_grad_checkpointing')
    parser.add_argument('--no-grad-checkpointing', action='store_false', dest='use_grad_checkpointing')
    parser.set_defaults(use_grad_checkpointing=False)
    parser.add_argument('--use-fast-conv1', action='store_true', dest='use_fast_conv1')
    parser.add_argument('--disable-fast-conv1', action='store_false', dest='use_fast_conv1')
    parser.set_defaults(use_fast_conv1=False)
    parser.add_argument('--use-flash-attn', action='store_true', dest='use_flash_attn')
    parser.add_argument('--disable-flash-attn', action='store_false', dest='use_flash_attn')
    parser.set_defaults(use_flash_attn=False)
    parser.add_argument('--patch-dropout', default=0., type=float)
    parser.add_argument('--drop-path-rate', default=0.1, type=float)
    parser.add_argument('--dropout-rate', default=0.5, type=float, help='dropout for the last linear layer')
    parser.add_argument('--num-classes', default=3806, type=float, help='number of classes for the last linear layer')
    parser.add_argument('--pretrain-model', default='', type=str, help='path of pretrained model')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    # mixup
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    # training
    parser.add_argument('--use-zero', action='store_true', dest='use_zero', help='use ZeRO optimizer')
    parser.add_argument('--no-use-zero', action='store_false', dest='use_zero', help='use ZeRO optimizer')
    parser.set_defaults(use_zero=False)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup-epochs', default=2, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=32, type=int, help='number of samples per-device/per-gpu')
    parser.add_argument('--optimizer', default='adamw', choices=['adamw', 'sgd'], type=str)
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float, help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float, help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int, help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.05, type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true', help='disable mixed-precision training (requires more memory and compute)')
    parser.add_argument('--grad-clip-norm', default=None, type=float)
    # system
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--pickle-filename', default='', type=str, help='pickle filename to dump')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    return parser


def main(args):
    init_dist_slurm(args)
    dist_utils.random_seed(args.seed, dist_utils.get_rank())

    if args.pretrain_model:
        ckpt_path = args.pretrain_model
    else:
        raise Exception('no checkpoint found, add it by `--pretrain-model ${CHECKPOINT_PATH}`')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    old_args = ckpt['args']
    print("=> creating model: {}".format(old_args.model))

    model = getattr(model_clip, old_args.model)(
        freeze_temperature=True,
        use_grad_checkpointing=args.use_grad_checkpointing,
        context_length=old_args.context_length,
        vocab_size=old_args.vocab_size,
        patch_dropout=args.patch_dropout,
        num_frames=args.clip_length,
        drop_path_rate=args.drop_path_rate,
        use_fast_conv1=args.use_fast_conv1,
        use_flash_attn=args.use_flash_attn,
        use_quick_gelu=True,
        project_embed_dim=old_args.project_embed_dim,
        pretrain_zoo=old_args.pretrain_zoo,
        pretrain_path=old_args.pretrain_path,
    )
    model.logit_scale.requires_grad = False
    print('=> inflating PE in models due to different frame numbers')
    state_dict = inflate_positional_embeds(
        model.state_dict(), state_dict,
        num_frames=args.clip_length,
        load_temporal_fix='bilinear',
    )
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt['epoch']))

    model = model_clip.VideoClassifier(
        model.visual,
        dropout=args.dropout_rate,
        num_classes=args.num_classes
    )
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)

    # define loss function (criterion) and optimizer
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    n_wd, n_non_wd = [], []
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if (p.ndim < 2 or 'bias' in n or
            'ln' in n or 'bn' in n or
            'pos_embed' in n or 'positional_embedding' in n
        ):
            n_non_wd.append(n)
            p_non_wd.append(p)
        else:
            n_wd.append(n)
            p_wd.append(p)

    # print('parameters without wd:', n_non_wd)
    # print('parameters with wd:', n_wd)
    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    total_batch_size = args.batch_size * dist_utils.get_world_size()
    args.lr = args.lr * total_batch_size / 128
    args.lr_start = args.lr_start * total_batch_size / 128
    args.lr_end = args.lr_end * total_batch_size / 128
    if args.optimizer == 'sgd':
        opt_fn = torch.optim.SGD
        opt_kwargs = {"lr": args.lr, "momentum": args.betas[0], "weight_decay": args.wd}
    elif args.optimizer == 'adamw':
        opt_fn = torch.optim.AdamW
        opt_kwargs = {"lr": args.lr, "betas": args.betas,
                      "eps": args.eps, "weight_decay": args.wd}
    if args.use_zero:
        print('Training with ZeroRedundancyOptimizer')
        optimizer = ZeroRedundancyOptimizer(optim_params, optimizer_class=opt_fn, **opt_kwargs)
    else:
        optimizer = opt_fn(optim_params, **opt_kwargs)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            if checkpoint['args'].clip_length != args.clip_length:
                load_temporal_embedding = checkpoint['state_dict']['module.visual.temporal_embedding']
                load_temporal_embedding = load_temporal_embedding.unsqueeze(0).permute(0, 2, 1)
                new_temporal_embed = F.interpolate(load_temporal_embedding, size=(args.clip_length,), mode='linear').permute(0, 2, 1).squeeze(0)
                checkpoint['state_dict']['module.visual.temporal_embedding'] = new_temporal_embed
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            best_acc1 = checkpoint['best_acc1']
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1']
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    torch.backends.cudnn.benchmark = True

    if args.norm_style == 'openai':
        mean, std = [122.7709383, 116.7460125, 104.09373615000001], [68.5005327, 66.6321579, 70.32316305]
    elif args.norm_style == 'timm':
        mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]
    else:
        raise ValueError('--norm-style should be in ["openai", "timm"]!')

    crop_size = 336 if old_args.model.endswith("_336PX") else 224

    if args.fused_decode_crop:
        base_train_transform_ls = [
            # Permute([3, 0, 1, 2]),
            # transforms_video.NormalizeVideo(mean=mean, std=std),
        ]
        gpu_train_transform_ls = [K.enhance.Normalize(mean=mean, std=std)]
        base_val_transform_ls = [
            # Permute([3, 0, 1, 2]),
            # torchvision.transforms.Resize(crop_size),
        ]
        gpu_val_transform_ls = [K.enhance.Normalize(mean=mean, std=std)]
    else:
        base_train_transform_ls = [
            Permute([3, 0, 1, 2]),
            torchvision.transforms.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
            transforms_video.NormalizeVideo(mean=mean, std=std),
        ]
        gpu_train_transform_ls = []
        base_val_transform_ls = [
            Permute([3, 0, 1, 2]),
            torchvision.transforms.Resize(crop_size),
            torchvision.transforms.CenterCrop(crop_size),
            transforms_video.NormalizeVideo(mean=mean, std=std),
        ]
        gpu_val_transform_ls = []
    train_transform = torchvision.transforms.Compose(base_train_transform_ls)
    train_transform_gpu = torch.nn.Sequential(*gpu_train_transform_ls)
    val_transform = torchvision.transforms.Compose(base_val_transform_ls)
    val_transform_gpu = torch.nn.Sequential(*gpu_val_transform_ls)

    # build dataset
    _, mapping_vn2act = generate_label_map(args.dataset)
    if args.dataset == 'ek100_cls':
        args.mapping_act2v = {i: int(vn.split(':')[0]) for (vn, i) in mapping_vn2act.items()}
        args.mapping_act2n = {i: int(vn.split(':')[1]) for (vn, i) in mapping_vn2act.items()}
        args.actions = pd.DataFrame.from_dict({'verb': args.mapping_act2v.values(), 'noun': args.mapping_act2n.values()})
    num_clips_at_val = args.num_clips
    args.num_clips = 1
    train_dataset = get_downstream_dataset(
        train_transform, crop_size, args, subset='train', label_mapping=mapping_vn2act,
    )
    args.num_clips = num_clips_at_val
    val_dataset = get_downstream_dataset(
        val_transform, crop_size, args, subset='val', label_mapping=mapping_vn2act,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    if args.use_multi_epochs_loader:
        train_loader = MultiEpochsDataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            collate_fn=None,
            num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True
        )
    print('len(train_loader) = {}'.format(len(train_loader)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler, drop_last=False
    )
    print('len(val_loader) = {}'.format(len(val_loader)))

    if args.evaluate:
        val_stats = validate(val_loader, val_transform_gpu, model, args, len(val_dataset))
        if dist_utils.is_main_process():
            with open(os.path.join(args.output_dir, 'eval_log.txt'), 'a') as f:
                f.write(json.dumps(val_stats) + '\n')
        return

    lr_schedule = cosine_scheduler(
        args.lr, args.lr_end, args.epochs, len(train_loader) // args.update_freq,
        warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start
    )

    print(args)

    print("=> beginning training")
    best_acc1 = 0.
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_stats = train(train_loader, train_transform_gpu, model, criterion, optimizer, scaler, epoch, mixup_fn, lr_schedule, args)

        if (epoch + 1) % args.eval_freq != 0:
            continue

        val_stats = validate(val_loader, val_transform_gpu, model, args, len(val_dataset))
        acc1 = val_stats['acc1']

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        print("=> saving checkpoint")
        if args.use_zero:
            print('consolidated on rank {} because of ZeRO'.format(args.rank))
            optimizer.consolidate_state_dict(0)
        dist_utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict() if dist_utils.is_main_process() else None,
                'scaler': scaler.state_dict(),
                'best_acc1': best_acc1,
                'args': args,
            }, is_best, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}

        if dist_utils.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train(train_loader, transform_gpu, model, criterion, optimizer, scaler, epoch, mixup_fn, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    model_time = AverageMeter('Model', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    losses = AverageMeter('Loss', ':.4e')
    metric_names = ['Acc@1', 'Acc@5', 'Noun Acc@1', 'Verb Acc@1']
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':6.2f')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, model_time, mem, losses, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for data_iter, (uid, videos, target) in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None:
                param_group['lr'] = lr_schedule[it]

        videos = videos.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        if args.fused_decode_crop and len(transform_gpu) > 0:
            videos = videos.permute(0, 4, 1, 2, 3)
            videos = transform_gpu(videos)
        
        if isinstance(mixup_fn, list):
            videos_mixed, targets_mixed = mixup_fn[2](videos, target)
        elif isinstance(mixup_fn, Mixup):
            videos_mixed, targets_mixed = mixup_fn(videos, target)
        else:
            videos_mixed, targets_mixed = videos, target

        optimizer.zero_grad()

        tic = time.time()
        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            output = model(videos_mixed)
            loss = criterion(output, targets_mixed)
            loss /= args.update_freq

        check_loss_nan(loss)
        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        if args.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # torch.cuda.empty_cache()
        model_time.update(time.time() - tic)

        output = torch.softmax(output, dim=1)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), videos.size(0))
        metrics['Acc@1'].update(acc1.item(), videos.size(0))
        metrics['Acc@5'].update(acc5.item(), videos.size(0))
        if args.dataset == 'ek100_cls':
            vi = get_marginal_indexes(args.actions, 'verb')
            ni = get_marginal_indexes(args.actions, 'noun')
            verb_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), vi)).cuda(args.gpu, non_blocking=True)
            noun_scores = torch.tensor(marginalize(output.detach().cpu().numpy(), ni)).cuda(args.gpu, non_blocking=True)
            target_to_verb = torch.tensor([args.mapping_act2v[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
            target_to_noun = torch.tensor([args.mapping_act2n[a] for a in target.tolist()]).cuda(args.gpu, non_blocking=True)
            acc1_verb, _ = accuracy(verb_scores, target_to_verb, topk=(1, 5))
            acc1_noun, _ = accuracy(noun_scores, target_to_noun, topk=(1, 5))
            metrics['Verb Acc@1'].update(acc1_verb.item(), videos.size(0))
            metrics['Noun Acc@1'].update(acc1_noun.item(), videos.size(0))
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            progress.display(optim_iter)
    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'loss': losses.avg}


def validate(val_loader, transform_gpu, model, args, num_videos):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['Acc@1', 'Acc@5']
    metrics = OrderedDict([(name, AverageMeter(name, ':6.2f')) for name in metric_names])
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Test: "
    )

    # switch to eval mode
    model.eval()

    all_logits = [[] for _ in range(args.world_size)]
    all_probs = [[] for _ in range(args.world_size)]
    all_targets = [[] for _ in range(args.world_size)]
    total_num = 0
    with amp.autocast(enabled=not args.disable_amp):
        with torch.no_grad():
            end = time.time()
            for i, (uid, videos, targets) in enumerate(val_loader):
                # measure data loading time
                data_time.update(time.time() - end)
                if isinstance(videos, torch.Tensor):
                    videos = [videos, ]
                logits_allcrops = []
                for crop in videos:
                    crop = crop.cuda(args.gpu, non_blocking=True)
                    if args.fused_decode_crop and len(transform_gpu) > 0:
                        crop = crop.permute(0, 4, 1, 2, 3)
                        crop = transform_gpu(crop)
                    logits = model(crop)
                    logits_allcrops.append(logits)
                logits_allcrops = torch.stack(logits_allcrops, 1)
                probs_allcrops = torch.softmax(logits_allcrops, dim=2)
                targets = targets.cuda(args.gpu, non_blocking=True)
                targets_repeated = torch.repeat_interleave(targets, len(videos))

                acc1, acc5 = accuracy(torch.flatten(logits_allcrops, 0, 1), targets_repeated, topk=(1, 5))
                metrics['Acc@1'].update(acc1.item(), targets_repeated.size(0))
                metrics['Acc@5'].update(acc5.item(), targets_repeated.size(0))

                gathered_logits = [torch.zeros_like(logits_allcrops) for _ in range(args.world_size)]
                gathered_probs = [torch.zeros_like(probs_allcrops) for _ in range(args.world_size)]
                gathered_targets = [torch.zeros_like(targets) for _ in range(args.world_size)]
                torch.distributed.all_gather(gathered_logits, logits_allcrops)
                torch.distributed.all_gather(gathered_probs, probs_allcrops)
                torch.distributed.all_gather(gathered_targets, targets)
                for j in range(args.world_size):
                    all_logits[j].append(gathered_logits[j].detach().cpu())
                    all_probs[j].append(gathered_probs[j].detach().cpu())
                    all_targets[j].append(gathered_targets[j].detach().cpu())
                total_num += logits_allcrops.shape[0] * args.world_size

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                mem.update(torch.cuda.max_memory_allocated() // 1e9)

                if i % args.print_freq == 0:
                    progress.display(i)
    progress.synchronize()
    for j in range(args.world_size):
        all_logits[j] = torch.cat(all_logits[j], dim=0).numpy()
        all_probs[j] = torch.cat(all_probs[j], dim=0).numpy()
        all_targets[j] = torch.cat(all_targets[j], dim=0).numpy()
    all_logits_reorg, all_probs_reorg, all_targets_reorg = [], [], []
    for i in range(total_num):
        all_logits_reorg.append(all_logits[i % args.world_size][i // args.world_size])
        all_probs_reorg.append(all_probs[i % args.world_size][i // args.world_size])
        all_targets_reorg.append(all_targets[i % args.world_size][i // args.world_size])
    all_logits = np.stack(all_logits_reorg, axis=0)
    all_probs = np.stack(all_probs_reorg, axis=0)
    all_targets = np.stack(all_targets_reorg, axis=0)
    all_logits = all_logits[:num_videos, :].mean(axis=1)
    all_probs = all_probs[:num_videos, :].mean(axis=1)
    all_targets = all_targets[:num_videos, ]
    if args.pickle_filename != '':
        prob_dict = {'logits': all_logits, 'probs': all_probs, 'targets': all_targets}
        pickle.dump(prob_dict, open(args.pickle_filename, 'wb'))
    for s, all_preds in zip(['logits', ' probs'], [all_logits, all_probs]):
        if s == 'logits': all_preds = scipy.special.softmax(all_preds, axis=1)
        acc1 = top_k_accuracy_score(all_targets, all_preds, k=1, labels=np.arange(0, args.num_classes))
        acc5 = top_k_accuracy_score(all_targets, all_preds, k=5, labels=np.arange(0, args.num_classes))
        dataset = 'EK100' if args.dataset == 'ek100_cls' else 'EGTEA'
        print('[Average {s}] {dataset} * Acc@1 {top1:.3f} Acc@5 {top5:.3f}'.format(s=s, dataset=dataset, top1=acc1, top5=acc5))
        cm = confusion_matrix(all_targets, all_preds.argmax(axis=1))
        mean_acc, acc = get_mean_accuracy(cm)
        print('Mean Acc. = {:.3f}, Top-1 Acc. = {:.3f}'.format(mean_acc, acc))

        if args.dataset == 'ek100_cls':
            vi = get_marginal_indexes(args.actions, 'verb')
            ni = get_marginal_indexes(args.actions, 'noun')
            verb_scores = marginalize(all_preds, vi)
            noun_scores = marginalize(all_preds, ni)
            target_to_verb = np.array([args.mapping_act2v[a] for a in all_targets.tolist()])
            target_to_noun = np.array([args.mapping_act2n[a] for a in all_targets.tolist()])
            cm = confusion_matrix(target_to_verb, verb_scores.argmax(axis=1))
            _, acc = get_mean_accuracy(cm)
            print('Verb Acc@1: {:.3f}'.format(acc))
            cm = confusion_matrix(target_to_noun, noun_scores.argmax(axis=1))
            _, acc = get_mean_accuracy(cm)
            print('Noun Acc@1: {:.3f}'.format(acc))
    return {'acc1': metrics['Acc@1'].avg, 'acc5': metrics['Acc@5'].avg, 'mean_acc': mean_acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LAVILA training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
