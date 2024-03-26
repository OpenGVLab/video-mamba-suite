import argparse
from collections import OrderedDict
from functools import partial
import json
import os
import time
import numpy as np
import pandas as pd

import kornia as K
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
import torchvision
import torchvision.transforms._transforms_video as transforms_video
from timm.data.loader import MultiEpochsDataLoader
from avion.data.clip_dataset import VideoCaptionDatasetCLIP
from avion.data.tokenizer import tokenize
from avion.data.transforms import Permute

from avion.utils.train_utils import init_dist_slurm
from avion.losses.losses import MaxMarginRankingLoss
import avion.models.model_clip as model_clip
from avion.models.utils import inflate_positional_embeds
from avion.optim.schedulers import cosine_scheduler
import avion.utils.distributed as dist_utils
from avion.utils.evaluation_ek100mir import get_mAP, get_nDCG
from avion.utils.meters import AverageMeter, ProgressMeter
from avion.utils.misc import check_loss_nan


def get_args_parser():
    parser = argparse.ArgumentParser(description='AVION finetune ek100 mir', add_help=False)
    parser.add_argument('--dataset', default='ek100_mir', type=str, choices=['ek100_mir'])
    parser.add_argument('--root', default='datasets/EK100/EK100_256p_15sec/', type=str, help='path to train dataset root')
    parser.add_argument('--train-metadata', type=str,
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_train.csv')
    parser.add_argument('--val-metadata', type=str,
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv')
    parser.add_argument('--relevancy-path', type=str,
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--video-chunk-length', default=15, type=int)
    parser.add_argument('--clip-length', default=16, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=4, type=int, help='clip stride')
    parser.add_argument('--norm-style', default='openai', type=str, choices=['openai', 'timm'])
    parser.add_argument('--fused-decode-crop', action='store_true', dest='fused_decode_crop')
    parser.add_argument('--no-fused-decode-crop', action='store_false', dest='fused_decode_crop')
    parser.set_defaults(fused_decode_crop=False)
    parser.add_argument('--decode-threads', default=1, type=int)
    parser.add_argument('--use-multi-epochs-loader', action='store_true')
    # model
    parser.add_argument('--model', default='CLIP_VITB16', type=str)
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
    parser.add_argument('--drop-path-rate', default=0., type=float)
    parser.add_argument('--pretrain-model', default='', type=str, help='path of pretrained model')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    # clip loss
    parser.add_argument('--local-loss', action='store_true')
    parser.add_argument('--gather-with-grad', action='store_true', dest='gather_with_grad')
    parser.add_argument('--no-gather-with-grad', action='store_false', dest='gather_with_grad')
    parser.set_defaults(gather_with_grad=True)
    # training
    parser.add_argument('--use-zero', action='store_true', dest='use_zero', help='use ZeRO optimizer')
    parser.add_argument('--no-use-zero', action='store_false', dest='use_zero', help='use ZeRO optimizer')
    parser.set_defaults(use_zero=False)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=16, type=int, help='number of samples per-device/per-gpu')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float, help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float, help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int, help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.01, type=float)
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

    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)

    criterion = MaxMarginRankingLoss(
        margin=0.2,
        fix_norm=True,
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        rank=args.rank,
        world_size=args.world_size,
    ).cuda(args.gpu)

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


    opt_fn = torch.optim.AdamW
    if args.use_zero:
        print('Training with ZeroRedundancyOptimizer')
        optimizer = ZeroRedundancyOptimizer(
            optim_params, optimizer_class=opt_fn,
            lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.wd
        )
    else:
        optimizer = opt_fn(optim_params, lr=args.lr, betas=args.betas,
                           eps=args.eps, weight_decay=args.wd)
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

    tokenizer = partial(tokenize, context_length=old_args.context_length)
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

    train_dataset = VideoCaptionDatasetCLIP(
        args.dataset, args.root, args.train_metadata,
        transform=train_transform, is_training=True, tokenizer=tokenizer,
        clip_length=args.clip_length, clip_stride=args.clip_stride,
        chunk_len=args.video_chunk_length,
        threads=args.decode_threads,
        fast_rrc=args.fused_decode_crop, rrc_params=(crop_size, (0.5, 1.0)),
    )

    val_dataset = VideoCaptionDatasetCLIP(
        args.dataset, args.root, args.val_metadata,
        transform=val_transform, is_training=False, tokenizer=tokenizer,
        clip_length=args.clip_length, clip_stride=args.clip_stride,
        chunk_len=args.video_chunk_length,
        fast_rcc=args.fused_decode_crop, rcc_params=(crop_size,),
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
        val_stats = validate_mir(val_loader, val_transform_gpu, model, criterion, args)
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

        # train for one epoch
        train_stats = train(train_loader, train_transform_gpu, model, criterion, optimizer, scaler, epoch, lr_schedule, args)

        if (epoch + 1) % args.eval_freq != 0:
            continue

        val_stats = validate_mir(val_loader, val_transform_gpu, model, criterion, args)
        acc1 = val_stats['avg_map']

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

def train(train_loader, transform_gpu, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    model_time = AverageMeter('Model', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss', 'max_margin_loss']
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, model_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    if args.update_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []


    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None:
                param_group['lr'] = lr_schedule[it]

        uids = inputs[0]
        inputs = inputs[1:]
        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
        relevancies = inputs.pop()  # loader will a "relevancy" variable; we need it for ek100_mir
        optimizer.zero_grad()

        tic = time.time()
        # compute output
        if args.update_freq == 1:
            with amp.autocast(enabled=not args.disable_amp):
                if args.fused_decode_crop and len(transform_gpu) > 0:
                    inputs[0] = inputs[0].permute(0, 4, 1, 2, 3)
                    inputs[0] = transform_gpu(inputs[0])
                image_features, text_features, logit_scale = model(*inputs)
                loss_dict = criterion(image_features, text_features, weight=relevancies)
                loss = loss_dict['loss']
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with amp.autocast(enabled=not args.disable_amp):
                    chunk_image_features, chunk_text_features, _ = model(images, texts)
                accum_image_features.append(chunk_image_features)
                accum_text_features.append(chunk_text_features)

                accum_images.append(images)
                accum_texts.append(texts)

            # If non-zero, move on to the next batch.
            if ((data_iter + 1) % args.update_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with amp.autocast(enabled=not args.disable_amp):
                    chunk_image_features, chunk_text_features, logit_scale = model(images, texts)
                    image_features = torch.cat(
                        accum_image_features[:j] + [chunk_image_features] + accum_image_features[j + 1:])
                    text_features = torch.cat(
                        accum_text_features[:j] + [chunk_text_features] + accum_text_features[j + 1:])
                    loss_dict = criterion(image_features, text_features, weight=relevancies)
                    loss = loss_dict['loss']

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

        dist_utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = dist_utils.get_model(model).logit_scale.exp().item()

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            progress.display(optim_iter)
    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


def validate_mir(val_loader, transform_gpu, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss', 'max_margin_loss']
    iters_per_epoch = len(val_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Test: "
    )

    # switch to eval mode
    model.eval()

    all_video_embed = [[] for _ in range(args.world_size)]
    all_text_embed = [[] for _ in range(args.world_size)]
    total_num = 0
    with amp.autocast(enabled=not args.disable_amp):
        with torch.no_grad():
            end = time.time()
            for i, inputs in enumerate(val_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                uids = inputs[0]
                inputs = inputs[1:]
                inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
                relevancies = inputs.pop()

                # compute output
                if args.fused_decode_crop and len(transform_gpu) > 0:
                    inputs[0] = inputs[0].permute(0, 4, 1, 2, 3)
                    inputs[0] = transform_gpu(inputs[0])
                image_features, text_features, logit_scale = model(*inputs)
                gathered_image_features = [torch.zeros_like(image_features) for _ in range(args.world_size)]
                gathered_text_features = [torch.zeros_like(text_features) for _ in range(args.world_size)]
                torch.distributed.all_gather(gathered_image_features, image_features)
                torch.distributed.all_gather(gathered_text_features, text_features)
                for j in range(args.world_size):
                    all_video_embed[j].append(gathered_image_features[j].detach().cpu())
                    all_text_embed[j].append(gathered_text_features[j].detach().cpu())
                loss_dict = criterion(image_features, text_features, weight=relevancies)

                for k in loss_dict:
                    metrics[k].update(loss_dict[k].item(), args.batch_size)

                total_num += image_features.shape[0] * args.world_size

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                mem.update(torch.cuda.max_memory_allocated() // 1e9)

                if i % args.print_freq == 0:
                    progress.display(i)
    progress.synchronize()
    for j in range(args.world_size):
        all_video_embed[j] = torch.cat(all_video_embed[j], dim=0).numpy()
        all_text_embed[j] = torch.cat(all_text_embed[j], dim=0).numpy()
    all_text_embed_reorg, all_video_embed_reorg = [], []
    for i in range(total_num):
        all_video_embed_reorg.append(all_video_embed[i % args.world_size][i // args.world_size])
        all_text_embed_reorg.append(all_text_embed[i % args.world_size][i // args.world_size])
    all_text_embed = np.vstack(all_text_embed_reorg)
    all_video_embed = np.vstack(all_video_embed_reorg)
    all_text_embed = all_text_embed[:9668, :]
    all_video_embed = all_video_embed[:9668, :]
    similarity_matrix = np.matmul(all_video_embed, all_text_embed.T)
    similarity_matrix = (similarity_matrix + 1) / 2
    video_id = pd.read_csv(args.val_metadata).values[:, 0]
    text_id = pd.read_csv(args.val_metadata.replace('test', 'test_sentence')).values[:, 0]
    indexes = [video_id.tolist().index(elem) for elem in text_id]
    similarity_matrix = similarity_matrix[:, indexes]
    print(similarity_matrix.shape)
    rel_matrix = pd.read_pickle(args.relevancy_path)
    vis_map, txt_map, avg_map = get_mAP(similarity_matrix, rel_matrix)
    print('mAP: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_map, txt_map, avg_map))
    vis_nDCG, txt_nDCG, avg_nDCG = get_nDCG(similarity_matrix, rel_matrix)
    print('nDCG: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_nDCG, txt_nDCG, avg_nDCG))
    return {**{k: v.avg for k, v in metrics.items()},
            'vis_map': vis_map, 'txt_map': txt_map, 'avg_map': avg_map,
            'vis_ndcg': vis_nDCG, 'txt_ndcg': txt_nDCG, 'avg_ndcg': avg_nDCG}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LAVILA training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
