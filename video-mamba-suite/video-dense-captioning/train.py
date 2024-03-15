# coding:utf-8
import numpy as np
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import json
import time
import torch
import os
import sys
import collections
# import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from os.path import dirname, abspath

pdvc_dir = dirname(abspath(__file__))
sys.path.insert(0, pdvc_dir)
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3'))
sys.path.insert(0, os.path.join(pdvc_dir, 'densevid_eval3/SODA'))
# print(sys.path)

from eval_utils import evaluate
import opts
from tensorboardX import SummaryWriter
from misc.utils import print_alert_message, build_floder, create_logger, backup_envir, print_opt, set_seed
from data.video_dataset import PropSeqDataset, collate_fn
from pdvc.pdvc import build
from collections import OrderedDict
import distributed as dist_utils
from ipdb import set_trace

def train(opt):
    args = opt
    set_seed(opt.seed)
    
    save_folder = build_floder(opt)    
    logger = create_logger(save_folder, 'train.log')
        
    dist_utils.init_distributed_mode(opt)

    if dist_utils.is_main_process() and not os.path.exists(os.path.join(save_folder, 'tf_summary')):
        tf_writer = SummaryWriter(os.path.join(save_folder, 'tf_summary'))


    if dist_utils.is_main_process() and not opt.start_from:
        backup_envir(save_folder)
        logger.info('backup evironment completed !')

    saved_info = {'best': {}, 'last': {}, 'history': {}, 'eval_history': {}}

    # continue training
    if opt.start_from:
        opt.pretrain = False
        infos_path = os.path.join(save_folder, 'info.json')
        with open(infos_path) as f:
            if dist_utils.is_main_process():
                logger.info('Load info from {}'.format(infos_path))
            saved_info = json.load(f)
            prev_opt = saved_info[opt.start_from_mode[:4]]['opt']

            exclude_opt = ['start_from', 'start_from_mode', 'pretrain']
            for opt_name in prev_opt.keys():
                if opt_name not in exclude_opt:
                    vars(opt).update({opt_name: prev_opt.get(opt_name)})
                if prev_opt.get(opt_name) != vars(opt).get(opt_name):
                    if dist_utils.is_main_process():
                        logger.info('Change opt {} : {} --> {}'.format(opt_name, prev_opt.get(opt_name),
                                                                    vars(opt).get(opt_name)))

    train_dataset = PropSeqDataset(opt.train_caption_file,
                                   opt.visual_feature_folder,
                                   opt.dict_file, True, 'gt',
                                   opt)

    val_dataset = PropSeqDataset(opt.val_caption_file,
                                 opt.visual_feature_folder,
                                 opt.dict_file, False, 'gt',
                                 opt)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, sampler=train_sampler,
                              shuffle=train_sampler is None, num_workers=opt.nthreads, collate_fn=collate_fn, drop_last=True)
    
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size_for_eval, sampler=val_sampler,
                            shuffle=False, num_workers=opt.nthreads, collate_fn=collate_fn)

    print('Len train loader:', len(train_loader))
    print('Len val loader:', len(val_loader))

    epoch = saved_info[opt.start_from_mode[:4]].get('epoch', 0)
    iteration = saved_info[opt.start_from_mode[:4]].get('iter', 0)
    best_val_score = saved_info[opt.start_from_mode[:4]].get('best_val_score', -1e5)
    val_result_history = saved_info['history'].get('val_result_history', {})
    loss_history = saved_info['history'].get('loss_history', {})
    lr_history = saved_info['history'].get('lr_history', {})
    opt.current_lr = vars(opt).get('current_lr', opt.lr)

    print('Building model')
    # Build model
    model, criterion, postprocessors = build(opt)
    model.translator = train_dataset.translator
    model.train()

    model.to(opt.device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu], bucket_cap_mb=200, find_unused_parameters=True)

    # Recover the parameters
    if opt.start_from and (not opt.pretrain):
        if opt.start_from_mode == 'best':
            model_pth = torch.load(os.path.join(save_folder, 'model-best.pth'))
        elif opt.start_from_mode == 'last':
            model_pth = torch.load(os.path.join(save_folder, 'model-last.pth'))
        
        if dist_utils.is_main_process():
            logger.info('Loading pth from {}, iteration:{}'.format(save_folder, iteration))
            
        model.load_state_dict(model_pth['model'])

    # Load the pre-trained model
    if opt.pretrain and (not opt.start_from):
        if dist_utils.is_main_process():
            logger.info('Load pre-trained parameters from {}'.format(opt.pretrain_path))
        model_pth = torch.load(opt.pretrain_path, map_location=torch.device(opt.device))
        # query_weight = model_pth['model'].pop('query_embed.weight')
        if opt.pretrain == 'encoder':
            encoder_filter = model.get_filter_rule_for_encoder()
            encoder_pth = {k:v for k,v in model_pth['model'].items() if encoder_filter(k)}
            model.load_state_dict(encoder_pth, strict=True)
        elif opt.pretrain == 'decoder':
            encoder_filter = model.get_filter_rule_for_encoder()
            decoder_pth = {k:v for k,v in model_pth['model'].items() if not encoder_filter(k)}
            model.load_state_dict(decoder_pth, strict=True)
            pass
        elif opt.pretrain == 'full':
            # model_pth = transfer(model, model_pth)
            model.load_state_dict(model_pth['model'], strict=True)
        else:
            raise ValueError("wrong value of opt.pretrain")

    # model.to(opt.device)
    print('Begin optimizer')

    if opt.optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    elif opt.optimizer_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    milestone = [opt.learning_rate_decay_start + opt.learning_rate_decay_every * _ for _ in range(int((opt.epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every))]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestone, gamma=opt.learning_rate_decay_rate)

    if opt.start_from:
        optimizer.load_state_dict(model_pth['optimizer'])
        lr_scheduler.step(epoch-1)

    # print the args for debugging
    print_opt(opt, model, logger)
    print_alert_message('Strat training !', logger)

    loss_sum = OrderedDict()
    bad_video_num = 0

    start = time.time()

    weight_dict = criterion.weight_dict
    if dist_utils.is_main_process():
        logger.info('loss type: {}'.format(weight_dict.keys()))
        logger.info('loss weights: {}'.format(weight_dict.values()))


    print('Begin training')
    # Epoch-level iteration
    while True:
        if True:
            # scheduled sampling rate update
            if epoch > opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.basic_ss_prob + opt.scheduled_sampling_increase_prob * frac,
                                  opt.scheduled_sampling_max_prob)
                model.caption_head.ss_prob = opt.ss_prob

            print('lr:{}'.format(float(opt.current_lr)))
            pass

        
        # Batch-level iteration
        for dt in tqdm(train_loader, disable=opt.disable_tqdm):
            if opt.device=='cuda':
                torch.cuda.synchronize(opt.device)
            if opt.debug:
                # each epoch contains less mini-batches for debugging
                if (iteration + 1) % 5 == 0:
                    iteration += 1
                    break
            iteration += 1

            optimizer.zero_grad()
            dt = {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in dt.items()}
            dt['video_target'] = [
                {key: _.to(opt.device) if isinstance(_, torch.Tensor) else _ for key, _ in vid_info.items()} for vid_info in
                dt['video_target']]

            # dt = collections.defaultdict(lambda: None, dt)
            # print(dt.keys())
            # set_trace()

            output, loss = model(dt, criterion, opt.transformer_input_type)
            # loss = model(dt, criterion, opt.transformer_input_type)

            final_loss = sum(loss[k] * weight_dict[k] for k in loss.keys() if k in weight_dict)
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

            optimizer.step()

            for loss_k,loss_v in loss.items():
                loss_sum[loss_k] = loss_sum.get(loss_k, 0)+ loss_v.item()
            loss_sum['total_loss'] = loss_sum.get('total_loss', 0) + final_loss.item()

            if opt.device=='cuda':
                torch.cuda.synchronize()

            losses_log_every = int(len(train_loader) / 10)

            if opt.debug:
                losses_log_every = 6

            if iteration % losses_log_every == 0:
                end = time.time()
                for k in loss_sum.keys():
                    loss_sum[k] = np.round(loss_sum[k] /losses_log_every, 3).item()

                if dist_utils.is_main_process():
                    logger.info(
                        "ID {} iter {} (epoch {}), \nloss = {}, \ntime/iter = {:.3f}, bad_vid = {:.3f}"
                            .format(opt.id, iteration, epoch, loss_sum,
                                    (end - start) / losses_log_every, bad_video_num))
                    
                    tf_writer.add_scalar('lr', opt.current_lr, iteration)
                    for loss_type in loss_sum.keys():
                        tf_writer.add_scalar(loss_type, loss_sum[loss_type], iteration)

                loss_history[iteration] = loss_sum
                lr_history[iteration] = opt.current_lr
                loss_sum = OrderedDict()
                start = time.time()
                bad_video_num = 0
                torch.cuda.empty_cache()
        
        
        # evaluation
        if (epoch % opt.save_checkpoint_every == 0) and (epoch >= opt.min_epoch_when_save):

            # Save model
            saved_pth = {'epoch': epoch,
                         'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(), }

            if opt.save_all_checkpoint:
                checkpoint_path = os.path.join(save_folder, 'model_iter_{}.pth'.format(iteration))
            else:
                checkpoint_path = os.path.join(save_folder, 'model-last.pth')

            if dist_utils.is_main_process():
                torch.save(saved_pth, checkpoint_path)

            model.eval()
            result_json_path = os.path.join(save_folder, 'prediction',
                                         'num{}_epoch{}.json'.format(
                                             len(val_dataset), epoch))


            eval_score, eval_loss = evaluate(opt, model, criterion, postprocessors, val_loader, result_json_path, logger=logger, alpha=opt.ec_alpha, device=opt.device, debug=opt.debug)
            
            
            if opt.caption_decoder_type == 'none':
                current_score = 2./(1./eval_score['Precision'] + 1./eval_score['Recall'])
            else:
                if opt.criteria_for_best_ckpt == 'dvc':
                    current_score = np.array(eval_score['METEOR']).mean() + np.array(eval_score['soda_c']).mean()
                else:
                    current_score = np.array(eval_score['para_METEOR']).mean() + np.array(eval_score['para_CIDEr']).mean() + np.array(eval_score['para_Bleu_4']).mean()

            # add to tf summary
            if dist_utils.is_main_process():
                for key in eval_score.keys():
                    tf_writer.add_scalar(key, np.array(eval_score[key]).mean(), iteration)

                for loss_type in eval_loss.keys():
                    tf_writer.add_scalar('eval_' + loss_type, eval_loss[loss_type], iteration)

            _ = [item.append(np.array(item).mean()) for item in eval_score.values() if isinstance(item, list)]
            print_info = '\n'.join([key + ":" + str(eval_score[key]) for key in eval_score.keys()])
            if dist_utils.is_main_process():
                logger.info('\nValidation results of iter {}:\n'.format(iteration) + print_info)
                logger.info('\noverall score of iter {}: {}\n'.format(iteration, current_score))
            val_result_history[epoch] = {'eval_score': eval_score}
            
            if dist_utils.is_main_process():
                logger.info('Save model at iter {} to {}.'.format(iteration, checkpoint_path))

            # save the model parameter and  of best epoch
            if current_score >= best_val_score:
                best_val_score = current_score
                best_epoch = epoch
                saved_info['best'] = {'opt': vars(opt),
                                      'iter': iteration,
                                      'epoch': best_epoch,
                                      'best_val_score': best_val_score,
                                      'result_json_path': result_json_path,
                                      'avg_proposal_num': eval_score['avg_proposal_number'],
                                      'Precision': eval_score['Precision'],
                                      'Recall': eval_score['Recall']
                                      }

                # suffix = "RL" if sc_flag else "CE"
                if dist_utils.is_main_process():
                    torch.save(saved_pth, os.path.join(save_folder, 'model-best.pth'))
                    logger.info('Save Best-model at iter {} to checkpoint file.'.format(iteration))

            saved_info['last'] = {'opt': vars(opt),
                                  'iter': iteration,
                                  'epoch': epoch,
                                  'best_val_score': best_val_score,
                                  }
            saved_info['history'] = {'val_result_history': val_result_history,
                                     'loss_history': loss_history,
                                     'lr_history': lr_history,
                                     # 'query_matched_fre_hist': query_matched_fre_hist,
                                     }

            if dist_utils.is_main_process():                                     
                with open(os.path.join(save_folder, 'info.json'), 'w') as f:
                    json.dump(saved_info, f)
                logger.info('Save info to info.json')

            model.train()

        epoch += 1
        lr_scheduler.step()
        opt.current_lr = optimizer.param_groups[0]['lr']
        torch.cuda.empty_cache()
        # Stop criterion
        if epoch >= opt.epoch:
            if dist_utils.is_main_process():
                tf_writer.close()
                break

    return saved_info


if __name__ == '__main__':
    opt = opts.parse_opts()
    if opt.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in opt.gpu_id])
    if opt.disable_cudnn:
        torch.backends.cudnn.enabled = False

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # to avoid OMP problem on macos
    train(opt)
