import torch
 
from model import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125 # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="50salads")
parser.add_argument('--split', default='1')
parser.add_argument('--model_dir', default='models')
parser.add_argument('--result_dir', default='results')
parser.add_argument('--mamba', action='store_true')
parser.add_argument('--drop_path_rate', type=float, default=0.3)
parser.add_argument('--channel_mask_rate', type=float, default=0.3)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--num_epochs', type=int, default=120)
parser.add_argument('--num_layers', type=int, default=8)
parser.add_argument('--load_epoch', type=int, default=0)
parser.add_argument('--encoder_only', action='store_true')

args = parser.parse_args()
if args.mamba:
    addstr = '_mamba_dp%.2f_l%d_m%.2f_e%d' % (args.drop_path_rate, args.num_layers, args.channel_mask_rate, args.num_epochs)
    # addstr = '_mamba_dp%f' % args.drop_path_rate
else:
    addstr = ''
num_epochs = args.num_epochs
if args.load_epoch == 0:
    load_epoch = num_epochs
else:
    load_epoch = args.load_epoch
print(args)

lr = args.lr
num_layers = args.num_layers
num_f_maps = 256
features_dim = 2048
bz = 1

channel_mask_rate = 0.3


# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads" or args.dataset == "goalsteps":
    sample_rate = 2

# To prevent over-fitting for GTEA. Early stopping & large dropout rate
if args.dataset == "gtea":
    channel_mask_rate = args.channel_mask_rate
    
if args.dataset == 'breakfast':
    lr = 0.0001


vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = "./data/"+args.dataset+"/features/"
gt_path = "./data/"+args.dataset+"/groundTruth/"
 
mapping_file = "./data/"+args.dataset+"/mapping.txt"
 
model_dir = "./{}/".format(args.model_dir)+args.dataset+"/split_"+args.split+addstr

results_dir = "./{}/".format(args.result_dir)+args.dataset+"/split_"+args.split+addstr
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
 
 
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
index2label = dict()
for k,v in actions_dict.items():
    index2label[v] = k
num_classes = len(actions_dict)


trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate, mamba=args.mamba, drop_path_rate=args.drop_path_rate, args=args)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)

    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)

    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)

if args.action == "predict":
    batch_gen_tst = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)
    trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, load_epoch, actions_dict, sample_rate)

