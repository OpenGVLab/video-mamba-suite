import torch
import torch.nn as nn


class PuppetCaptionModel(nn.Module):
    def __init__(self, opt):
        super(PuppetCaptionModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.opt = opt
        self.puppet_layer= nn.Linear(1,1)

    def forward(self, event, clip, clip_mask, seq):
        N, L = seq.shape
        output = torch.zeros((N, L-1, self.vocab_size + 1), device=seq.device)
        return output

    def sample(self, event, clip, clip_mask, opt={}):
        N, _, C = clip.shape
        output = torch.zeros((N, 3), device=clip.device)
        prob = torch.zeros((N, 3), device=clip.device)
        return output, prob

    def build_loss(self, input, target, mask):
        one_hot = torch.nn.functional.one_hot(target, self.opt.vocab_size+1)
        output = - (one_hot * input * mask[..., None]).sum(2).sum(1) / (mask.sum(1) + 1e-6)
        return output