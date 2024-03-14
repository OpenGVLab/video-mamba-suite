import torch
import torch.nn as nn

from torchvision.models.video import r2plus1d_18 as _r2plus1d_18
from torchvision.models.video import r3d_18 as _r3d_18
from torchvision.models.video.resnet import VideoResNet, R2Plus1dStem, BasicBlock

__all__ = ['r2plus1d_34', 'r2plus1d_18', 'r3d_18']

R2PLUS1D_34_MODEL_URL="https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ft_kinetics_from_ig65m-0aa0550b.pth"


def r2plus1d_34(pretrained=True, progress=False, **kwargs):
    model = VideoResNet(
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 4, 6, 3],
        stem=R2Plus1dStem,
        **kwargs,
    )

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            R2PLUS1D_34_MODEL_URL, progress=progress
        )
        model.load_state_dict(state_dict)

    return model


def r2plus1d_18(pretrained=True, progress=False, **kwargs):
    return _r2plus1d_18(pretrained=pretrained, progress=progress, **kwargs)


def r3d_18(pretrained=True, progress=False, **kwargs):
    return _r3d_18(pretrained=pretrained, progress=progress, **kwargs)


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):

        midplanes = (in_planes * out_planes * 3 * 3 * 3) // (
            in_planes * 3 * 3 + 3 * out_planes
        )
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes,
                out_planes,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
            ),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)