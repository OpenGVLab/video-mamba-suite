# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area

def box_cl_to_xy(x):
    c, l = x.unbind(-1)
    b = [c - 0.5 * l, c + 0.5 * l]
    return torch.stack(b, dim=-1)

def box_xy_to_cl(x):
    x0, x1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (x1 - x0)]
    return torch.stack(b, dim=-1)

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = boxes1[:, 1] - boxes1[:, 0]
    area2 = boxes2[:, 1] - boxes2[:, 0]
    lt = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N,M,2]
    inter = (rb - lt).clamp(min=0)  # [N,M,2]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-5)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 1:] >= boxes1[:, :1]).all()
    assert (boxes2[:, 1:] >= boxes2[:, :1]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    rb = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    area = (rb - lt).clamp(min=0)  # [N,M,2]
    giou = iou - (area - union) / (area + 1e-5)
    return giou