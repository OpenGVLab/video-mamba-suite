# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from misc.detr_utils.box_ops import box_cl_to_xy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1,
                 cost_alpha = 0.25,
                 cost_gamma = 2):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        # self.cost_caption = cost_caption
        self.cost_alpha = cost_alpha
        self.cost_gamma = cost_gamma

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_caption!=0, "all costs cant be 0"

    def forward(self, outputs, targets, verbose=False, many_to_one=False):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            # alpha = 0.25
            alpha = self.cost_alpha
            gamma = self.cost_gamma
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cl_to_xy(out_bbox),
                                             box_cl_to_xy(tgt_bbox))

            # cost_caption = outputs['caption_costs'].flatten(0, 1)

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

            costs = {'cost_bbox': cost_bbox,
                     'cost_class': cost_class,
                     'cost_giou': cost_giou,
                     # 'cost_caption': cost_caption,
                     'out_bbox': out_bbox[:, 0::2]}

            if verbose:
                print('\n')
                print(self.cost_bbox, cost_bbox.var(dim=0), cost_bbox.max(dim=0)[0] - cost_bbox.min(dim=0)[0])
                print(self.cost_class, cost_class.var(dim=0), cost_class.max(dim=0)[0] - cost_class.min(dim=0)[0])
                print(self.cost_giou, cost_giou.var(dim=0), cost_giou.max(dim=0)[0] - cost_giou.min(dim=0)[0])
                # print(self.cost_caption, cost_caption.var(dim=0), cost_caption.max(dim=0)[0] - cost_caption.min(dim=0)[0])

            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            # pdb.set_trace()
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            m2o_rate = 4
            rl_indices = [linear_sum_assignment(torch.cat([c[i]]*m2o_rate, -1)) for i, c in enumerate(C.split(sizes, -1))]
            rl_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j%sizes[ii], dtype=torch.int64)) for ii,(i, j) in
                       enumerate(rl_indices)]

            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

            if verbose:
                print('------matching results:')
                print(indices)
                for indice in indices:
                    for i, j in zip(*indice):
                        print(out_bbox[i][0::2], tgt_bbox[j][0::2])
                print('-----topK scores:')
                topk_indices = out_prob.topk(10, dim=0)
                print(topk_indices)
                for i,(v,ids) in enumerate(zip(*topk_indices)):
                    print('top {}'.format(i))
                    s= ''
                    for name,cost in costs.items():
                        s += name + ':{} '.format(cost[ids])
                    print(s)

            return indices, rl_indices


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou,
                            cost_alpha = args.cost_alpha,
                            cost_gamma = args.cost_gamma
                            )
