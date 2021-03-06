from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import one_hot_embedding
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, num_classes=80):
        '''
        
        Args:
          num_classes: (int) number of classes
        '''
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes


    def focal_loss(self, x, y, alpha=0.25, gamma=2):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
          alpha: (float) weight for rare class, 0.25 works good.
          gamma: (float) suppression of good result, 2 works good.

        Return:
          (tensor) focal loss.
        '''
        t = one_hot_embedding(y.data.cpu(), 1 + self.num_classes)  # [N,#cls + 1]
        t = t[:, 1:]                           # exclude background
        t = Variable(t).cuda()                 # [N,#cls]

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)         # pt = p if t > 0 else 1-p

        w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1 - pt).pow(gamma)
        #  w = Variable(w).cuda()
        #  return F.binary_cross_entropy_with_logits(x, t, w, reduction='sum')
        loss = -w * pt.log()
        return loss.sum()

    def focal_loss_alt(self, x, y, alpha=0.25):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        t = one_hot_embedding(y.data.cpu(), 1 + self.num_classes)
        t = t[:, 1:]
        t = Variable(t).cuda()

        xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
        pt = (2 * xt + 1).sigmoid()

        w = alpha * t + (1 - alpha) * (1 - t)
        loss = -w * pt.log() / 2
        return loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)     # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4) # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets,
                reduction='sum') / num_pos

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds,
                cls_targets[pos_neg]) / num_pos

        loss = loc_loss + cls_loss

        # Manage NaN loss exception
        # TODO This code is not optimized
        if num_pos.item() == 0:
            return (Variable(torch.tensor(0).type_as(loss.data), requires_grad=True),
                    Variable(torch.tensor(0).type_as(loc_loss.data), requires_grad=True),
                    Variable(torch.tensor(0).type_as(cls_loss.data), requires_grad=True))
        else:
            return loss, loc_loss, cls_loss
