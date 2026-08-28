# BCE + Dice loss with deep supervision, adapted from the official
# implementation (utils.py: BCELoss/DiceLoss/BceDiceLoss/GT_BceDiceLoss).
#
# Source: https://github.com/JCruan519/EGE-UNet (Apache-2.0)
# Commit: f52ba30c6bf7d0ca479c2c9d4a3cbda999f49d3a
#
# Deviation from the official code, documented explicitly per task
# instructions: the official GT_BceDiceLoss hardcodes the deep-supervision
# weights inline (0.1, 0.2, 0.3, 0.4, 0.5 for gt_pre5..gt_pre1) instead of
# taking them as a parameter. Those five numbers are algebraically identical
# to the paper's lambda = [1, 0.5, 0.4, 0.3, 0.2, 0.1] (stage 0 = final
# output, weight 1; gt_pre1 nearest the output gets 0.5, ..., gt_pre5
# deepest gets 0.1) -- so no numeric change, just made the weights an
# explicit, checkable parameter instead of inline constants.

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        return self.bceloss(pred.view(size, -1), target.view(size, -1))


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + self.smooth) / (
            pred_.sum(1) + target_.sum(1) + self.smooth
        )
        return 1 - dice_score.sum() / size


class BceDiceLoss(nn.Module):
    """wb * BCE + wd * Dice, both weights default to 1 as in the paper/official code."""

    def __init__(self, wb=1, wd=1):
        super().__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        return self.wd * self.dice(pred, target) + self.wb * self.bce(pred, target)


class GTBceDiceLoss(nn.Module):
    """
    Deep-supervision BCE+Dice loss matching EGE-UNet paper Eq. for L_total:

        L = lambda_0 * L(out, gt) + sum_{i=1..5} lambda_i * L(gt_pre_i, gt)

    with lambda = [1, 0.5, 0.4, 0.3, 0.2, 0.1] (stage 0 = final output,
    stage 1 = shallowest supervision head nearest the output, ...,
    stage 5 = deepest supervision head), per the task spec and matching
    the official GT_BceDiceLoss's hardcoded 0.5/0.4/0.3/0.2/0.1 weights.

    model.forward() returns gt_pre = (gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1)
    i.e. deepest-first; ds_weights below is indexed to match that order.
    """

    def __init__(self, wb=1, wd=1, ds_weights=(0.1, 0.2, 0.3, 0.4, 0.5), final_weight=1.0):
        super().__init__()
        assert len(ds_weights) == 5, "EGE-UNet has exactly 5 deep-supervision heads"
        self.bcedice = BceDiceLoss(wb, wd)
        self.ds_weights = ds_weights
        self.final_weight = final_weight

    def forward(self, gt_pre, out, target):
        loss = self.final_weight * self.bcedice(out, target)
        for pred_i, w_i in zip(gt_pre, self.ds_weights):
            loss = loss + w_i * self.bcedice(pred_i, target)
        return loss
