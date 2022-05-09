from cmath import cos
import enum
from typing import List
from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import gorilla
import torch_scatter
from typing import Optional


@torch.jit.script
def batch_sigmoid_bce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: (num_querys, N)
        targets: (num_inst, N)
    Returns:
        Loss tensor
    """
    N = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction="none")
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction="none")

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))

    return loss / N


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss


def batch_center_loss(out_center, tgt_center):
    '''
    out_center: (num_query, 3)
    tgt_center: (num_inst, 3)
    '''
    loss = torch.norm(out_center[:, None] - tgt_center[None, :], dim=-1)
    return loss


def get_iou(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


@torch.jit.script
def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)  # 为什么这里是+1？
    return loss.mean()


@torch.jit.script
def dice_loss_multi_calsses(
    input: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-5, weight: Optional[float] = None
) -> torch.Tensor:
    r"""
    modify compute_per_channel_dice from https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py
    """
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # convert the feature channel(category channel) as first
    # axis_order = (1, 0) + tuple(range(2, input.dim()))
    input = input.permute(1, 0)
    target = target.permute(1, 0)

    target = target.float()
    # Compute per channel Dice Coefficient
    per_channel_dice = (2 * torch.sum(input * target, dim=1) + epsilon) / (
        torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4 + epsilon
    )

    loss = 1.0 - per_channel_dice

    return loss.mean()


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_weight):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.register_buffer('cost_weight', torch.tensor(cost_weight))

    @torch.no_grad()
    def forward(self, pred_labels, pred_masks, insts):
        '''
        pred_masks: List[Tensor] len(p2c) == B, Tensor.shape == (n, N)
        pred_labels: (B, n_q, 19)
        insts: List[Instances3D]
        '''
        indices = []
        for pred_label, pred_mask, inst in zip(pred_labels, pred_masks, insts):
            if len(inst) == 0:
                indices.append(([], []))
                continue
            pred_label = pred_label.softmax(-1)  # (n_q, 19)
            tgt_idx = inst.gt_labels  # (num_inst,)
            cost_class = -pred_label[:, tgt_idx]  # (n_q, num_inst)

            tgt_mask = inst.gt_spmasks  # (num_inst, N)

            cost_mask = batch_sigmoid_bce_loss(pred_mask, tgt_mask.float())
            cost_dice = batch_dice_loss(pred_mask, tgt_mask.float())

            # cost_center = batch_center_loss(out_center, tgt_center)
            C = (
                self.cost_weight[0] * cost_class
                + self.cost_weight[1] * cost_mask
                + self.cost_weight[2] * cost_dice
                # + self.cost_weight[3] * cost_center
            )
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# @gorilla.LOSSES.register_module()
class SpfCriterion(nn.Module):
    def __init__(
        self,
        ignore_label=-100,
        loss_weight=[1.0, 1.0, 1.0, 1.0, 1.0],
        cost_weight=[1.0, 1.0, 1.0],
        non_object_weight=0.1,
        num_classes=18,
    ):
        super().__init__()
        class_weight = torch.ones(num_classes + 1)
        class_weight[-1] = non_object_weight
        self.register_buffer('class_weight', class_weight)
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)
        self.matcher = HungarianMatcher(cost_weight)
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        # self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_inst_info(self, batched_gt_instance, coords, batch_offsets):
        for i, gt_inst in enumerate(batched_gt_instance):
            start_id = batch_offsets[i]
            end_id = batch_offsets[i + 1]
            coord = coords[start_id:end_id]  # (N, 3)
            inst_idx, point_idx = torch.nonzero(gt_inst['gt_masks'], as_tuple=True)
            inst_point = coord[point_idx]
            gt_inst['gt_center'] = torch_scatter.segment_coo(inst_point, inst_idx.cuda(), reduce='mean')

    def get_layer_loss(self, layer, aux_outputs, insts):
        loss_out = {}
        pred_labels = aux_outputs['labels']
        pred_masks = aux_outputs['masks']
        indices = self.matcher(pred_labels, pred_masks, insts)
        idx = self._get_src_permutation_idx(indices)

        # class loss
        tgt_class_o = torch.cat([inst.gt_labels[idx_gt] for inst, (_, idx_gt) in zip(insts, indices)])
        tgt_class = torch.full(
            pred_labels.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=pred_labels.device,
        )  # (B, num_query)
        tgt_class[idx] = tgt_class_o
        class_loss = F.cross_entropy(pred_labels.transpose(1, 2), tgt_class, self.class_weight)

        loss_out[f'layer_{layer}_cls_loss'] = class_loss * self.loss_weight[0]

        # # score loss
        # score_loss = torch.tensor([0.0], device=pred_sem.device)

        # mask loss
        mask_bce_loss = torch.tensor([0.0], device=pred_labels.device)
        mask_dice_loss = torch.tensor([0.0], device=pred_labels.device)
        for mask, inst, (idx_q, idx_gt) in zip(pred_masks, insts, indices):
            if len(inst) == 0:
                continue
            # pred_score = score[idx_q]
            pred_mask = mask[idx_q]  # (num_inst, N)
            tgt_mask = inst.gt_spmasks[idx_gt]  # (num_inst, N)
            # with torch.no_grad():
            #     tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)
            # score_loss += F.binary_cross_entropy_with_logits(pred_score, tgt_score)
            mask_bce_loss += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            mask_dice_loss += dice_loss(pred_mask, tgt_mask.float())
        # score_loss = score_loss / len(pred_masks)
        mask_bce_loss = mask_bce_loss / len(pred_masks)
        mask_dice_loss = mask_dice_loss / len(pred_masks)

        # loss_out['score_loss'] = score_loss.item()
        loss_out[f'layer_{layer}_mask_bce_loss'] = mask_bce_loss * self.loss_weight[1]
        loss_out[f'layer_{layer}_mask_dice_loss'] = mask_dice_loss * self.loss_weight[2]
        return loss_out

    def forward(self, pred, insts):
        '''
        pred_masks: List[Tensor (n, M)] len(p2c) == B
        pred_labels: (B, n, 19)
        pred_scores: (B, n, 1) or [(B, n, 1)]
        pred_cls: (B, n, 19)
        pred_sem: (B*N, 20)
        cluster_coords: (B, n, 3)
        point_offsets: (B*N, 3)
        sem: (B*N, )
        insts: List[Instance3D]

        coords: (B*N, 3)
        batched_gt_instance: List of Dicts
        instance_info: (B*N, 9), float32 tensor (meanxyz, minxyz, maxxyz)
        instance_labels: (B*N), long
        '''
        loss_out = {}

        # pred_offsets = pred['offsets']
        # pred_sem = pred['sem']

        pred_labels = pred['labels']
        pred_masks = pred['masks']

        # semantic_loss = self.semantic_criterion(pred_sem, sem)
        # filter_ids = sem != self.ignore_label
        # pred_sem = pred_sem[filter_ids]
        # pred_sem = F.softmax(pred_sem, dim=-1)
        # sem = sem[filter_ids]
        # one_hot_labels = F.one_hot(sem, num_classes=20)
        # semantic_loss += dice_loss_multi_calsses(pred_sem, one_hot_labels).mean()
        # loss_out["semantic_loss"] = semantic_loss.item()

        # gt_offsets = pt_center[:, 0:3] - xyz  # [N, 3]
        # pt_diff = pred_offsets - gt_offsets  # [N, 3]
        # pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)  # [N] 这里用的是L1 Loss？
        # valid = pt_center[:, 0] != self.ignore_label

        # offset_norm_loss = torch.sum(pt_dist * valid) / (valid.sum() + 1e-6)

        # gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)  # [N], float
        # gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        # pt_offsets_norm = torch.norm(pred_offsets, p=2, dim=1)
        # pt_offsets_ = pred_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        # direction_diff = -(gt_offsets_ * pt_offsets_).sum(-1)  # [N]
        # offset_dir_loss = torch.sum(direction_diff * valid) / (valid.sum() + 1e-6)

        # offset_loss = offset_norm_loss + offset_dir_loss
        # loss_out["offset_loss"] = offset_loss.item()

        # match
        # List of Tuple,len is B, (idx of query, idx of gt)
        indices = self.matcher(pred_labels, pred_masks, insts)
        idx = self._get_src_permutation_idx(indices)

        # class loss
        tgt_class_o = torch.cat([inst.gt_labels[idx_gt] for inst, (_, idx_gt) in zip(insts, indices)])
        tgt_class = torch.full(
            pred_labels.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=pred_labels.device,
        )  # (B, num_query)
        tgt_class[idx] = tgt_class_o
        class_loss = F.cross_entropy(pred_labels.transpose(1, 2), tgt_class, self.class_weight)

        loss_out['cls_loss'] = class_loss * self.loss_weight[0]

        # # score loss
        # score_loss = torch.tensor([0.0], device=pred_sem.device)

        # mask loss
        mask_bce_loss = torch.tensor([0.0], device=pred_labels.device)
        mask_dice_loss = torch.tensor([0.0], device=pred_labels.device)
        for mask, inst, (idx_q, idx_gt) in zip(pred_masks, insts, indices):
            if len(inst) == 0:
                continue
            # pred_score = score[idx_q]
            pred_mask = mask[idx_q]  # (num_inst, N)
            tgt_mask = inst.gt_spmasks[idx_gt]  # (num_inst, N)
            # with torch.no_grad():
            #     tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)
            # score_loss += F.binary_cross_entropy_with_logits(pred_score, tgt_score)
            mask_bce_loss += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            mask_dice_loss += dice_loss(pred_mask, tgt_mask.float())
        # score_loss = score_loss / len(pred_masks)
        mask_bce_loss = mask_bce_loss / len(pred_masks)
        mask_dice_loss = mask_dice_loss / len(pred_masks)

        # loss_out['score_loss'] = score_loss.item()
        loss_out['mask_bce_loss'] = mask_bce_loss * self.loss_weight[1]
        loss_out['mask_dice_loss'] = mask_dice_loss * self.loss_weight[2]

        if "aux_outputs" in pred:
            for i, aux_outputs in enumerate(pred["aux_outputs"]):
                loss_out_i = self.get_layer_loss(i, aux_outputs, insts)
                loss_out.update(loss_out_i)
        return loss_out
