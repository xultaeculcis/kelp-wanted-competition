from typing import Any, Optional

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from kornia.losses import HausdorffERLoss
from torch import Tensor, nn

from kelp import consts


class XEDiceLoss(nn.Module):
    """
    Computes (0.5 * CrossEntropyLoss) + (0.5 * DiceLoss).
    """

    def __init__(
        self,
        mode: str,
        weight_ce: float = 0.5,
        weight_dice: float = 0.5,
        ce_class_weights: Optional[Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.xe = nn.CrossEntropyLoss(weight=ce_class_weights)
        self.dice = smp.losses.DiceLoss(mode=mode)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return self.weight_ce * self.xe(y_pred, y_true) + self.weight_dice * self.dice(y_pred, y_true)


class FocalTverskyLoss(nn.Module):
    """
    Focal-Tversky Loss.

    This loss is similar to Tversky Loss, but with a small adjustment
    With input shape (batch, n_classes, h, w) then TI has shape [batch, n_classes]
    In their paper TI_c is the tensor w.r.t to n_classes index

    References:
        [This paper](https://arxiv.org/pdf/1810.07842.pdf)

        FTL = Sum_index_c(1 - TI_c)^gamma
    """

    def __init__(self, gamma: float = 1.0, beta: float = 0.5, use_softmax: bool = True) -> None:
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.use_softmax = use_softmax

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        num_classes = y_pred.shape[1]
        if self.use_softmax:
            y_pred = F.softmax(y_pred, dim=1)  # predicted value
        y_true = F.one_hot(y_true.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        assert y_pred.shape == y_true.shape
        numerator = torch.sum(y_pred * y_true, dim=(-2, -1))
        denominator = (
            numerator
            + self.beta * torch.sum((1 - y_true) * y_pred, dim=(-2, -1))
            + (1 - self.beta) * torch.sum(y_true * (1 - y_pred), dim=(-2, -1))
        )
        TI = torch.mean((numerator + consts.data.EPS) / (denominator + consts.data.EPS), dim=0)
        return torch.sum(torch.pow(1.0 - TI, self.gamma))


class LogCoshDiceLoss(nn.Module):
    """
    LogCoshDice Loss.

    L_{lc-dce} = log(cosh(DiceLoss)
    """

    def __init__(self, use_softmax: bool = True) -> None:
        super(LogCoshDiceLoss, self).__init__()
        self.use_softmax = use_softmax

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            y_pred = nn.Softmax(dim=1)(y_pred)
        one_hot_target = F.one_hot(y_true.to(torch.int64), num_classes=2).permute((0, 3, 1, 2)).to(torch.float)
        assert y_pred.shape == one_hot_target.shape
        numerator = 2.0 * torch.sum(y_pred * one_hot_target, dim=(-2, -1))
        denominator = torch.sum(y_pred + one_hot_target, dim=(-2, -1))
        return torch.log(torch.cosh(1 - torch.mean((numerator + consts.data.EPS) / (denominator + consts.data.EPS))))


class TLoss(nn.Module):
    """Implementation of the TLoss."""

    def __init__(
        self,
        device: torch.device,
        image_size: int = 352,
        nu: float = 1.0,
        epsilon: float = 1e-8,
        reduction: str = "mean",
        use_softmax: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.D = torch.tensor(
            (self.image_size * self.image_size),
            dtype=torch.float,
            device=device,
        )

        self.lambdas = torch.ones(
            (self.image_size, self.image_size),
            dtype=torch.float,
            device=device,
        )
        self.nu = nn.Parameter(torch.tensor(nu, dtype=torch.float, device=device))
        self.epsilon = torch.tensor(epsilon, dtype=torch.float, device=device)
        self.reduction = reduction
        self.use_softmax = use_softmax

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.use_softmax:
            y_pred = nn.Softmax(dim=1)(y_pred)[:, 1, ...]
        delta_i = y_pred - y_true
        sum_nu_epsilon = torch.exp(self.nu) + self.epsilon
        first_term = -torch.lgamma((sum_nu_epsilon + self.D) / 2)
        second_term = torch.lgamma(sum_nu_epsilon / 2)
        third_term = -0.5 * torch.sum(self.lambdas + self.epsilon)
        fourth_term = (self.D / 2) * torch.log(torch.tensor(np.pi))
        fifth_term = (self.D / 2) * (self.nu + self.epsilon)

        delta_squared = torch.pow(delta_i, 2)
        lambdas_exp = torch.exp(self.lambdas + self.epsilon).to(delta_squared.device)
        numerator = delta_squared * lambdas_exp
        numerator = torch.sum(numerator, dim=(1, 2))

        fraction = numerator / sum_nu_epsilon
        sixth_term = ((sum_nu_epsilon + self.D) / 2) * torch.log(1 + fraction)

        total_losses = first_term + second_term + third_term + fourth_term + fifth_term + sixth_term

        if self.reduction == "mean":
            return total_losses.mean()
        elif self.reduction == "sum":
            return total_losses.sum()
        elif self.reduction == "none":
            return total_losses
        else:
            raise ValueError(f"The reduction method '{self.reduction}' is not implemented.")


class HausdorffLoss(nn.Module):
    """
    The Hausdorff loss.
    """

    def __init__(self, use_softmax: bool = True) -> None:
        super().__init__()
        self.hausdorfer = HausdorffERLoss()
        self.use_softmax = use_softmax

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        y_true = y_true.unsqueeze(1)
        return self.hausdorfer(y_pred, y_true)


class ComboLoss(nn.Module):
    """
    It is defined as a weighted sum of Dice loss and a modified cross entropy. It attempts to leverage the
    flexibility of Dice loss of class imbalance and at same time use cross-entropy for curve smoothing.

    This loss will look like "batch bce-loss" when we consider all pixels flattened are predicted as correct or not

    This loss is perfect loss when the training loss come to -0.5 (with the default config)

    References:
        [Paper](https://arxiv.org/pdf/1805.02798.pdf). See the original paper at formula (3)
        [Author's implementation in Keras](https://github.com/asgsaeid/ComboLoss/blob/master/combo_loss.py)

    """

    def __init__(self, use_softmax: bool = True, ce_w: float = 0.5, ce_d_w: float = 0.5) -> None:
        super(ComboLoss, self).__init__()
        self.use_softmax = use_softmax
        self.ce_w = ce_w
        self.ce_d_w = ce_d_w
        self.eps = 1e-12
        self.smooth = 1

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        one_hot_target = F.one_hot(y_true.to(torch.int64), num_classes=2).permute((0, 3, 1, 2)).to(torch.float)

        # At this time, the output and one_hot_target have the same shape
        y_true_f = torch.flatten(one_hot_target)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        d = (2.0 * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)

        out = -(
            self.ce_w * y_true_f * torch.log(y_pred_f + self.eps)
            + (1 - self.ce_w) * (1.0 - y_true_f) * torch.log(1.0 - y_pred_f + self.eps)
        )
        weighted_ce = torch.mean(out, dim=-1)

        combo = (self.ce_d_w * weighted_ce) - ((1 - self.ce_d_w) * d)
        return combo


def soft_dice_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    numerator = 2.0 * torch.sum(y_pred * y_true, dim=(-2, -1))
    denominator = torch.sum(y_pred + y_true, dim=(-2, -1))
    return (numerator + consts.data.EPS) / (denominator + consts.data.EPS)


class SoftDiceLoss(nn.Module):
    """
    SoftDice loss.

    References:
        [JeremyJordan's
        Implementation](https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py)

        Paper related to this function:

        [Formula for binary segmentation case -
        A survey of loss functions for semantic segmentation](https://arxiv.org/pdf/2006.14822.pdf)

        [Formula for multiclass segmentation cases - Segmentation of Head and Neck Organs at Risk Using CNN with Batch
        Dice Loss](https://arxiv.org/pdf/1812.02427.pdf)
    """

    def __init__(self, reduction: str = "mean", use_softmax: bool = True) -> None:
        """
        Args:
            use_softmax: Set it to False when use the function for testing purpose
        """
        super(SoftDiceLoss, self).__init__()
        self.use_softmax = use_softmax
        self.reduction = reduction

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Calculate SoftDice loss.

        Args:
            y_pred: Tensor shape (N, N_Class, H, W), torch.float
            y_true: Tensor shape (N, H, W)

        Returns:

        """
        num_classes = y_pred.shape[1]
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        one_hot_target = (
            F.one_hot(y_true.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2)).to(torch.float)
        )
        assert y_pred.shape == one_hot_target.shape
        if self.reduction == "none":
            return 1.0 - soft_dice_loss(y_pred, one_hot_target)
        elif self.reduction == "mean":
            return 1.0 - torch.mean(soft_dice_loss(y_pred, one_hot_target))
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")


class BatchSoftDice(nn.Module):
    """
    This is the variance of SoftDiceLoss, it in introduced in this [paper](https://arxiv.org/pdf/1812.02427.pdf)

    References:
        [Segmentation of Head and Neck Organs at Risk Using CNN with
        Batch Dice Loss](https://arxiv.org/pdf/1812.02427.pdf)
    """

    def __init__(self, use_square: bool = False) -> None:
        """
        Args:
            use_square: If use square then the denominator will the sum of square
        """
        super(BatchSoftDice, self).__init__()
        self._use_square = use_square

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Calculates batch soft-dice loss.

        Args:
            y_pred: Tensor shape (N, N_Class, H, W), torch.float
            y_true: Tensor shape (N, H, W)
        Returns:
        """
        num_classes = y_pred.shape[1]
        batch_size = y_pred.shape[0]
        axes = (-2, -1)
        y_pred = F.softmax(y_pred, dim=1)
        one_hot_target = F.one_hot(y_true.to(torch.int64), num_classes=num_classes).permute((0, 3, 1, 2))
        assert y_pred.shape == one_hot_target.shape
        numerator = 2.0 * torch.sum(y_pred * one_hot_target, dim=axes)
        if self._use_square:
            denominator = torch.sum(torch.square(y_pred) + torch.square(one_hot_target), dim=axes)
        else:
            denominator = torch.sum(y_pred + one_hot_target, dim=axes)
        return (1 - torch.mean((numerator + consts.data.EPS) / (denominator + consts.data.EPS))) * batch_size


class ExponentialLogarithmicLoss(nn.Module):
    """
    This loss is focuses on less accurately predicted structures using the combination of Dice Loss ans Cross Entropy
    Loss

    References:
        [Original paper](https://arxiv.org/pdf/1809.00076.pdf)

        See the paper at 2.2 w_l = ((Sum k f_k) / f_l) ** 0.5 is the label weight

    Note:
        - Input for CrossEntropyLoss is the logits - Raw output from the model
    """

    def __init__(
        self,
        class_weights: Tensor,
        w_dice: float = 0.5,
        w_cross: float = 0.5,
        gamma: float = 0.3,
        use_softmax: bool = True,
    ) -> None:
        super(ExponentialLogarithmicLoss, self).__init__()
        self.w_dice = w_dice
        self.gamma = gamma
        self.w_cross = w_cross
        self.use_softmax = use_softmax
        self.class_weights = class_weights

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        self.class_weights = self.class_weights.to(y_true.device)
        weight_map = self.class_weights[y_true]

        y_true = F.one_hot(y_true.to(torch.int64), num_classes=2).permute((0, 3, 1, 2)).to(torch.float)
        if self.use_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        l_dice = torch.mean(torch.pow(-torch.log(soft_dice_loss(y_pred, y_true)), self.gamma))  # mean w.r.t to label
        l_cross = torch.mean(
            torch.mul(weight_map, torch.pow(F.cross_entropy(y_pred, y_true, reduction="none"), self.gamma))
        )
        return self.w_dice * l_dice + self.w_cross * l_cross


LOSS_REGISTRY = {
    "ce": nn.CrossEntropyLoss,
    "jaccard": smp.losses.JaccardLoss,
    "dice": smp.losses.DiceLoss,
    "focal": smp.losses.FocalLoss,
    "lovasz": smp.losses.LovaszLoss,
    "tversky": smp.losses.TverskyLoss,
    "soft_ce": smp.losses.SoftCrossEntropyLoss,
    "xedice": XEDiceLoss,
    "focal_tversky": FocalTverskyLoss,
    "log_cosh_dice": LogCoshDiceLoss,
    "hausdorff": HausdorffLoss,
    "t_loss": TLoss,
    "combo": ComboLoss,
    "exp_log_loss": ExponentialLogarithmicLoss,
    "soft_dice": SoftDiceLoss,
    "batch_soft_dice": BatchSoftDice,
}
