from typing import Any, Optional

from segmentation_models_pytorch.losses import DiceLoss
from torch import Tensor, nn


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
        self.dice = DiceLoss(mode=mode)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return self.weight_ce * self.xe(y_pred, y_true) + self.weight_dice * self.dice(y_pred, y_true)
