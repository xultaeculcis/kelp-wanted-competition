from __future__ import annotations

from typing import Dict, List, Optional, Type

import segmentation_models_pytorch as smp
import torch
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_utilities import module_available
from segmentation_models_pytorch.base import SegmentationModel
from torch import nn

from kelp import consts
from kelp.models.efficientunetplusplus.model import EfficientUnetPlusPlus
from kelp.models.resunet.model import ResUnet
from kelp.models.resunetplusplus.model import ResUnetPlusPlus

_MODEL_LOOKUP: Dict[str, Type[SegmentationModel]] = {
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus,
    "efficientunet++": EfficientUnetPlusPlus,
    "fpn": smp.FPN,
    "linknet": smp.Linknet,
    "manet": smp.MAnet,
    "pan": smp.PAN,
    "pspnet": smp.PSPNet,
    "resunet": ResUnet,
    "resunet++": ResUnetPlusPlus,
    "unet": smp.Unet,
    "unet++": smp.UnetPlusPlus,
}


def resolve_loss(
    loss_fn: str,
    objective: str,
    device: torch.device,
    num_classes: int = consts.data.NUM_CLASSES,
    ce_smooth_factor: float = 0.0,
    ce_class_weights: Optional[List[float]] = None,
    ignore_index: Optional[int] = None,
) -> nn.Module:
    if loss_fn == "ce":
        loss = nn.CrossEntropyLoss(
            weight=torch.tensor(ce_class_weights, device=device),
            ignore_index=ignore_index or -100,
        )
    elif loss_fn == "jaccard":
        loss = smp.losses.JaccardLoss(
            mode="multiclass",  # must be multiclass since we return predictions in shape NxCxHxW
            classes=list(range(num_classes)) if objective != "binary" else None,
        )
    elif loss_fn == "dice":
        loss = smp.losses.DiceLoss(
            mode="multiclass",  # must be multiclass since we return predictions in shape NxCxHxW
            classes=list(range(num_classes)) if objective != "binary" else None,
            ignore_index=ignore_index,
        )
    elif loss_fn == "focal":
        loss = smp.losses.FocalLoss(
            mode="multiclass",  # must be multiclass since we return predictions in shape NxCxHxW
            ignore_index=ignore_index,
        )
    elif loss_fn == "lovasz":
        loss = smp.losses.LovaszLoss(
            mode="multiclass",  # must be multiclass since we return predictions in shape NxCxHxW
            ignore_index=ignore_index,
        )
    elif loss_fn == "tversky":
        loss = smp.losses.TverskyLoss(
            mode="multiclass",  # must be multiclass since we return predictions in shape NxCxHxW
            ignore_index=ignore_index,
        )
    elif loss_fn == "soft_ce":
        loss = smp.losses.SoftCrossEntropyLoss(ignore_index=ignore_index, smooth_factor=ce_smooth_factor)
    else:
        raise ValueError(f"{loss_fn=} is not supported.")
    return loss


def resolve_model(
    architecture: str,
    encoder: str,
    classes: int,
    in_channels: int,
    encoder_weights: Optional[str] = None,
    decoder_attention_type: Optional[str] = None,
    pretrained: bool = False,
    compile: bool = False,
    compile_mode: str = "default",
    compile_dynamic: Optional[bool] = None,
    ort: bool = False,
) -> nn.Module:
    if architecture in _MODEL_LOOKUP:
        model_kwargs = {
            "encoder_name": encoder,
            "encoder_weights": encoder_weights if pretrained else None,
            "in_channels": in_channels,
            "classes": classes,
            "decoder_attention_type": decoder_attention_type,
        }
        if "unet" not in architecture:
            model_kwargs.pop("decoder_attention_type")
        model = _MODEL_LOOKUP[architecture](**model_kwargs)
    else:
        raise ValueError(f"{architecture=} is not supported.")

    if compile:
        model = torch.compile(
            model,
            mode=compile_mode,
            dynamic=compile_dynamic,
        )

    if ort:
        if module_available("torch_ort"):
            from torch_ort import ORTModule  # noqa

            model = ORTModule(model)
        else:
            raise MisconfigurationException(
                "Torch ORT is required to use ORT. See here for installation: https://github.com/pytorch/ort"
            )

    return model
