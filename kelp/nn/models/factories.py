from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Type

import segmentation_models_pytorch as smp
import torch
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_utilities import module_available
from segmentation_models_pytorch.base import SegmentationModel
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    OneCycleLR,
    ReduceLROnPlateau,
)

from kelp import consts
from kelp.nn.models.efficientunetplusplus.model import EfficientUnetPlusPlus
from kelp.nn.models.fcn.model import FCN
from kelp.nn.models.losses import LOSS_REGISTRY
from kelp.nn.models.resunet.model import ResUnet
from kelp.nn.models.resunetplusplus.model import ResUnetPlusPlus

_MODEL_LOOKUP: Dict[str, Type[SegmentationModel]] = {
    "deeplabv3": smp.DeepLabV3,
    "deeplabv3+": smp.DeepLabV3Plus,
    "efficientunet++": EfficientUnetPlusPlus,
    "fcn": FCN,
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
    if loss_fn not in LOSS_REGISTRY:
        raise ValueError(f"{loss_fn=} is not supported.")

    loss_kwargs: Dict[str, Any]
    if loss_fn in ["jaccard", "dice"]:
        loss_kwargs = {
            "mode": "multiclass",
            "ignore_index": ignore_index,
            "classes": list(range(num_classes)) if objective != "binary" else None,
        }
    elif loss_fn == "ce":
        loss_kwargs = {
            "weight": torch.tensor(ce_class_weights, device=device),
            "ignore_index": ignore_index or -100,
        }
    elif loss_fn == "soft_ce":
        loss_kwargs = {
            "ignore_index": ignore_index,
            "smooth_factor": ce_smooth_factor,
        }
    elif loss_fn == "xedice":
        loss_kwargs = {
            "mode": "multiclass",
            "ce_class_weights": torch.tensor(ce_class_weights, device=device),
        }
    elif loss_fn in [
        "focal_tversky",
        "log_cosh_dice",
        "hausdorff",
        "combo",
        "soft_dice",
        "batch_soft_dice",
        "sens_spec_loss",
    ]:
        loss_kwargs = {}
    elif loss_fn == "t_loss":
        loss_kwargs = {
            "device": device,
        }
    elif loss_fn == "exp_log_loss":
        loss_kwargs = {
            "class_weights": torch.tensor(ce_class_weights, device=device),
        }
    else:
        loss_kwargs = {
            "mode": "multiclass",
            "ignore_index": ignore_index,
        }

    return LOSS_REGISTRY[loss_fn](**loss_kwargs)


def resolve_model(
    architecture: str,
    encoder: str,
    classes: int,
    in_channels: int,
    encoder_weights: Optional[str] = None,
    encoder_depth: int = 5,
    decoder_channels: Optional[List[int]] = None,
    decoder_attention_type: Optional[str] = None,
    pretrained: bool = False,
    compile: bool = False,
    compile_mode: str = "default",
    compile_dynamic: Optional[bool] = None,
    ort: bool = False,
) -> nn.Module:
    if decoder_channels is None:
        decoder_channels = [256, 128, 64, 32, 16][:encoder_depth]

    if architecture in _MODEL_LOOKUP:
        model_kwargs = {
            "encoder_name": encoder,
            "encoder_weights": encoder_weights if pretrained else None,
            "in_channels": in_channels,
            "classes": classes,
            "encoder_depth": encoder_depth,
            "decoder_channels": decoder_channels,
            "decoder_attention_type": decoder_attention_type,
        }
        if "unet" not in architecture or architecture == "efficientunet++":
            model_kwargs.pop("decoder_attention_type")
        if architecture == "fcn":
            model_kwargs.pop("encoder_name")
            model_kwargs.pop("encoder_weights")
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


def resolve_optimizer(params: Iterator[Parameter], hyperparams: Dict[str, Any]) -> torch.optim.Optimizer:
    if (optimizer := hyperparams["optimizer"]) == "adam":
        optimizer = Adam(params, lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
    elif optimizer == "adamw":
        optimizer = AdamW(params, lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
    elif optimizer == "sgd":
        optimizer = SGD(params, lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"])
    else:
        raise ValueError(f"Optimizer: {optimizer} is not supported.")
    return optimizer


def resolve_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    steps_per_epoch: int,
    hyperparams: Dict[str, Any],
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    if (lr_scheduler := hyperparams["lr_scheduler"]) is None:
        return None
    elif lr_scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=hyperparams["lr"],
            total_steps=num_training_steps,
            pct_start=hyperparams["onecycle_pct_start"],
            div_factor=hyperparams["onecycle_div_factor"],
            final_div_factor=hyperparams["onecycle_final_div_factor"],
        )
    elif lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=hyperparams["epochs"],
            eta_min=hyperparams["cosine_eta_min"],
        )
    elif lr_scheduler == "cyclic":
        scheduler = CyclicLR(
            optimizer=optimizer,
            max_lr=hyperparams["lr"],
            base_lr=hyperparams["cyclic_base_lr"],
            step_size_up=steps_per_epoch,
            step_size_down=steps_per_epoch,
            mode=hyperparams["cyclic_mode"],
        )
    elif lr_scheduler == "cosine_with_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=steps_per_epoch,
            T_mult=hyperparams["cosine_T_mult"],
            eta_min=hyperparams["cosine_eta_min"],
        )
    elif lr_scheduler == "reduce_lr_on_plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=hyperparams["reduce_lr_on_plateau_factor"],
            patience=hyperparams["reduce_lr_on_plateau_patience"],
            threshold=hyperparams["reduce_lr_on_plateau_threshold"],
            min_lr=hyperparams["reduce_lr_on_plateau_min_lr"],
            verbose=True,
        )
    else:
        raise ValueError(f"LR Scheduler: {lr_scheduler} is not supported.")
    return scheduler
