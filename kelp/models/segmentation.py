from typing import Any, Dict, cast

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import Tensor
from torchmetrics import Accuracy, ConfusionMatrix, Dice, F1Score, JaccardIndex, MetricCollection, Precision, Recall

from kelp.data.utils import unbind_samples


class KelpForestSegmentationTask(pl.LightningModule):
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            architecture: Name of the segmentation model type to use
            encoder: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            in_channels: Number of channels in input image
            num_classes: Number of semantic classes to predict
            loss: Name of the loss function
            objective: Name of the mode for the loss function
            ignore_index: Whether to ignore the "0" class value in the loss and metrics

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        self.ignore_index = self.hyperparams["ignore_index"]

        self.loss = self.resolve_loss()
        self.model = self.config_task()

        self.train_metrics = MetricCollection(
            metrics={
                "iou": JaccardIndex(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                ),
            },
            prefix="train/",
        )
        self.val_metrics = MetricCollection(
            metrics={
                "dice": Dice(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    average="macro",
                ),
                "iou": JaccardIndex(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                ),
                "per_class_iou": JaccardIndex(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                ),
                "accuracy": Accuracy(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                ),
                "recall": Recall(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                ),
                "precision": Precision(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                ),
                "f1": F1Score(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                ),
                "conf_mtrx": ConfusionMatrix(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                ),
                "norm_conf_mtrx": ConfusionMatrix(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    normalize="true",
                ),
            },
            prefix="val/",
        )
        self.test_metrics = self.val_metrics.clone(prefix="test/")

    def resolve_loss(self) -> nn.Module:
        loss_fn = self.hyperparams["loss"]
        ignore_index = self.hyperparams["ignore_index"]
        num_classes = self.hyperparams["num_classes"]
        objective = self.hyperparams["objective"]

        if loss_fn == "ce":
            loss = nn.CrossEntropyLoss(ignore_index=ignore_index or -100)
        elif loss_fn == "jaccard":
            loss = smp.losses.JaccardLoss(mode=objective, classes=num_classes)
        elif loss_fn == "dice":
            loss = smp.losses.DiceLoss(mode=objective, classes=num_classes, ignore_index=ignore_index)
        elif loss_fn == "focal":
            loss = smp.losses.FocalLoss(mode=objective, ignore_index=ignore_index)
        elif loss_fn == "lovasz":
            loss = smp.losses.LovaszLoss(mode=objective, ignore_index=ignore_index)
        elif loss_fn == "tversky":
            loss = smp.losses.TverskyLoss(mode=objective, ignore_index=ignore_index)
        elif loss_fn == "soft_ce":
            loss = smp.losses.SoftCrossEntropyLoss(ignore_index=ignore_index)
        elif loss_fn == "soft_bce_with_logits":
            loss = smp.losses.SoftBCEWithLogitsLoss(ignore_index=ignore_index)
        else:
            raise ValueError(f"{loss_fn=} is not supported.")
        return loss

    def config_task(self) -> nn.Module:
        architecture = self.hyperparams["architecture"]
        encoder = self.hyperparams["encoder"]
        encoder_weights = self.hyperparams["encoder_weights"]
        in_channels = self.hyperparams["in_channels"]
        classes = self.hyperparams["num_classes"]

        if architecture == "unet":
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
            )
        else:
            raise ValueError(f"{architecture=} is not supported.")

        return model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        batch = args[0]
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        self.log("train/loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.val_metrics(y_hat_hard, y)

        # Ensure global step is non-zero -> that we are not running plotting during sanity val step check
        if batch_idx < 3 and self.global_step > 0:
            datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            for sample in unbind_samples(batch):
                fig = datamodule.plot(sample, suptitle=f"Tile ID: {sample['tile_id']}")
                self.logger.experiment.log_figure(
                    run_id=self.logger.run_id,
                    figure=fig,
                    artifact_file=f"image/{sample['tile_id']}_{self.current_epoch:02d}.jpg",
                )
                plt.close(fig)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        metrics.pop("val/per_class_iou")
        metrics.pop("val/norm_conf_mtrx")
        metrics.pop("val/conf_mtrx")
        self.log_dict(metrics)
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        batch = args[0]
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.test_metrics(y_hat_hard, y)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        if self.hyperparams["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.hyperparams["lr"], weight_decay=self.hyperparams["weight_decay"]
            )
        else:
            raise ValueError(f"Optimizer: {self.hyperparams['optimizer']} is not supported.")
        return {
            "optimizer": optimizer,
        }