from __future__ import annotations

import dataclasses
from typing import Any, Dict, cast

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from lightning_fabric.utilities.exceptions import MisconfigurationException
from lightning_utilities.core.imports import module_available
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import Accuracy, ConfusionMatrix, Dice, F1Score, JaccardIndex, MetricCollection, Precision, Recall

from kelp import consts


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
        self.loss = self._resolve_loss()
        self.model = self._configure_task()

        self.train_metrics = MetricCollection(
            metrics={
                "dice": Dice(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.hyperparams["ignore_index"],
                    average="macro",
                ),
                "iou": JaccardIndex(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.hyperparams["ignore_index"],
                ),
            },
            prefix="train/",
        )
        self.val_metrics = MetricCollection(
            metrics={
                "dice": Dice(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.hyperparams["ignore_index"],
                    average="macro",
                ),
                "iou": JaccardIndex(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.hyperparams["ignore_index"],
                ),
                "per_class_iou": JaccardIndex(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.hyperparams["ignore_index"],
                    average="none",
                ),
                "accuracy": Accuracy(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.hyperparams["ignore_index"],
                ),
                "recall": Recall(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.hyperparams["ignore_index"],
                ),
                "precision": Precision(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.hyperparams["ignore_index"],
                ),
                "f1": F1Score(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.hyperparams["ignore_index"],
                ),
                "conf_mtrx": ConfusionMatrix(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.hyperparams["ignore_index"],
                ),
                "norm_conf_mtrx": ConfusionMatrix(
                    task=self.hparams["objective"],
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.hyperparams["ignore_index"],
                    normalize="true",
                ),
            },
            prefix="val/",
        )
        self.test_metrics = self.val_metrics.clone(prefix="test/")

    def _resolve_loss(self) -> nn.Module:
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

    def _configure_task(self) -> nn.Module:
        architecture = self.hyperparams["architecture"]
        encoder = self.hyperparams["encoder"]
        encoder_weights = self.hyperparams["encoder_weights"]
        in_channels = self.hyperparams["in_channels"]
        classes = self.hyperparams["num_classes"]

        if architecture == "unet":
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights=encoder_weights if self.hyperparams["pretrained"] else None,
                in_channels=in_channels,
                classes=classes,
                decoder_attention_type=self.hyperparams["decoder_attention_type"],
            )
        else:
            raise ValueError(f"{architecture=} is not supported.")

        if self.hyperparams["compile"]:
            model = torch.compile(
                model,
                mode=self.hyperparams["compile_mode"],
                dynamic=self.hyperparams["compile_dynamic"],
            )

        if self.hyperparams["ort"]:
            if module_available("torch_ort"):
                from torch_ort import ORTModule  # noqa

                model = ORTModule(model)
            else:
                raise MisconfigurationException(
                    "Torch ORT is required to use ORT. See here for installation: https://github.com/pytorch/ort"
                )

        return model

    def _log_predictions_batch(self, batch: dict[str, Tensor], batch_idx: int, y_hat_hard: Tensor) -> None:
        # Ensure global step is non-zero -> that we are not running plotting during sanity val step check
        epoch = self.current_epoch
        if batch_idx < self.hyperparams["plot_n_batches"] and self.global_step:
            datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            fig_grids = datamodule.plot_batch(
                batch=batch,
                plot_true_color=epoch == 0,
                plot_color_infrared_grid=epoch == 0,
                plot_short_wave_infrared_grid=epoch == 0,
                plot_qa_grid=epoch == 0,
                plot_dem_grid=epoch == 0,
                plot_mask_grid=epoch == 0,
                plot_prediction_grid=True,
            )
            for key, fig in dataclasses.asdict(fig_grids).items():
                self.logger.experiment.log_figure(  # type: ignore[attr-defined]
                    run_id=self.logger.run_id,  # type: ignore[attr-defined]
                    figure=fig,
                    artifact_file=f"images/{key}/{key}_{epoch=:02d}_{batch_idx=}.jpg",
                )
                plt.close(fig)

    def _log_confusion_matrices(self, metrics: dict[str, Tensor], cmap: str = "Blues") -> None:
        for metric_key, title, matrix_kind in zip(
            ["val/conf_mtrx", "val/norm_conf_mtrx"],
            ["Confusion matrix", "Normalized confusion matrix"],
            ["confusion_matrix", "confusion_matrix_normalized"],
        ):
            conf_matrix = metrics.pop(metric_key)
            fig, axes = plt.subplots(1, 1, figsize=(7, 5))
            ConfusionMatrixDisplay(
                confusion_matrix=conf_matrix.detach().cpu().numpy(),
                display_labels=consts.data.CLASSES,
            ).plot(
                colorbar=True,
                cmap=cmap,
                ax=axes,
            )
            axes.set_title(title)
            self.logger.experiment.log_figure(  # type: ignore[attr-defined]
                run_id=self.logger.run_id,  # type: ignore[attr-defined]
                figure=fig,
                artifact_file=f"image/{matrix_kind}/{matrix_kind}_{self.current_epoch:02d}.jpg",
            )
            plt.close(fig)

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
        self._log_predictions_batch(batch, batch_idx, y_hat_hard)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        per_class_iou = metrics.pop("val/per_class_iou")
        self._log_confusion_matrices(metrics)
        per_class_iou_score_dict = {
            f"val/iou_{consts.data.CLASSES[idx]}": iou_score for idx, iou_score in enumerate(per_class_iou)
        }
        metrics.update(per_class_iou_score_dict)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        batch = args[0]
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.test_metrics(y_hat_hard, y)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        metrics.pop("test/per_class_iou")
        metrics.pop("test/norm_conf_mtrx")
        metrics.pop("test/conf_mtrx")
        self.log_dict(metrics, on_step=False, on_epoch=True)
        self.test_metrics.reset()

    def predict_step(self, *args: Any, **kwargs: Any) -> Tensor:
        batch = args[0]
        x = batch.pop("image")
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)
        batch["prediction"] = y_hat_hard
        return batch

    def configure_optimizers(self) -> Dict[str, Any]:
        if (optimizer := self.hyperparams["optimizer"]) == "adam":
            optimizer = Adam(
                self.model.parameters(), lr=self.hyperparams["lr"], weight_decay=self.hyperparams["weight_decay"]
            )
        elif optimizer == "adamw":
            optimizer = AdamW(
                self.model.parameters(), lr=self.hyperparams["lr"], weight_decay=self.hyperparams["weight_decay"]
            )
        else:
            raise ValueError(f"Optimizer: {optimizer} is not supported.")

        if (lr_scheduler := self.hyperparams["lr_scheduler"]) == "onecycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hyperparams["lr"],
                steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),  # type: ignore[attr-defined]
                epochs=self.hyperparams["epochs"],
                pct_start=self.hyperparams["pct_start"],
                div_factor=self.hyperparams["div_factor"],
                final_div_factor=self.hyperparams["final_div_factor"],
            )
        elif lr_scheduler is None:
            return {"optimizer": optimizer}
        else:
            raise ValueError(f"LR Scheduler: {lr_scheduler} is not supported.")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
