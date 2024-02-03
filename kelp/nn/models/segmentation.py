from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, Literal, Tuple, cast

import pytorch_lightning as pl
import torch
import ttach as tta
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor
from torchmetrics import Accuracy, ConfusionMatrix, Dice, F1Score, JaccardIndex, MetricCollection, Precision, Recall

from kelp import consts
from kelp.nn.models.factories import resolve_loss, resolve_lr_scheduler, resolve_model, resolve_optimizer

_test_time_transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
    ]
)


class KelpForestSegmentationTask(pl.LightningModule):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(Dict[str, Any], self.hparams)
        self.loss = resolve_loss(
            loss_fn=self.hyperparams["loss"],
            device=self.device,
            ignore_index=self.hyperparams["ignore_index"],
            num_classes=self.hyperparams["num_classes"],
            objective=self.hyperparams["objective"],
            ce_smooth_factor=self.hyperparams["ce_smooth_factor"],
            ce_class_weights=self.hyperparams["ce_class_weights"],
        )
        self.model = resolve_model(
            architecture=self.hyperparams["architecture"],
            encoder=self.hyperparams["encoder"],
            encoder_weights=self.hyperparams["encoder_weights"],
            decoder_attention_type=self.hyperparams["decoder_attention_type"],
            pretrained=self.hyperparams["pretrained"],
            in_channels=self.hyperparams["in_channels"],
            classes=self.hyperparams["num_classes"],
            compile=self.hyperparams["compile"],
            compile_mode=self.hyperparams["compile_mode"],
            compile_dynamic=self.hyperparams["compile_dynamic"],
            ort=self.hyperparams["ort"],
        )
        self.train_metrics = MetricCollection(
            metrics={
                "dice": Dice(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.hyperparams["ignore_index"],
                    average="macro",
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
                    task="multiclass",  # must be 'multiclass' for per-class IoU
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

    @property
    def num_training_steps(self) -> int:
        return self.trainer.estimated_stepping_batches  # type: ignore[no-any-return]

    def _log_predictions_batch(self, batch: Dict[str, Tensor], batch_idx: int, y_hat_hard: Tensor) -> None:
        # Ensure global step is non-zero -> that we are not running plotting during sanity val step check
        epoch = self.current_epoch
        step = self.global_step
        if batch_idx < self.hyperparams["plot_n_batches"] and step:
            datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
            band_index_lookup = {band: idx for idx, band in enumerate(datamodule.bands_to_use)}
            can_plot_true_color = all(band in band_index_lookup for band in ["R", "G", "B"])
            can_plot_color_infrared_color = all(band in band_index_lookup for band in ["NIR", "R", "G"])
            can_plot_shortwave_infrared_color = all(band in band_index_lookup for band in ["SWIR", "NIR", "R"])
            batch["prediction"] = y_hat_hard
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            fig_grids = datamodule.plot_batch(
                batch=batch,
                band_index_lookup=band_index_lookup,
                plot_true_color=epoch == 0 and can_plot_true_color,
                plot_color_infrared_grid=epoch == 0 and can_plot_color_infrared_color,
                plot_short_wave_infrared_grid=epoch == 0 and can_plot_shortwave_infrared_color,
                plot_spectral_indices=epoch == 0,
                plot_qa_grid=epoch == 0 and "QA" in band_index_lookup,
                plot_dem_grid=epoch == 0 and "DEM" in band_index_lookup,
                plot_mask_grid=epoch == 0,
                plot_prediction_grid=True,
            )
            for key, fig in dataclasses.asdict(fig_grids).items():
                if fig is None:
                    continue

                if isinstance(fig, dict):
                    for nested_key, nested_figure in fig.items():
                        self.logger.experiment.log_figure(  # type: ignore[attr-defined]
                            run_id=self.logger.run_id,  # type: ignore[attr-defined]
                            figure=nested_figure,
                            artifact_file=f"images/{key}/{nested_key}_{batch_idx=}_{epoch=:02d}_{step=:04d}.jpg",
                        )
                        plt.close(nested_figure)
                else:
                    self.logger.experiment.log_figure(  # type: ignore[attr-defined]
                        run_id=self.logger.run_id,  # type: ignore[attr-defined]
                        figure=fig,
                        artifact_file=f"images/{key}/{key}_{batch_idx=}_{epoch=:02d}_{step=:04d}.jpg",
                    )
            plt.close()

    def _log_confusion_matrices(
        self, metrics: Dict[str, Tensor], stage: Literal["val", "test"], cmap: str = "Blues"
    ) -> None:
        epoch = self.current_epoch
        step = self.global_step
        for metric_key, title, matrix_kind in zip(
            [f"{stage}/conf_mtrx", f"{stage}/norm_conf_mtrx"],
            ["Confusion matrix", "Normalized confusion matrix"],
            ["confusion_matrix", "confusion_matrix_normalized"],
        ):
            conf_matrix = metrics.pop(metric_key)
            # Ensure global step is non-zero -> that we are not running plotting during sanity val step check
            if step == 0:
                continue
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
                artifact_file=f"images/{stage}_{matrix_kind}/{matrix_kind}_{epoch=:02d}_{step=:04d}.jpg",
            )
            plt.close(fig)

    def _predict_with_tta_if_necessary(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.hyperparams.get("tta", False):
            tta_model = tta.SegmentationTTAWrapper(
                model=self.model,
                transforms=_test_time_transforms,
                merge_mode=self.hyperparams.get("tta_merge_mode", "mean"),
            )
            y_hat = tta_model(x)
        else:
            y_hat = self.forward(x)

        if self.hyperparams.get("decision_threshold", None):
            y_hat_hard = (  # type: ignore[attr-defined]
                y_hat.sigmoid()[:, 1, :, :] >= self.hyperparams["decision_threshold"]
            ).long()
        else:
            y_hat_hard = y_hat.argmax(dim=1)

        return y_hat, y_hat_hard

    def _guard_against_nan(self, x: Tensor) -> None:
        if torch.isnan(x):
            raise ValueError("NaN encountered during training! Aborting.")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        batch = args[0]
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)
        loss = self.loss(y_hat, y)
        self._guard_against_nan(loss)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
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
        self._guard_against_nan(loss)
        self.log("val/loss", loss, on_step=False, on_epoch=True, batch_size=x.shape[0], prog_bar=True)
        self.val_metrics(y_hat_hard, y)
        self._log_predictions_batch(batch, batch_idx, y_hat_hard)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        per_class_iou = metrics.pop("val/per_class_iou")
        self._log_confusion_matrices(metrics, stage="val")
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
        y_hat, y_hat_hard = self._predict_with_tta_if_necessary(x)
        loss = self.loss(y_hat, y)
        self._guard_against_nan(loss)
        self.log("test/loss", loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.test_metrics(y_hat_hard, y)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        per_class_iou = metrics.pop("test/per_class_iou")
        self._log_confusion_matrices(metrics, stage="test")
        per_class_iou_score_dict = {
            f"test/iou_{consts.data.CLASSES[idx]}": iou_score for idx, iou_score in enumerate(per_class_iou)
        }
        metrics.update(per_class_iou_score_dict)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        self.test_metrics.reset()

    def predict_step(self, *args: Any, **kwargs: Any) -> Tensor:
        batch = args[0]
        x = batch.pop("image")
        y_hat, y_hat_hard = self._predict_with_tta_if_necessary(x)
        batch["prediction"] = y_hat_hard
        return batch

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = resolve_optimizer(
            params=self.model.parameters(),
            hyperparams=self.hyperparams,
        )
        total_steps = self.num_training_steps
        scheduler = resolve_lr_scheduler(
            optimizer=optimizer,
            num_training_steps=total_steps,
            steps_per_epoch=math.ceil(total_steps / self.hyperparams["epochs"]),
            hyperparams=self.hyperparams,
        )
        if scheduler is None:
            return {"optimizer": optimizer}
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
                if self.hyperparams["lr_scheduler"] in ["onecycle", "cyclic", "cosine_with_warm_restarts"]
                else "epoch",
                "monitor": "val/loss",
            },
        }
