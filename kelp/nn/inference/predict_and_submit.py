import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml

from kelp.core.submission import create_submission_tar
from kelp.nn.inference.predict import PredictConfig, build_prediction_arg_parser, run_prediction, run_sahi_prediction
from kelp.nn.inference.preview_submission import plot_first_n_samples

torch.set_float32_matmul_precision("medium")


class PredictAndSubmitConfig(PredictConfig):
    """Config for running prediction and submission file generation in a single pass."""

    preview_submission: bool = False
    preview_first_n: int = 10


def parse_args() -> PredictAndSubmitConfig:
    """
    Parse command line arguments.

    Returns: An instance of PredictAndSubmitConfig.

    """
    parser = build_prediction_arg_parser()
    parser.add_argument("--preview_submission", action="store_true")
    parser.add_argument("--preview_first_n", type=int, default=10)
    args = parser.parse_args()
    cfg = PredictAndSubmitConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


def copy_run_artifacts(run_dir: Path, output_dir: Path) -> None:
    """
    Copies run artifacts from run_dir to output_dir.

    Args:
        run_dir: The directory to copy run artifacts from.
        output_dir: The output directory.

    """
    shutil.copytree(run_dir, output_dir / run_dir.name, dirs_exist_ok=True)


def main() -> None:
    """The main entry point for running prediction and submission file generation."""
    cfg = parse_args()
    now = datetime.utcnow().isoformat()
    out_dir = cfg.output_dir / now
    preds_dir = cfg.output_dir / now / "predictions"
    preds_dir.mkdir(exist_ok=False, parents=True)
    (out_dir / "predict_config.yaml").write_text(yaml.dump(cfg.model_dump(mode="json")))
    if cfg.training_config.sahi:
        run_sahi_prediction(
            data_dir=cfg.data_dir,
            output_dir=preds_dir,
            model_checkpoint=cfg.model_checkpoint,
            use_mlflow=cfg.use_mlflow,
            train_cfg=cfg.training_config,
            tta=cfg.tta,
            soft_labels=cfg.soft_labels,
            tta_merge_mode=cfg.tta_merge_mode,
            decision_threshold=cfg.decision_threshold,
            sahi_tile_size=cfg.sahi_tile_size,
            sahi_overlap=cfg.sahi_overlap,
        )
    else:
        run_prediction(
            data_dir=cfg.data_dir,
            output_dir=preds_dir,
            model_checkpoint=cfg.model_checkpoint,
            use_mlflow=cfg.use_mlflow,
            train_cfg=cfg.training_config,
            tta=cfg.tta,
            tta_merge_mode=cfg.tta_merge_mode,
            decision_threshold=cfg.decision_threshold,
        )
    create_submission_tar(
        preds_dir=preds_dir,
        output_dir=out_dir,
    )
    copy_run_artifacts(
        run_dir=cfg.run_dir,  # type: ignore[arg-type]
        output_dir=out_dir,
    )
    if cfg.preview_submission:
        plot_first_n_samples(
            data_dir=cfg.data_dir,
            submission_dir=out_dir,
            output_dir=out_dir / "previews",
            n=cfg.preview_first_n,
        )


if __name__ == "__main__":
    main()
