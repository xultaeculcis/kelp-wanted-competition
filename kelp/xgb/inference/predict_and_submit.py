import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from kelp.core.submission import create_submission_tar
from kelp.nn.inference.preview_submission import plot_first_n_samples
from kelp.xgb.inference.predict import PredictConfig, build_prediction_arg_parser, run_prediction


class PredictAndSubmitConfig(PredictConfig):
    preview_submission: bool = False
    submission_preview_output_dir: Optional[Path] = None
    preview_first_n: int = 10


def parse_args() -> PredictAndSubmitConfig:
    """
    Parse command line arguments.

    Returns: An instance of PredictAndSubmitConfig.

    """
    parser = build_prediction_arg_parser()
    parser.add_argument("--preview_submission", action="store_true")
    parser.add_argument("--submission_preview_output_dir", type=str)
    parser.add_argument("--preview_first_n", type=int, default=10)
    args = parser.parse_args()
    cfg = PredictAndSubmitConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


def copy_run_artifacts(run_dir: Path, output_dir: Path) -> None:
    shutil.copytree(run_dir, output_dir / run_dir.name, dirs_exist_ok=True)


def main() -> None:
    cfg = parse_args()
    now = datetime.utcnow().isoformat()
    out_dir = cfg.output_dir / now
    preds_dir = cfg.output_dir / now / "predictions"
    preds_dir.mkdir(exist_ok=False, parents=True)
    (out_dir / "predict_config.yaml").write_text(yaml.dump(cfg.model_dump(mode="json")))
    run_prediction(
        data_dir=cfg.data_dir,
        output_dir=preds_dir,
        model_dir=cfg.model_path,
        spectral_indices=cfg.training_config.spectral_indices,
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
