import shutil
from datetime import datetime
from pathlib import Path

from kelp.entrypoints.predict import parse_args, run_prediction
from kelp.entrypoints.submission import create_submission_tar


def copy_run_artifacts(run_dir: Path, output_dir: Path) -> None:
    shutil.copytree(run_dir, output_dir / run_dir.name, dirs_exist_ok=True)


def main() -> None:
    cfg = parse_args()
    now = datetime.utcnow().isoformat()
    out_dir = cfg.output_dir / now
    preds_dir = cfg.output_dir / now / "predictions"
    preds_dir.mkdir(exist_ok=False, parents=True)
    run_prediction(
        data_dir=cfg.data_dir,
        output_dir=preds_dir,
        model_checkpoint=cfg.model_checkpoint,
        use_mlflow=cfg.use_mlflow,
        train_cfg=cfg.training_config,
    )
    create_submission_tar(
        preds_dir=preds_dir,
        output_dir=out_dir,
    )
    copy_run_artifacts(
        run_dir=cfg.run_dir,  # type: ignore[arg-type]
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
