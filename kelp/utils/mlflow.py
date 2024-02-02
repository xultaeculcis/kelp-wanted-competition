from __future__ import annotations

from pathlib import Path

from mlflow import ActiveRun


def get_mlflow_run_dir(current_run: ActiveRun, output_dir: Path) -> Path:
    return Path(output_dir / str(current_run.info.experiment_id) / current_run.info.run_id)
