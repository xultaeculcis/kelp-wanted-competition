from __future__ import annotations

from pathlib import Path

from mlflow import ActiveRun


def get_mlflow_run_dir(current_run: ActiveRun, output_dir: Path) -> Path:
    """
    Gets MLFlow run directory given the active run and output directory.
    Args:
        current_run: The current active run.
        output_dir: The output directory.

    Returns: A path to the MLFlow run directory.

    """
    return Path(output_dir / str(current_run.info.experiment_id) / current_run.info.run_id)
