import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import rasterio
import yaml
from pydantic import model_validator
from tqdm import tqdm

from kelp.consts.data import META
from kelp.core.configs import ConfigBase
from kelp.core.submission import create_submission_tar
from kelp.nn.inference.preview_submission import plot_first_n_samples
from kelp.utils.logging import get_logger

FOLDS = 10
_logger = get_logger(__name__)


class AveragePredictionsConfig(ConfigBase):
    """Config for running prediction averaging logic."""

    predictions_dirs: List[Path]
    output_dir: Path
    decision_threshold: float = 0.5
    weights: List[float]
    preview_submission: bool = False
    test_data_dir: Optional[Path] = None
    preview_first_n: int = 10

    @model_validator(mode="before")
    def validate_cfg(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["preview_submission"] and values.get("test_data_dir", None) is None:
            raise ValueError("Please provide test_data_dir param if running submission preview!")
        return values


def parse_args() -> AveragePredictionsConfig:
    """
    Parse command line arguments.

    Returns: An instance of AveragePredictionsConfig.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dirs", nargs="*", required=True)
    parser.add_argument("--weights", nargs="*", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--decision_threshold", type=float, default=0.5)
    parser.add_argument("--preview_submission", action="store_true")
    parser.add_argument("--test_data_dir", type=str)
    parser.add_argument("--preview_first_n", type=int, default=10)
    args = parser.parse_args()
    cfg = AveragePredictionsConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


def average_predictions(
    preds_dirs: List[Path],
    output_dir: Path,
    weights: List[float],
    decision_threshold: float = 0.5,
) -> None:
    """
    Average predictions given a list of directories with predictions from single models.

    Args:
        preds_dirs: The list of directories with predictions from single model.
        output_dir: The output directory.
        weights: The list of weights for each fold (prediction directory).
        decision_threshold: The final decision threshold.

    """
    if len(weights) != len(preds_dirs):
        raise ValueError("Number of weights must match the number prediction dirs!")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: Dict[str, Dict[str, Union[np.ndarray, float, int]]] = {}  # type: ignore[type-arg]

    for preds_dir, weight in zip(preds_dirs, weights):
        if weight == 0.0:
            _logger.info(f"Weight for {preds_dir.name} == 0.0. Skipping this fold.")
            continue
        for pred_file in tqdm(
            sorted(list(preds_dir.glob("*.tif"))),
            desc=f"Processing files for {preds_dir.name}, {weight=}",
        ):
            file_name = pred_file.name
            with rasterio.open(pred_file) as src:
                pred_array = src.read(1) * weight
                if file_name not in predictions:
                    predictions[file_name] = {
                        "data": np.zeros_like(pred_array, dtype=np.float32),
                        "count": 1,
                        "weight_sum": weight,
                    }
                predictions[file_name]["data"] += pred_array
                predictions[file_name]["count"] += 1
                predictions[file_name]["weight_sum"] += weight

    for file_name, content in tqdm(predictions.items(), desc="Saving predictions"):
        content["data"] = content["data"] / content["weight_sum"]
        content["data"] = np.where(content["data"] >= decision_threshold, 1, 0).astype(np.uint8)
        output_file = output_dir / file_name
        with rasterio.open(output_file, "w", **META) as dst:
            dst.write(content["data"].astype(rasterio.uint8), 1)  # type: ignore[union-attr]


def main() -> None:
    """Main entrypoint for averaging the predictions and creating a submission file."""
    cfg = parse_args()
    now = datetime.utcnow().isoformat()
    out_dir = cfg.output_dir / now
    preds_dir = cfg.output_dir / now / "predictions"
    preds_dir.mkdir(exist_ok=False, parents=True)
    avg_preds_config = cfg.model_dump(mode="json")
    (out_dir / "predict_config.yaml").write_text(yaml.dump(avg_preds_config))
    average_predictions(
        preds_dirs=cfg.predictions_dirs,
        output_dir=preds_dir,
        weights=cfg.weights,
        decision_threshold=cfg.decision_threshold,
    )
    create_submission_tar(
        preds_dir=preds_dir,
        output_dir=out_dir,
    )
    if cfg.preview_submission:
        plot_first_n_samples(
            data_dir=cfg.test_data_dir,  # type: ignore[arg-type]
            submission_dir=out_dir,
            output_dir=out_dir / "previews",
            n=cfg.preview_first_n,
        )


if __name__ == "__main__":
    main()
