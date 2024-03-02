import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import field_validator

from kelp import consts
from kelp.core.configs import ConfigBase
from kelp.core.indices import ALL_INDICES, SPECTRAL_INDEX_LOOKUP
from kelp.utils.logging import get_logger

_logger = get_logger(__name__)


class TrainConfig(ConfigBase):
    """The XGBoost training configuration."""

    dataset_fp: Path
    train_data_dir: Path
    output_dir: Path
    spectral_indices: List[str]
    sample_size: float = 1.0
    seed: int = consts.reproducibility.SEED
    plot_n_samples: int = 10
    experiment: str = "train-xgb-clf-exp"
    explain_model: bool = False

    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 1000

    @field_validator("spectral_indices", mode="before")
    def validate_spectral_indices(cls, value: Union[str, Optional[List[str]]] = None) -> List[str]:
        if not value:
            return ["DEMWM", "NDVI"]

        if value == "all" or value == ALL_INDICES:
            return ALL_INDICES

        indices = value if isinstance(value, list) else [index.strip() for index in value.split(",")]

        if "DEMWM" in indices:
            _logger.warning("DEMWM is automatically added during training. No need to add it twice.")
            indices.remove("DEMWM")

        if "NDVI" in indices:
            _logger.warning("NDVI is automatically added during training. No need to add it twice.")
            indices.remove("NDVI")

        unknown_indices = set(indices).difference(list(SPECTRAL_INDEX_LOOKUP.keys()))
        if unknown_indices:
            raise ValueError(
                f"Unknown spectral indices were provided: {', '.join(unknown_indices)}. "
                f"Please provide comma separated indices. Valid choices are: {', '.join(SPECTRAL_INDEX_LOOKUP.keys())}."
            )

        return ["DEMWM", "NDVI"] + indices

    @property
    def resolved_experiment_name(self) -> str:
        return os.environ.get("MLFLOW_EXPERIMENT_NAME", self.experiment)

    @property
    def run_id_from_context(self) -> Optional[str]:
        return os.environ.get("MLFLOW_RUN_ID", None)

    @property
    def tags(self) -> Dict[str, Any]:
        return {"trained_at": datetime.utcnow().isoformat()}

    @property
    def columns_to_load(self) -> List[str]:
        return self.model_input_columns + ["label", "tile_id", "split"]

    @property
    def model_input_columns(self) -> List[str]:
        return list(consts.data.ORIGINAL_BANDS) + self.spectral_indices

    @property
    def random_forest_model_params(self) -> Dict[str, Any]:
        return {
            "random_state": self.seed,
        }

    @property
    def gradient_boosting_tree_model_params(self) -> Dict[str, Any]:
        return {
            "random_state": self.seed,
        }

    @property
    def catboost_model_params(self) -> Dict[str, Any]:
        return {
            "random_state": self.seed,
        }

    @property
    def xgboost_model_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.xgb_n_estimators,
            "learning_rate": self.xgb_learning_rate,
            "random_state": self.seed,
            "device": "cuda",
            "n_jobs": -1,
        }

    @property
    def lightgbm_model_params(self) -> Dict[str, Any]:
        return {
            "random_state": self.seed,
        }
