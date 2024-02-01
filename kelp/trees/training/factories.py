from typing import Any

import mlflow
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

from kelp import consts
from kelp.trees.training.estimator import Estimator


def model_factory(model_type: str, seed: int = consts.reproducibility.SEED, **kwargs: Any) -> Estimator:
    if model_type == "rf":
        mlflow.sklearn.autolog()
        return RandomForestClassifier(**kwargs.get("rf_params", {}))  # type: ignore[no-any-return]
    elif model_type == "gbt":
        mlflow.sklearn.autolog()
        return GradientBoostingClassifier(**kwargs.get("catboost_params", {}))  # type: ignore[no-any-return]
    elif model_type == "xgboost":
        mlflow.xgboost.autolog(model_format="json")
        return XGBClassifier(device="cuda", **kwargs.get("catboost_params", {}))  # type: ignore[no-any-return]
    elif model_type == "catboost":
        return CatBoostClassifier(task_type="GPU", **kwargs.get("catboost_params", {}))  # type: ignore[no-any-return]
    elif model_type == "lightgbm":
        mlflow.lightgbm.autolog()
        return LGBMClassifier(device="gpu", **kwargs.get("catboost_params", {}))  # type: ignore[no-any-return]
    else:
        raise ValueError(f"{model_type=} is not supported")
