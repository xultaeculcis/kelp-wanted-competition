from typing import Any, Protocol, Union

import numpy as np
import pandas as pd


class Estimator(Protocol):
    feature_importances_: np.ndarray  # type: ignore[type-arg]
    estimators_: np.ndarray  # type: ignore[type-arg]

    def fit(
        self,
        x: Union[np.ndarray, pd.DataFrame],  # type: ignore[type-arg]
        y: Union[np.ndarray, pd.Series],  # type: ignore[type-arg]
        **kwargs: Any,
    ) -> None:
        ...

    def predict(self, x: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:  # type: ignore[type-arg]
        ...

    def predict_proba(
        self, x: Union[np.ndarray, pd.DataFrame]  # type: ignore[type-arg]
    ) -> Union[np.ndarray, pd.Series]:  # type: ignore[type-arg]
        ...
