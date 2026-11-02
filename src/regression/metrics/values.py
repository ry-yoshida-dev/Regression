from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from ..types import FloatArray, TargetArray
from .metric import RegressionMetric


@dataclass(frozen=True)
class RegressionMetrics:
    """
    Performance of a regressor on one set of samples.

    Attributes
    ----------
    num_samples : int
        Number of evaluated samples.
    mae : float
        Mean absolute error.
    mse : float
        Mean squared error.
    rmse : float
        Root mean squared error.
    median_absolute_error : float
        Median absolute error.
    max_error : float
        Largest absolute error.
    r2 : float | None
        Coefficient of determination; None when the targets are constant.
    explained_variance : float | None
        Explained variance score; None when the targets are constant.
    mape : float | None
        Mean absolute percentage error as a fraction; None when a target is zero.
    """

    num_samples: int
    mae: float
    mse: float
    rmse: float
    median_absolute_error: float
    max_error: float
    r2: float | None
    explained_variance: float | None
    mape: float | None

    @classmethod
    def compute(cls, y_true: TargetArray, y_pred: FloatArray) -> RegressionMetrics:
        """
        Compute all metrics from true and predicted values.

        Parameters
        ----------
        y_true : TargetArray
            The true values with shape (n_samples,).
        y_pred : FloatArray
            The predicted values with shape (n_samples,).

        Returns
        -------
        RegressionMetrics
            The computed metrics.

        Raises
        ------
        ValueError
            If the inputs are empty or misaligned.
        """
        if y_true.ndim != 1 or y_true.size == 0:
            raise ValueError(f"y_true must be non-empty and 1-D, got {y_true.shape}")
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"y_pred must have shape {y_true.shape}, got {y_pred.shape}"
            )
        is_target_constant = bool(np.all(np.equal(y_true, y_true[:1])))
        is_target_zero_present = bool(np.any(np.equal(y_true, 0)))
        return cls(
            num_samples=y_true.size,
            mae=float(mean_absolute_error(y_true, y_pred)),
            mse=float(mean_squared_error(y_true, y_pred)),
            rmse=float(root_mean_squared_error(y_true, y_pred)),
            median_absolute_error=float(median_absolute_error(y_true, y_pred)),
            max_error=float(max_error(y_true, y_pred)),
            r2=None if is_target_constant else float(r2_score(y_true, y_pred)),
            explained_variance=(
                None
                if is_target_constant
                else float(explained_variance_score(y_true, y_pred))
            ),
            mape=(
                None
                if is_target_zero_present
                else float(mean_absolute_percentage_error(y_true, y_pred))
            ),
        )

    def get(self, metric: RegressionMetric) -> float | None:
        """
        Get the value of a metric.

        Parameters
        ----------
        metric : RegressionMetric
            The metric to look up.

        Returns
        -------
        float | None
            The value, or None when the metric is undefined for this evaluation.
        """
        match metric:
            case RegressionMetric.MAE:
                return self.mae
            case RegressionMetric.MSE:
                return self.mse
            case RegressionMetric.RMSE:
                return self.rmse
            case RegressionMetric.MEDIAN_ABSOLUTE_ERROR:
                return self.median_absolute_error
            case RegressionMetric.MAX_ERROR:
                return self.max_error
            case RegressionMetric.R2:
                return self.r2
            case RegressionMetric.EXPLAINED_VARIANCE:
                return self.explained_variance
            case RegressionMetric.MAPE:
                return self.mape
