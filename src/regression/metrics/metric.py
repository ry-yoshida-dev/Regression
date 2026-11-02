from __future__ import annotations

from enum import Enum


class RegressionMetric(Enum):
    """
    Scalar performance metric of a regressor.

    Attributes
    ----------
    MAE
        Mean absolute error.
    MSE
        Mean squared error.
    RMSE
        Root mean squared error.
    MEDIAN_ABSOLUTE_ERROR
        Median absolute error, robust to outliers.
    MAX_ERROR
        Largest absolute error.
    R2
        Coefficient of determination; undefined for constant targets.
    EXPLAINED_VARIANCE
        Explained variance score, which ignores systematic bias; undefined for
        constant targets.
    MAPE
        Mean absolute percentage error as a fraction; undefined when a target is
        zero.
    """

    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    MEDIAN_ABSOLUTE_ERROR = "median_absolute_error"
    MAX_ERROR = "max_error"
    R2 = "r2"
    EXPLAINED_VARIANCE = "explained_variance"
    MAPE = "mape"

    @property
    def is_higher_better(self) -> bool:
        """
        Check the direction in which this metric improves.

        Returns
        -------
        bool
            True if larger values indicate better performance.
        """
        match self:
            case RegressionMetric.R2 | RegressionMetric.EXPLAINED_VARIANCE:
                return True
            case _:
                return False
