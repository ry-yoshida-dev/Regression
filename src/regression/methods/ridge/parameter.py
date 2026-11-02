from dataclasses import dataclass
from typing import override

from sklearn.linear_model import Ridge

from ...feature_scaling import FeatureScaling
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor
from ...random_state import DEFAULT_RANDOM_STATE


@dataclass(frozen=True)
class RidgeParameters(RegressionParameters):
    """
    Parameters for scikit-learn's Ridge.

    Attributes
    ----------
    alpha : float
        Regularization strength.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    max_iter : int | None
        Maximum number of solver iterations; solver default when None.
    tol : float
        Stopping tolerance.
    scaling : FeatureScaling
        Feature scaling; the penalty depends on feature scale.
    random_state : int | None
        Seed for the solvers that shuffle the data.
    """

    alpha: float = 1.0
    is_intercept_fitted: bool = True
    max_iter: int | None = None
    tol: float = 1e-4
    scaling: FeatureScaling = FeatureScaling.STANDARD
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.max_iter is not None and self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0 when set")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted ridge regression processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator = Ridge(
            alpha=self.alpha,
            fit_intercept=self.is_intercept_fitted,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        return RegressionProcessor(
            method=RegressionMethod.RIDGE,
            estimator=self.scaling.build_pipeline(estimator),
        )
