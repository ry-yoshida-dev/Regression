from dataclasses import dataclass
from typing import override

from sklearn.linear_model import ElasticNet

from ...feature_scaling import FeatureScaling
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor


@dataclass(frozen=True)
class ElasticNetParameters(RegressionParameters):
    """
    Parameters for scikit-learn's ElasticNet.

    Attributes
    ----------
    alpha : float
        Regularization strength; use LinearRegressionParameters for no penalty.
    l1_ratio : float
        Mixing of the penalties: 0 is pure L2 and 1 is pure L1.
    is_intercept_fitted : bool
        Whether to fit an intercept.
    is_coefficient_positive : bool
        Whether coefficients are constrained to be non-negative.
    max_iter : int
        Maximum number of coordinate descent iterations.
    tol : float
        Stopping tolerance.
    scaling : FeatureScaling
        Feature scaling; the penalty depends on feature scale.
    """

    alpha: float = 1.0
    l1_ratio: float = 0.5
    is_intercept_fitted: bool = True
    is_coefficient_positive: bool = False
    max_iter: int = 1000
    tol: float = 1e-4
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be greater than 0")
        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be in the range [0, 1]")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted elastic net processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.is_intercept_fitted,
            positive=self.is_coefficient_positive,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        return RegressionProcessor(
            method=RegressionMethod.ELASTIC_NET,
            estimator=self.scaling.build_pipeline(estimator),
        )
