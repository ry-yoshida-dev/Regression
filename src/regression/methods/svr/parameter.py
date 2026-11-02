from dataclasses import dataclass
from typing import override

from sklearn.svm import SVR

from ...feature_scaling import FeatureScaling
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor
from .gamma import SVRGamma
from .kernel import SVRKernel


@dataclass(frozen=True)
class SVRParameters(RegressionParameters):
    """
    Parameters for scikit-learn's SVR.

    Attributes
    ----------
    C : float
        Inverse regularization strength.
    epsilon : float
        Half-width of the tube within which errors are not penalized, in target
        units.
    kernel : SVRKernel
        The kernel function.
    degree : int
        Degree of the polynomial kernel.
    gamma : SVRGamma | float
        Kernel coefficient, either a rule or a positive value.
    coef0 : float
        Independent term of the polynomial and sigmoid kernels.
    tol : float
        Stopping tolerance.
    max_iter : int | None
        Maximum number of solver iterations; no limit when None.
    scaling : FeatureScaling
        Feature scaling; kernels depend on feature scale.
    """

    C: float = 1.0
    epsilon: float = 0.1
    kernel: SVRKernel = SVRKernel.RBF
    degree: int = 3
    gamma: SVRGamma | float = SVRGamma.SCALE
    coef0: float = 0.0
    tol: float = 1e-3
    max_iter: int | None = None
    scaling: FeatureScaling = FeatureScaling.STANDARD

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.C <= 0:
            raise ValueError("C must be greater than 0")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.degree < 0:
            raise ValueError("degree must be non-negative")
        if not isinstance(self.gamma, SVRGamma) and self.gamma <= 0:
            raise ValueError("gamma must be greater than 0 when numeric")
        if self.tol <= 0:
            raise ValueError("tol must be greater than 0")
        if self.max_iter is not None and self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0 when set")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted support vector regression processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator = SVR(
            C=self.C,
            epsilon=self.epsilon,
            kernel=self.kernel.value,
            degree=self.degree,
            gamma=self.gamma.value if isinstance(self.gamma, SVRGamma) else self.gamma,
            coef0=self.coef0,
            tol=self.tol,
            max_iter=-1 if self.max_iter is None else self.max_iter,
        )
        return RegressionProcessor(
            method=RegressionMethod.SVR,
            estimator=self.scaling.build_pipeline(estimator),
        )
