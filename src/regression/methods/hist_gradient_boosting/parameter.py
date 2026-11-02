from dataclasses import dataclass
from typing import override

from sklearn.ensemble import HistGradientBoostingRegressor

from ...feature_scaling import FeatureScaling
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor
from ...random_state import DEFAULT_RANDOM_STATE
from .loss import HistGradientBoostingLoss


@dataclass(frozen=True)
class HistGradientBoostingParameters(RegressionParameters):
    """
    Parameters for scikit-learn's HistGradientBoostingRegressor.

    This method accepts missing feature values (NaN). Early stopping is disabled
    by default, unlike scikit-learn's size-dependent default, so that the number
    of boosting iterations does not silently change with the dataset size.

    Attributes
    ----------
    loss : HistGradientBoostingLoss
        The loss function to optimize.
    learning_rate : float
        Shrinkage applied to each tree.
    max_iter : int
        Maximum number of boosting iterations.
    max_leaf_nodes : int | None
        Maximum number of leaves per tree; no limit when None.
    max_depth : int | None
        Maximum tree depth; no limit when None.
    min_samples_leaf : int
        Minimum number of samples required at a leaf.
    l2_regularization : float
        L2 regularization of leaf values.
    max_bins : int
        Maximum number of histogram bins for non-missing values.
    is_early_stopping_enabled : bool
        Whether to stop when an internal validation split stops improving.
    validation_fraction : float
        Fraction of the training data held out for early stopping.
    n_iter_no_change : int
        Iterations without improvement before early stopping.
    tol : float
        Minimum improvement counted by early stopping.
    random_state : int | None
        Seed for binning subsampling and the early stopping split.
    """

    loss: HistGradientBoostingLoss = HistGradientBoostingLoss.SQUARED_ERROR
    learning_rate: float = 0.1
    max_iter: int = 100
    max_leaf_nodes: int | None = 31
    max_depth: int | None = None
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    max_bins: int = 255
    is_early_stopping_enabled: bool = False
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    tol: float = 1e-7
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be greater than 0")
        if self.max_leaf_nodes is not None and self.max_leaf_nodes < 2:
            raise ValueError("max_leaf_nodes must be at least 2 when set")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be greater than 0 when set")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be greater than 0")
        if self.l2_regularization < 0:
            raise ValueError("l2_regularization must be non-negative")
        if not 2 <= self.max_bins <= 255:
            raise ValueError("max_bins must be in the range [2, 255]")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in the range (0, 1)")
        if self.n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be greater than 0")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted histogram gradient boosting processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator = HistGradientBoostingRegressor(
            loss=self.loss.value,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_bins=self.max_bins,
            early_stopping=self.is_early_stopping_enabled,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
            random_state=self.random_state,
        )
        return RegressionProcessor(
            method=RegressionMethod.HIST_GRADIENT_BOOSTING,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )
