from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.exceptions import NotFittedError

from .dataset import SupervisedDataset
from .method import RegressionMethod
from .metrics import RegressionMetrics
from .protocol import RegressorLike
from .types import FloatArray, NumericArray, TargetArray


@dataclass
class RegressionProcessor:
    """
    Unified interface over a fitted-on-demand regressor.

    Attributes
    ----------
    method : RegressionMethod
        The regression method this processor implements.
    estimator : RegressorLike
        The underlying estimator (structural type).
    """

    method: RegressionMethod
    estimator: RegressorLike
    _is_fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, X: NumericArray, y: TargetArray) -> None:
        """
        Fit the regressor.

        Parameters
        ----------
        X : NumericArray
            The feature matrix with shape (n_samples, n_features).
        y : TargetArray
            The target values with shape (n_samples,).

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        self._is_fitted = False
        dataset = SupervisedDataset(X=X, y=y)
        _ = self.estimator.fit(dataset.X, dataset.y)
        self._is_fitted = True

    def predict(self, X: NumericArray) -> FloatArray:
        """
        Predict target values.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.

        Returns
        -------
        FloatArray
            The predicted values with shape (n_samples,).

        Raises
        ------
        NotFittedError
            If fit has not completed successfully yet.
        """
        if not self._is_fitted:
            raise NotFittedError("Processor is not fitted yet. Call fit() first.")
        return self.estimator.predict(X)

    def evaluate(self, X: NumericArray, y: TargetArray) -> RegressionMetrics:
        """
        Measure the performance of the fitted regressor on labeled data.

        Parameters
        ----------
        X : NumericArray
            The feature matrix.
        y : TargetArray
            The true target values.

        Returns
        -------
        RegressionMetrics
            The performance metrics.

        Raises
        ------
        NotFittedError
            If fit has not completed successfully yet.
        ValueError
            If the data is invalid.
        """
        dataset = SupervisedDataset(X=X, y=y)
        return RegressionMetrics.compute(
            y_true=dataset.y,
            y_pred=self.predict(dataset.X),
        )
