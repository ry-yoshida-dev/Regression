"""Structural type for regressors used by RegressionProcessor."""

from __future__ import annotations

from typing import Protocol

from .types import FloatArray, NumericArray, TargetArray


class RegressorLike(Protocol):
    """
    Structural type for regressors.

    Any object with these methods can be wrapped, including custom scikit-learn
    compatible estimators.
    """

    def fit(self, X: NumericArray, y: TargetArray) -> object: ...

    def predict(self, X: NumericArray) -> FloatArray: ...
