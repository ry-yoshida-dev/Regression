from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricSummary:
    """
    Aggregate of one metric over evaluation folds.

    Attributes
    ----------
    mean : float
        Mean value.
    std : float | None
        Sample standard deviation; None with fewer than two folds, where the
        variability is unknown rather than zero.
    minimum : float
        Smallest value.
    maximum : float
        Largest value.
    num_folds : int
        Number of folds in which the metric is defined.
    num_total_folds : int
        Number of evaluated folds, including those where the metric is undefined.
    """

    mean: float
    std: float | None
    minimum: float
    maximum: float
    num_folds: int
    num_total_folds: int

    @property
    def is_partial(self) -> bool:
        """
        Check whether the metric is undefined in some of the evaluated folds.

        Returns
        -------
        bool
            True if the summary covers only a subset of the folds.
        """
        return self.num_folds < self.num_total_folds

    @classmethod
    def from_values(cls, values: list[float], num_total_folds: int) -> MetricSummary:
        """
        Summarize per-fold metric values.

        Parameters
        ----------
        values : list[float]
            The value of each fold in which the metric is defined.
        num_total_folds : int
            The number of evaluated folds, including undefined ones.

        Returns
        -------
        MetricSummary
            The summary.

        Raises
        ------
        ValueError
            If values is empty or has more entries than num_total_folds.
        """
        if not values:
            raise ValueError("values must not be empty")
        if len(values) > num_total_folds:
            raise ValueError(
                f"got {len(values)} values for {num_total_folds} evaluated folds"
            )
        array = np.array(values, dtype=np.float64)
        return cls(
            mean=float(np.mean(array)),
            std=float(np.std(array, ddof=1)) if array.size > 1 else None,
            minimum=float(np.min(array)),
            maximum=float(np.max(array)),
            num_folds=int(array.size),
            num_total_folds=num_total_folds,
        )

    def format(self) -> str:
        """
        Format the mean and standard deviation for display.

        Returns
        -------
        str
            The formatted summary, which states the fold coverage when the
            metric is undefined in some folds.
        """
        value = (
            f"{self.mean:.4f}"
            if self.std is None
            else f"{self.mean:.4f} ± {self.std:.4f}"
        )
        if self.is_partial:
            return f"{value} ({self.num_folds}/{self.num_total_folds} folds)"
        return value
