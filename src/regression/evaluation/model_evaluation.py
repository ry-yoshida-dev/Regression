from __future__ import annotations

from dataclasses import dataclass

from ..method import RegressionMethod
from ..metrics import RegressionMetric
from .fold import FoldEvaluation
from .metric_summary import MetricSummary
from .partition import EvaluationPartition


@dataclass(frozen=True)
class ModelEvaluation:
    """
    Performance of one model configuration across all folds.

    Attributes
    ----------
    method : RegressionMethod
        The evaluated regression method.
    folds : tuple[FoldEvaluation, ...]
        The result of each fold.
    """

    method: RegressionMethod
    folds: tuple[FoldEvaluation, ...]

    def is_defined(
        self,
        metric: RegressionMetric,
        partition: EvaluationPartition = EvaluationPartition.TEST,
    ) -> bool:
        """
        Check whether a metric is defined in at least one fold.

        Parameters
        ----------
        metric : RegressionMetric
            The metric.
        partition : EvaluationPartition
            The partition to look at.

        Returns
        -------
        bool
            True if summarize would succeed for this metric and partition.
        """
        return bool(self._collect_values(metric, partition))

    def summarize(
        self,
        metric: RegressionMetric,
        partition: EvaluationPartition = EvaluationPartition.TEST,
    ) -> MetricSummary:
        """
        Summarize a metric over the folds in which it is defined.

        Parameters
        ----------
        metric : RegressionMetric
            The metric.
        partition : EvaluationPartition
            The partition to look at.

        Returns
        -------
        MetricSummary
            The summary.

        Raises
        ------
        ValueError
            If the metric is undefined in every fold.
        """
        values = self._collect_values(metric, partition)
        if not values:
            raise ValueError(
                f"{metric.value} is undefined for {self.method.value} on the {partition.value} partition"
            )
        return MetricSummary.from_values(values, num_total_folds=len(self.folds))

    def _collect_values(
        self,
        metric: RegressionMetric,
        partition: EvaluationPartition,
    ) -> list[float]:
        values = [fold.get_metrics(partition).get(metric) for fold in self.folds]
        return [value for value in values if value is not None]
