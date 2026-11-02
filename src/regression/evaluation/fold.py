from dataclasses import dataclass

from ..metrics import RegressionMetrics
from .partition import EvaluationPartition


@dataclass(frozen=True)
class FoldEvaluation:
    """
    Performance of a model trained and tested on one split.

    Attributes
    ----------
    fold_index : int
        Position of the split.
    train_metrics : RegressionMetrics
        Performance on the training samples.
    test_metrics : RegressionMetrics
        Performance on the held-out samples.
    """

    fold_index: int
    train_metrics: RegressionMetrics
    test_metrics: RegressionMetrics

    def get_metrics(self, partition: EvaluationPartition) -> RegressionMetrics:
        """
        Get the metrics of a partition.

        Parameters
        ----------
        partition : EvaluationPartition
            The partition.

        Returns
        -------
        RegressionMetrics
            The metrics measured on that partition.
        """
        match partition:
            case EvaluationPartition.TRAIN:
                return self.train_metrics
            case EvaluationPartition.TEST:
                return self.test_metrics
