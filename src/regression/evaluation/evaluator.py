from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..dataset import SupervisedDataset
from ..methods import BaselineParameters
from ..metrics import RegressionMetric
from ..parameter import RegressionParameters
from ..types import NumericArray, TargetArray
from .comparison import ModelComparison
from .data_split import DataSplit
from .fold import FoldEvaluation
from .model_evaluation import ModelEvaluation
from .ranked_model import RankedModel
from .report import EvaluationReport
from .splitter import DatasetSplitter
from .splitters import KFoldSplitter


@dataclass(frozen=True)
class RegressionEvaluator:
    """
    Estimates generalization performance by training on splits of a dataset.

    Each fold builds a fresh processor from the parameters, so nothing carries
    over between folds and all preprocessing is fitted on training samples only.

    Attributes
    ----------
    splitter : DatasetSplitter
        The strategy that partitions the dataset.
    baseline : RegressionParameters | None
        The reference model evaluated on the same splits; None to skip it.
    """

    splitter: DatasetSplitter = field(default_factory=KFoldSplitter)
    baseline: RegressionParameters | None = field(default_factory=BaselineParameters)

    def evaluate(
        self,
        parameters: RegressionParameters,
        X: NumericArray,
        y: TargetArray,
    ) -> EvaluationReport:
        """
        Evaluate one model configuration.

        Parameters
        ----------
        parameters : RegressionParameters
            The model configuration.
        X : NumericArray
            The feature matrix.
        y : TargetArray
            The target values.

        Returns
        -------
        EvaluationReport
            The per-fold results and the baseline.

        Raises
        ------
        ValueError
            If the data is invalid.
        """
        dataset = SupervisedDataset(X=X, y=y)
        splits = self.splitter.split(dataset)
        return EvaluationReport(
            model=self._evaluate_on_splits(parameters, dataset, splits),
            baseline=self._evaluate_baseline(dataset, splits),
        )

    def compare(
        self,
        candidates: Mapping[str, RegressionParameters],
        X: NumericArray,
        y: TargetArray,
        metric: RegressionMetric = RegressionMetric.RMSE,
    ) -> ModelComparison:
        """
        Rank model configurations evaluated on identical splits.

        Parameters
        ----------
        candidates : Mapping[str, RegressionParameters]
            The model configurations by name.
        X : NumericArray
            The feature matrix.
        y : TargetArray
            The target values.
        metric : RegressionMetric
            The metric to rank by, using its mean over the test partitions.

        Returns
        -------
        ModelComparison
            The candidates, best first.

        Raises
        ------
        ValueError
            If there are no candidates, the data is invalid, or the metric is
            undefined in any test fold of a candidate.
        """
        if not candidates:
            raise ValueError("candidates must not be empty")
        dataset = SupervisedDataset(X=X, y=y)
        splits = self.splitter.split(dataset)
        evaluated = [
            self._rank_candidate(name, parameters, dataset, splits, metric)
            for name, parameters in candidates.items()
        ]
        ranking = sorted(
            evaluated,
            key=lambda entry: entry.score.mean,
            reverse=metric.is_higher_better,
        )
        return ModelComparison(
            metric=metric,
            ranking=tuple(ranking),
            baseline=self._evaluate_baseline(dataset, splits),
        )

    def _rank_candidate(
        self,
        name: str,
        parameters: RegressionParameters,
        dataset: SupervisedDataset,
        splits: tuple[DataSplit, ...],
        metric: RegressionMetric,
    ) -> RankedModel:
        evaluation = self._evaluate_on_splits(parameters, dataset, splits)
        score = evaluation.summarize(metric)
        if score.is_partial:
            raise ValueError(
                f"cannot rank by {metric.value}: defined in only {score.num_folds} of {score.num_total_folds} test folds"
            )
        return RankedModel(name=name, evaluation=evaluation, score=score)

    def _evaluate_baseline(
        self,
        dataset: SupervisedDataset,
        splits: tuple[DataSplit, ...],
    ) -> ModelEvaluation | None:
        if self.baseline is None:
            return None
        return self._evaluate_on_splits(self.baseline, dataset, splits)

    @staticmethod
    def _evaluate_on_splits(
        parameters: RegressionParameters,
        dataset: SupervisedDataset,
        splits: tuple[DataSplit, ...],
    ) -> ModelEvaluation:
        method = parameters.build_processor().method
        folds = tuple(
            RegressionEvaluator._evaluate_fold(parameters, dataset, split, fold_index)
            for fold_index, split in enumerate(splits)
        )
        return ModelEvaluation(method=method, folds=folds)

    @staticmethod
    def _evaluate_fold(
        parameters: RegressionParameters,
        dataset: SupervisedDataset,
        split: DataSplit,
        fold_index: int,
    ) -> FoldEvaluation:
        train = dataset.subset(split.train_indices)
        test = dataset.subset(split.test_indices)
        processor = parameters.build_processor()
        processor.fit(train.X, train.y)
        return FoldEvaluation(
            fold_index=fold_index,
            train_metrics=processor.evaluate(train.X, train.y),
            test_metrics=processor.evaluate(test.X, test.y),
        )
