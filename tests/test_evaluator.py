from __future__ import annotations

import math
import unittest

import numpy as np

from regression import (
    DecisionTreeParameters,
    EvaluationPartition,
    GroupKFoldSplitter,
    HoldoutSplitter,
    KFoldSplitter,
    KNearestNeighborsParameters,
    LinearRegressionParameters,
    MetricSummary,
    RandomForestParameters,
    RegressionEvaluator,
    RegressionMetric,
    RidgeParameters,
    SupervisedDataset,
    TimeSeriesSplitter,
)
from regression.types import NumericArray, TargetArray
from tests.helpers import linear_data, noise_data


def _targets_with_zero_in_some_test_folds() -> tuple[NumericArray, TargetArray]:
    X, y = linear_data()
    y = y + 100.0
    splits = KFoldSplitter().split(SupervisedDataset(X, y))
    for split in splits[:4]:
        y[split.test_indices[0]] = 0.0
    return X, y


class TestEvaluate(unittest.TestCase):
    def test_linear_signal_beats_baseline(self) -> None:
        X, y = linear_data()
        report = RegressionEvaluator().evaluate(LinearRegressionParameters(), X, y)
        self.assertEqual(len(report.model.folds), 5)
        self.assertGreater(report.model.summarize(RegressionMetric.R2).mean, 0.99)
        self.assertGreater(report.get_baseline_improvement(RegressionMetric.RMSE), 1.0)
        self.assertGreater(report.get_baseline_improvement(RegressionMetric.R2), 0.9)

    def test_noise_does_not_beat_baseline(self) -> None:
        X, y = noise_data()
        report = RegressionEvaluator().evaluate(
            RandomForestParameters(n_estimators=20), X, y
        )
        self.assertLess(report.get_baseline_improvement(RegressionMetric.RMSE), 0.0)

    def test_overfitting_shows_as_train_test_gap(self) -> None:
        X, y = noise_data()
        model = (
            RegressionEvaluator(baseline=None)
            .evaluate(DecisionTreeParameters(), X, y)
            .model
        )
        train = model.summarize(RegressionMetric.R2, EvaluationPartition.TRAIN)
        test = model.summarize(RegressionMetric.R2)
        self.assertEqual(train.mean, 1.0)
        self.assertLess(test.mean, 0.0)

    def test_scaling_is_fitted_on_training_samples_only(self) -> None:
        X, y = linear_data()
        splitter = HoldoutSplitter(test_fraction=0.25)
        test_indices = splitter.split(SupervisedDataset(X, y))[0].test_indices
        X_with_test_outlier = X.copy()
        X_with_test_outlier[test_indices[:1], 0] = 1e9
        evaluator = RegressionEvaluator(splitter=splitter, baseline=None)
        clean = evaluator.evaluate(KNearestNeighborsParameters(), X, y)
        perturbed = evaluator.evaluate(
            KNearestNeighborsParameters(), X_with_test_outlier, y
        )
        self.assertEqual(
            clean.model.folds[0].train_metrics,
            perturbed.model.folds[0].train_metrics,
        )

    def test_is_reproducible(self) -> None:
        X, y = noise_data()
        evaluator = RegressionEvaluator()
        first = evaluator.evaluate(RandomForestParameters(n_estimators=5), X, y)
        second = evaluator.evaluate(RandomForestParameters(n_estimators=5), X, y)
        self.assertEqual(first, second)

    def test_zero_targets_leave_mape_undefined(self) -> None:
        X, y = linear_data()
        y[:] = np.round(y)
        y[np.arange(0, y.size, 2)] = 0.0
        report = RegressionEvaluator().evaluate(RidgeParameters(), X, y)
        self.assertFalse(report.model.is_defined(RegressionMetric.MAPE))
        self.assertTrue(report.model.is_defined(RegressionMetric.MAE))
        with self.assertRaises(ValueError):
            _ = report.model.summarize(RegressionMetric.MAPE)

    def test_partially_defined_metric_states_fold_coverage(self) -> None:
        X, y = _targets_with_zero_in_some_test_folds()
        report = RegressionEvaluator().evaluate(RidgeParameters(), X, y)
        summary = report.model.summarize(RegressionMetric.MAPE)
        self.assertTrue(summary.is_partial)
        self.assertEqual((summary.num_folds, summary.num_total_folds), (1, 5))
        self.assertIn("(1/5 folds)", report.format_table())
        self.assertFalse(report.model.summarize(RegressionMetric.RMSE).is_partial)

    def test_no_baseline(self) -> None:
        X, y = linear_data()
        report = RegressionEvaluator(baseline=None).evaluate(RidgeParameters(), X, y)
        self.assertIsNone(report.baseline)
        with self.assertRaises(ValueError):
            _ = report.get_baseline_improvement(RegressionMetric.RMSE)

    def test_holdout_has_unknown_variability(self) -> None:
        X, y = linear_data()
        report = RegressionEvaluator(splitter=HoldoutSplitter()).evaluate(
            RidgeParameters(), X, y
        )
        summary = report.model.summarize(RegressionMetric.RMSE)
        self.assertIsNone(summary.std)
        self.assertEqual(summary.num_folds, 1)

    def test_group_splitter(self) -> None:
        X, y = linear_data()
        report = RegressionEvaluator(
            splitter=GroupKFoldSplitter(groups=np.arange(y.size) % 12, n_splits=3)
        ).evaluate(RidgeParameters(), X, y)
        self.assertEqual(len(report.model.folds), 3)

    def test_time_series_splitter(self) -> None:
        X, y = linear_data()
        report = RegressionEvaluator(splitter=TimeSeriesSplitter(n_splits=3)).evaluate(
            RidgeParameters(), X, y
        )
        self.assertEqual(len(report.model.folds), 3)

    def test_format_table_lists_every_metric(self) -> None:
        X, y = linear_data()
        table = RegressionEvaluator().evaluate(RidgeParameters(), X, y).format_table()
        for metric in RegressionMetric:
            self.assertIn(metric.value, table)


class TestCompare(unittest.TestCase):
    def test_lower_better_default_metric_ranks_ascending(self) -> None:
        X, y = linear_data()
        comparison = RegressionEvaluator().compare(
            {
                "underfit": DecisionTreeParameters(max_depth=1),
                "fit": LinearRegressionParameters(),
            },
            X,
            y,
        )
        self.assertEqual(comparison.metric, RegressionMetric.RMSE)
        self.assertEqual(comparison.best.name, "fit")
        self.assertEqual(
            [entry.name for entry in comparison.ranking], ["fit", "underfit"]
        )
        self.assertIn("(baseline)", comparison.format_table())

    def test_higher_better_metric_ranks_descending(self) -> None:
        X, y = linear_data()
        comparison = RegressionEvaluator().compare(
            {
                "underfit": DecisionTreeParameters(max_depth=1),
                "fit": LinearRegressionParameters(),
            },
            X,
            y,
            metric=RegressionMetric.R2,
        )
        self.assertEqual(comparison.best.name, "fit")
        self.assertGreater(
            comparison.ranking[0].score.mean, comparison.ranking[1].score.mean
        )

    def test_candidates_share_splits(self) -> None:
        X, y = noise_data()
        evaluator = RegressionEvaluator(splitter=KFoldSplitter(n_splits=3))
        comparison = evaluator.compare(
            {"first": DecisionTreeParameters(), "second": DecisionTreeParameters()},
            X,
            y,
        )
        self.assertEqual(
            comparison.ranking[0].evaluation, comparison.ranking[1].evaluation
        )

    def test_partially_defined_metric_cannot_rank(self) -> None:
        X, y = _targets_with_zero_in_some_test_folds()
        with self.assertRaisesRegex(ValueError, "1 of 5 test folds"):
            _ = RegressionEvaluator().compare(
                {"ridge": RidgeParameters()},
                X,
                y,
                metric=RegressionMetric.MAPE,
            )

    def test_empty_candidates_raise(self) -> None:
        X, y = linear_data()
        with self.assertRaises(ValueError):
            _ = RegressionEvaluator().compare({}, X, y)


class TestMetricSummary(unittest.TestCase):
    def test_from_values(self) -> None:
        summary = MetricSummary.from_values([1.0, 3.0], num_total_folds=2)
        self.assertEqual(summary.mean, 2.0)
        self.assertAlmostEqual(summary.std or 0.0, math.sqrt(2.0))
        self.assertEqual((summary.minimum, summary.maximum), (1.0, 3.0))
        self.assertFalse(summary.is_partial)
        self.assertEqual(summary.format(), "2.0000 ± 1.4142")

    def test_partial_format_states_coverage(self) -> None:
        summary = MetricSummary.from_values([1.0, 3.0], num_total_folds=5)
        self.assertTrue(summary.is_partial)
        self.assertEqual(summary.format(), "2.0000 ± 1.4142 (2/5 folds)")

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = MetricSummary.from_values([], num_total_folds=5)

    def test_more_values_than_folds_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = MetricSummary.from_values([1.0, 2.0], num_total_folds=1)


if __name__ == "__main__":
    _ = unittest.main()
