from __future__ import annotations

import math
import unittest

import numpy as np
from sklearn.metrics import explained_variance_score, r2_score

from regression import RegressionMetric, RegressionMetrics


class TestRegressionMetrics(unittest.TestCase):
    def test_hand_computed_values(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 3.0, 2.0, 8.0])
        metrics = RegressionMetrics.compute(y_true, y_pred)
        self.assertEqual(metrics.num_samples, 4)
        self.assertAlmostEqual(metrics.mae, 1.5)
        self.assertAlmostEqual(metrics.mse, 4.5)
        self.assertAlmostEqual(metrics.rmse, math.sqrt(4.5))
        self.assertAlmostEqual(metrics.median_absolute_error, 1.0)
        self.assertAlmostEqual(metrics.max_error, 4.0)
        self.assertAlmostEqual(metrics.mape or 0.0, (0 + 0.5 + 1 / 3 + 1) / 4)

    def test_agrees_with_sklearn(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.3, 2.8, 4.4])
        metrics = RegressionMetrics.compute(y_true, y_pred)
        self.assertAlmostEqual(metrics.r2 or 0.0, r2_score(y_true, y_pred))
        self.assertAlmostEqual(
            metrics.explained_variance or 0.0,
            explained_variance_score(y_true, y_pred),
        )

    def test_constant_targets_leave_r2_undefined(self) -> None:
        metrics = RegressionMetrics.compute(np.array([2.0, 2.0]), np.array([1.0, 3.0]))
        self.assertIsNone(metrics.r2)
        self.assertIsNone(metrics.explained_variance)
        self.assertAlmostEqual(metrics.mae, 1.0)

    def test_single_sample_leaves_r2_undefined(self) -> None:
        metrics = RegressionMetrics.compute(np.array([2.0]), np.array([1.0]))
        self.assertIsNone(metrics.r2)

    def test_zero_target_leaves_mape_undefined(self) -> None:
        metrics = RegressionMetrics.compute(np.array([0.0, 1.0]), np.array([0.1, 1.0]))
        self.assertIsNone(metrics.mape)
        self.assertIsNotNone(metrics.r2)

    def test_misaligned_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            _ = RegressionMetrics.compute(np.array([1.0, 2.0]), np.array([1.0]))
        with self.assertRaises(ValueError):
            _ = RegressionMetrics.compute(np.zeros(0), np.zeros(0))

    def test_every_metric_has_a_getter(self) -> None:
        metrics = RegressionMetrics.compute(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        for metric in RegressionMetric:
            self.assertIsNotNone(metrics.get(metric))

    def test_only_explained_fraction_metrics_are_higher_better(self) -> None:
        higher_better = {
            metric for metric in RegressionMetric if metric.is_higher_better
        }
        self.assertEqual(
            higher_better, {RegressionMetric.R2, RegressionMetric.EXPLAINED_VARIANCE}
        )


if __name__ == "__main__":
    _ = unittest.main()
