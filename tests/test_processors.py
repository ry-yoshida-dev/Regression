from __future__ import annotations

import unittest

import numpy as np
from sklearn.exceptions import NotFittedError

from regression import (
    BaselineParameters,
    BaselineStrategy,
    DecisionTreeParameters,
    ElasticNetParameters,
    HistGradientBoostingParameters,
    KNearestNeighborsParameters,
    LassoParameters,
    LinearRegressionParameters,
    RandomForestParameters,
    RegressionMethod,
    RegressionParameters,
    RidgeParameters,
    SVRParameters,
)
from tests.helpers import linear_data

ALL_PARAMETERS: tuple[RegressionParameters, ...] = (
    LinearRegressionParameters(),
    RidgeParameters(),
    LassoParameters(alpha=0.01),
    ElasticNetParameters(alpha=0.01),
    SVRParameters(C=10.0),
    KNearestNeighborsParameters(),
    DecisionTreeParameters(),
    RandomForestParameters(n_estimators=10),
    HistGradientBoostingParameters(min_samples_leaf=5),
)


class TestAllProcessors(unittest.TestCase):
    def test_learns_linear_signal(self) -> None:
        X, y = linear_data()
        for parameters in ALL_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                processor = parameters.build_processor()
                processor.fit(X, y)
                self.assertEqual(processor.predict(X).shape, y.shape)
                metrics = processor.evaluate(X, y)
                assert metrics.r2 is not None
                self.assertGreater(metrics.r2, 0.85)

    def test_predict_before_fit_raises(self) -> None:
        X, _ = linear_data()
        for parameters in ALL_PARAMETERS:
            with self.subTest(parameters=type(parameters).__name__):
                with self.assertRaises(NotFittedError):
                    _ = parameters.build_processor().predict(X)

    def test_failed_refit_clears_fitted_state(self) -> None:
        X, y = linear_data()
        processor = LinearRegressionParameters().build_processor()
        processor.fit(X, y)
        with self.assertRaises(ValueError):
            processor.fit(X, y[:-1])
        with self.assertRaises(NotFittedError):
            _ = processor.predict(X)

    def test_linear_regression_recovers_coefficients(self) -> None:
        X, y = linear_data(noise_scale=0.0)
        processor = LinearRegressionParameters().build_processor()
        processor.fit(X, y)
        np.testing.assert_allclose(processor.predict(X), y)

    def test_nan_features_are_supported_by_hist_gradient_boosting(self) -> None:
        X, y = linear_data()
        X[0, 0] = np.nan
        processor = HistGradientBoostingParameters(
            max_iter=5, min_samples_leaf=2
        ).build_processor()
        processor.fit(X, y)
        self.assertEqual(processor.predict(X).shape, y.shape)

    def test_method_is_reported(self) -> None:
        self.assertEqual(SVRParameters().build_processor().method, RegressionMethod.SVR)


class TestBaseline(unittest.TestCase):
    def test_predicts_training_statistic(self) -> None:
        X = np.zeros((3, 1))
        y = np.array([1.0, 2.0, 9.0])
        for strategy, expected in (
            (BaselineStrategy.MEAN, 4.0),
            (BaselineStrategy.MEDIAN, 2.0),
        ):
            with self.subTest(strategy=strategy):
                processor = BaselineParameters(strategy=strategy).build_processor()
                processor.fit(X, y)
                self.assertEqual(processor.predict(X).tolist(), [expected] * 3)


if __name__ == "__main__":
    _ = unittest.main()
