from __future__ import annotations

import unittest

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVR

from regression import (
    DecisionTreeParameters,
    ElasticNetParameters,
    FeatureScaling,
    HistGradientBoostingLoss,
    HistGradientBoostingParameters,
    KNearestNeighborsAlgorithm,
    KNearestNeighborsMetric,
    KNearestNeighborsParameters,
    KNearestNeighborsWeighting,
    LassoParameters,
    RandomForestMaxFeatures,
    RandomForestParameters,
    RidgeParameters,
    SVRGamma,
    SVRParameters,
)


class TestParameterValidation(unittest.TestCase):
    def test_ridge(self) -> None:
        with self.assertRaises(ValueError):
            _ = RidgeParameters(alpha=-1.0)
        with self.assertRaises(ValueError):
            _ = RidgeParameters(max_iter=0)
        self.assertEqual(RidgeParameters(alpha=0.0).alpha, 0.0)

    def test_lasso(self) -> None:
        with self.assertRaises(ValueError):
            _ = LassoParameters(alpha=0.0)
        with self.assertRaises(ValueError):
            _ = LassoParameters(max_iter=0)

    def test_elastic_net(self) -> None:
        with self.assertRaises(ValueError):
            _ = ElasticNetParameters(alpha=0.0)
        with self.assertRaises(ValueError):
            _ = ElasticNetParameters(l1_ratio=1.5)

    def test_svr(self) -> None:
        with self.assertRaises(ValueError):
            _ = SVRParameters(C=0.0)
        with self.assertRaises(ValueError):
            _ = SVRParameters(epsilon=-0.1)
        with self.assertRaises(ValueError):
            _ = SVRParameters(gamma=0.0)

    def test_k_nearest_neighbors(self) -> None:
        with self.assertRaises(ValueError):
            _ = KNearestNeighborsParameters(n_neighbors=0)
        with self.assertRaises(ValueError):
            _ = KNearestNeighborsParameters(
                metric=KNearestNeighborsMetric.COSINE,
                algorithm=KNearestNeighborsAlgorithm.BALL_TREE,
            )

    def test_decision_tree(self) -> None:
        with self.assertRaises(ValueError):
            _ = DecisionTreeParameters(max_depth=0)
        with self.assertRaises(ValueError):
            _ = DecisionTreeParameters(min_samples_split=1)

    def test_random_forest(self) -> None:
        with self.assertRaises(ValueError):
            _ = RandomForestParameters(n_estimators=0)

    def test_hist_gradient_boosting(self) -> None:
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(learning_rate=0.0)
        with self.assertRaises(ValueError):
            _ = HistGradientBoostingParameters(max_bins=256)


class TestParameterForwarding(unittest.TestCase):
    def test_ridge_scales_by_default(self) -> None:
        processor = RidgeParameters(alpha=2.0).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        self.assertIsInstance(processor.estimator.named_steps["scaler"], StandardScaler)
        estimator = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, Ridge)
        self.assertEqual(estimator.alpha, 2.0)

    def test_elastic_net(self) -> None:
        processor = ElasticNetParameters(l1_ratio=0.25).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, ElasticNet)
        self.assertEqual(estimator.l1_ratio, 0.25)

    def test_svr_gamma_rule_and_value(self) -> None:
        for gamma, expected in ((SVRGamma.AUTO, "auto"), (0.25, 0.25)):
            with self.subTest(gamma=gamma):
                processor = SVRParameters(gamma=gamma).build_processor()
                assert isinstance(processor.estimator, Pipeline)
                estimator = processor.estimator.named_steps["estimator"]
                assert isinstance(estimator, SVR)
                self.assertEqual(estimator.gamma, expected)
                self.assertEqual(estimator.max_iter, -1)

    def test_k_nearest_neighbors(self) -> None:
        processor = KNearestNeighborsParameters(
            n_neighbors=3, weighting=KNearestNeighborsWeighting.DISTANCE
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, KNeighborsRegressor)
        self.assertEqual(estimator.n_neighbors, 3)
        self.assertEqual(estimator.weights, "distance")

    def test_random_forest_max_features(self) -> None:
        for max_features, expected in (
            (None, None),
            (RandomForestMaxFeatures.SQRT, "sqrt"),
        ):
            with self.subTest(max_features=max_features):
                processor = RandomForestParameters(
                    max_features=max_features
                ).build_processor()
                assert isinstance(processor.estimator, Pipeline)
                estimator = processor.estimator.named_steps["estimator"]
                assert isinstance(estimator, RandomForestRegressor)
                self.assertEqual(estimator.max_features, expected)

    def test_hist_gradient_boosting_loss(self) -> None:
        processor = HistGradientBoostingParameters(
            loss=HistGradientBoostingLoss.ABSOLUTE_ERROR
        ).build_processor()
        assert isinstance(processor.estimator, Pipeline)
        estimator = processor.estimator.named_steps["estimator"]
        assert isinstance(estimator, HistGradientBoostingRegressor)
        self.assertEqual(estimator.loss, "absolute_error")
        self.assertFalse(estimator.early_stopping)


class TestFeatureScaling(unittest.TestCase):
    def test_scaler_types(self) -> None:
        estimator = object()
        self.assertIsInstance(
            FeatureScaling.MIN_MAX.build_pipeline(estimator).named_steps["scaler"],
            MinMaxScaler,
        )
        self.assertIsInstance(
            FeatureScaling.ROBUST.build_pipeline(estimator).named_steps["scaler"],
            RobustScaler,
        )
        self.assertEqual(
            FeatureScaling.NONE.build_pipeline(estimator).steps[0],
            ("scaler", "passthrough"),
        )


if __name__ == "__main__":
    _ = unittest.main()
