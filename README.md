# Regression

## Overview

Regression (`regression`) is a Python package that wraps scikit-learn regressors behind a small, consistent API and adds a performance evaluation layer. Parameter dataclasses validate their fields and build `RegressionProcessor` instances with `fit`, `predict`, and `evaluate`; `RegressionEvaluator` estimates generalization performance by cross-validation and compares configurations.

Supported methods are linear regression, ridge, lasso, elastic net, support vector regression, k-nearest neighbors, decision trees, random forests, and histogram gradient boosting. Everything is exported from the top-level `regression` package.

For module-level design notes, see [src/regression/README.md](src/regression/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## Quick example

Train a model and measure it on separate test data:

```python
from regression import RandomForestParameters

processor = RandomForestParameters(n_estimators=200).build_processor()
processor.fit(X_train, y_train)
predictions = processor.predict(X_test)

metrics = processor.evaluate(X_test, y_test)
print(metrics.rmse, metrics.r2)
```

Cross-validate a configuration against a feature-independent baseline:

```python
from regression import RegressionEvaluator, RegressionMetric, RidgeParameters

report = RegressionEvaluator().evaluate(RidgeParameters(alpha=1.0), X, y)
print(report.format_table())
print(report.model.summarize(RegressionMetric.RMSE).mean)
print(report.get_baseline_improvement(RegressionMetric.RMSE))
```

Rank configurations on identical splits, by RMSE unless another metric is given:

```python
from regression import (
    LassoParameters,
    RandomForestParameters,
    RegressionEvaluator,
    RidgeParameters,
)

comparison = RegressionEvaluator().compare(
    {
        "ridge": RidgeParameters(),
        "lasso": LassoParameters(alpha=0.1),
        "forest": RandomForestParameters(),
    },
    X,
    y,
)
print(comparison.best.name)
print(comparison.format_table())
```

Keep correlated samples together, or respect temporal order, by choosing a splitter:

```python
from regression import GroupKFoldSplitter, RegressionEvaluator, TimeSeriesSplitter

by_subject = RegressionEvaluator(splitter=GroupKFoldSplitter(groups=subject_ids))
forward = RegressionEvaluator(splitter=TimeSeriesSplitter(n_splits=5, gap=1))
```

## Notes

- Scaling is part of the estimator pipeline, and every fold builds a fresh processor, so test samples never influence preprocessing. Regularized and distance-based methods default to `FeatureScaling.STANDARD`.
- Targets must be finite numbers with shape `(n_samples,)`. Multi-output targets are not supported.
- Metrics that cannot be computed are `None`, not a misleading number: `r2` and `explained_variance` for constant targets, and `mape` when a target is zero. When that affects only some folds, summaries average the remaining folds and report their coverage, such as `(1/5 folds)`, and `compare` refuses to rank by that metric.
- `MetricSummary.std` is `None` for a single holdout split: the variability is unknown, not zero.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run basedpyright src tests
uv run pytest
```

The same checks run in CI on every push to `main` and every pull request.

## License

Released under the [MIT License](LICENSE).
