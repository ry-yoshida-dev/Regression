# Tests

## Overview

Unit tests for the `regression` package: data validation, metrics checked by hand and against scikit-learn, parameter validation and forwarding, processor behavior for all methods, splitters, and end-to-end evaluation.

## Components

| File | Description |
| --- | --- |
| [`helpers.py`](helpers.py) | Shared synthetic datasets. |
| [`test_dataset.py`](test_dataset.py) | Tests for `SupervisedDataset` validation and subsetting. |
| [`test_metrics.py`](test_metrics.py) | Tests for `RegressionMetrics`, including undefined metrics. |
| [`test_parameters.py`](test_parameters.py) | Tests for parameter validation, forwarding to estimators, and `FeatureScaling`. |
| [`test_processors.py`](test_processors.py) | Tests for fit, predict, and evaluate across all methods. |
| [`test_splitters.py`](test_splitters.py) | Tests for partition invariants of every splitter. |
| [`test_evaluator.py`](test_evaluator.py) | Tests for `RegressionEvaluator`, reports, comparisons, and absence of test-to-train leakage. |

## Examples

```bash
python -m pytest tests/ -v
```
