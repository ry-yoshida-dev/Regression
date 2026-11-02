# regression

## Overview

This module provides a unified interface for regression. Layers depend only on those listed before them: shared types and `dataset.py`; `metrics/`, which computes performance from predictions; `processor.py`, which wraps estimators; `methods/`, whose parameter classes build processors; and `evaluation/`, which trains models from parameter classes on dataset splits.

## Components

| Component | Description |
|-----------|-------------|
| [types.py](./types.py) | Shared type aliases. |
| [random_state.py](./random_state.py) | `DEFAULT_RANDOM_STATE`, the default seed of estimators and splitters. |
| [method.py](./method.py) | `RegressionMethod` enum of available methods. |
| [dataset.py](./dataset.py) | `SupervisedDataset`, the validated pair of features and targets. |
| [feature_scaling.py](./feature_scaling.py) | `FeatureScaling` enum that builds the scaler and estimator pipeline. |
| [tree_criterion.py](./tree_criterion.py) | `TreeCriterion` enum, shared by decision trees and random forests. |
| [parameter.py](./parameter.py) | Abstract base class for regression parameters. |
| [processor.py](./processor.py) | `RegressionProcessor`, with validation, fitted state, `predict`, and `evaluate`. |
| [protocol.py](./protocol.py) | `RegressorLike`, the structural type of wrapped estimators. |
| [methods/](./methods/README.md) | Per-method parameter classes and enums. |
| [metrics/](./metrics/README.md) | Performance metrics computed from true and predicted values. |
| [evaluation/](./evaluation/README.md) | Splitters, cross-validated evaluation, reports, and model comparison. |
