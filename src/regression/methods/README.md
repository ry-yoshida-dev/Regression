# methods

## Overview

Parameter classes and enums for each regression method. Every parameter class is frozen, validates its fields on construction, and builds an unfitted processor via `build_processor()`.

## Components

| Component | Description |
|-----------|-------------|
| [linear_regression/](./linear_regression/README.md) | Ordinary least squares parameters. |
| [ridge/](./ridge/README.md) | Ridge parameters. |
| [lasso/](./lasso/README.md) | Lasso parameters. |
| [elastic_net/](./elastic_net/README.md) | Elastic net parameters. |
| [svr/](./svr/README.md) | Support vector regression parameters, kernels, and gamma rules. |
| [k_nearest_neighbors/](./k_nearest_neighbors/README.md) | K-nearest neighbors parameters and related enums. |
| [decision_tree/](./decision_tree/README.md) | Decision tree parameters. |
| [random_forest/](./random_forest/README.md) | Random forest parameters and feature sampling rules. |
| [hist_gradient_boosting/](./hist_gradient_boosting/README.md) | Histogram gradient boosting parameters and losses. |
| [baseline/](./baseline/README.md) | Feature-independent reference regressor. |
