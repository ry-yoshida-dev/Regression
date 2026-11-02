# metrics

## Overview

Performance metrics computed from true and predicted values. Metrics that cannot be computed for an input are `None` rather than a placeholder number.

## Components

| Component | Description |
|-----------|-------------|
| [values.py](./values.py) | `RegressionMetrics`, all metrics for one set of samples. |
| [metric.py](./metric.py) | `RegressionMetric` enum of scalar metrics, with their direction of improvement. |
