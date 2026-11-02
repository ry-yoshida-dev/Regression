from enum import Enum


class RegressionMethod(Enum):
    """
    Regression method implemented by a processor.

    Attributes
    ----------
    LINEAR_REGRESSION
        Ordinary least squares.
    RIDGE
        Linear regression with L2 regularization.
    LASSO
        Linear regression with L1 regularization.
    ELASTIC_NET
        Linear regression with combined L1 and L2 regularization.
    SVR
        Support vector regressor.
    K_NEAREST_NEIGHBORS
        K-nearest neighbors regressor.
    DECISION_TREE
        Decision tree regressor.
    RANDOM_FOREST
        Random forest regressor.
    HIST_GRADIENT_BOOSTING
        Histogram-based gradient boosting regressor.
    BASELINE
        Feature-independent reference regressor.
    """

    LINEAR_REGRESSION = "LinearRegression"
    RIDGE = "Ridge"
    LASSO = "Lasso"
    ELASTIC_NET = "ElasticNet"
    SVR = "SVR"
    K_NEAREST_NEIGHBORS = "KNearestNeighbors"
    DECISION_TREE = "DecisionTree"
    RANDOM_FOREST = "RandomForest"
    HIST_GRADIENT_BOOSTING = "HistGradientBoosting"
    BASELINE = "Baseline"
