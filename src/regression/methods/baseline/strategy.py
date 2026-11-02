from enum import Enum


class BaselineStrategy(Enum):
    """
    Prediction rule of a feature-independent reference regressor.

    Attributes
    ----------
    MEAN
        Predicts the mean of the training targets.
    MEDIAN
        Predicts the median of the training targets.
    """

    MEAN = "mean"
    MEDIAN = "median"
