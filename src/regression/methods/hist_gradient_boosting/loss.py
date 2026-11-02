from enum import Enum


class HistGradientBoostingLoss(Enum):
    """
    Loss function for scikit-learn's HistGradientBoostingRegressor.

    Attributes
    ----------
    SQUARED_ERROR
        Squared error.
    ABSOLUTE_ERROR
        Absolute error, robust to outliers.
    POISSON
        Poisson deviance; requires non-negative targets.
    GAMMA
        Gamma deviance; requires positive targets.
    """

    SQUARED_ERROR = "squared_error"
    ABSOLUTE_ERROR = "absolute_error"
    POISSON = "poisson"
    GAMMA = "gamma"
