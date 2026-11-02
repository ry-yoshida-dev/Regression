from enum import Enum


class TreeCriterion(Enum):
    """
    Split quality criterion, shared by decision trees and random forests.

    Attributes
    ----------
    SQUARED_ERROR
        Mean squared error.
    FRIEDMAN_MSE
        Mean squared error with Friedman's improvement score.
    ABSOLUTE_ERROR
        Mean absolute error.
    POISSON
        Poisson deviance; requires non-negative targets.
    """

    SQUARED_ERROR = "squared_error"
    FRIEDMAN_MSE = "friedman_mse"
    ABSOLUTE_ERROR = "absolute_error"
    POISSON = "poisson"
