from enum import Enum


class SVRGamma(Enum):
    """
    Automatic kernel coefficient rule for scikit-learn's SVR.

    Attributes
    ----------
    SCALE
        1 / (n_features * X.var()).
    AUTO
        1 / n_features.
    """

    SCALE = "scale"
    AUTO = "auto"
