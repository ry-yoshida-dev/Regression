from dataclasses import dataclass
from typing import override

from sklearn.dummy import DummyRegressor

from ...feature_scaling import FeatureScaling
from ...method import RegressionMethod
from ...parameter import RegressionParameters
from ...processor import RegressionProcessor
from .strategy import BaselineStrategy


@dataclass(frozen=True)
class BaselineParameters(RegressionParameters):
    """
    Parameters for a reference regressor that ignores the features.

    A model is only useful if it beats this reference, which is what makes
    error magnitudes like RMSE interpretable.

    Attributes
    ----------
    strategy : BaselineStrategy
        The prediction rule.
    """

    strategy: BaselineStrategy = BaselineStrategy.MEAN

    @override
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted baseline processor.

        Returns
        -------
        RegressionProcessor
            The processor.
        """
        estimator = DummyRegressor(strategy=self.strategy.value)
        return RegressionProcessor(
            method=RegressionMethod.BASELINE,
            estimator=FeatureScaling.NONE.build_pipeline(estimator),
        )
