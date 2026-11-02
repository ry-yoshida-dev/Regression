from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .processor import RegressionProcessor


class RegressionParameters(ABC):
    """
    Base class for regression parameters that build a processor.
    """

    @abstractmethod
    def build_processor(self) -> RegressionProcessor:
        """
        Build an unfitted regression processor.

        Returns
        -------
        RegressionProcessor
            The regression processor.
        """
