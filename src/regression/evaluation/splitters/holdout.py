from dataclasses import dataclass
from typing import override

from sklearn.model_selection import ShuffleSplit

from ...dataset import SupervisedDataset
from ...random_state import DEFAULT_RANDOM_STATE
from ..data_split import DataSplit
from ..splitter import DatasetSplitter


@dataclass(frozen=True)
class HoldoutSplitter(DatasetSplitter):
    """
    Single random train/test partition.

    Attributes
    ----------
    test_fraction : float
        Fraction of the samples in the test partition.
    random_state : int | None
        Seed for the partition.
    """

    test_fraction: float = 0.2
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be in the range (0, 1)")

    @override
    def split(self, dataset: SupervisedDataset) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into one train/test split.

        Parameters
        ----------
        dataset : SupervisedDataset
            The dataset to partition.

        Returns
        -------
        tuple[DataSplit, ...]
            The single split.
        """
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=self.test_fraction,
            random_state=self.random_state,
        )
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in splitter.split(dataset.X)
        )
