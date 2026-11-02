from dataclasses import dataclass
from typing import override

from sklearn.model_selection import KFold

from ...dataset import SupervisedDataset
from ...random_state import DEFAULT_RANDOM_STATE
from ..data_split import DataSplit
from ..splitter import DatasetSplitter


@dataclass(frozen=True)
class KFoldSplitter(DatasetSplitter):
    """
    K-fold cross-validation.

    Attributes
    ----------
    n_splits : int
        Number of folds.
    is_shuffled : bool
        Whether samples are shuffled before folding.
    random_state : int | None
        Seed for the shuffle; ignored when is_shuffled is False.
    """

    n_splits: int = 5
    is_shuffled: bool = True
    random_state: int | None = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")

    @override
    def split(self, dataset: SupervisedDataset) -> tuple[DataSplit, ...]:
        """
        Partition a dataset into folds.

        Parameters
        ----------
        dataset : SupervisedDataset
            The dataset to partition.

        Returns
        -------
        tuple[DataSplit, ...]
            One split per fold.
        """
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=self.is_shuffled,
            random_state=self.random_state if self.is_shuffled else None,
        )
        return tuple(
            DataSplit(train_indices=train, test_indices=test)
            for train, test in splitter.split(dataset.X)
        )
