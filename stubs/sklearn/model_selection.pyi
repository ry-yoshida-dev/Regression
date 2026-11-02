from collections.abc import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray

class ShuffleSplit:
    def __init__(
        self,
        n_splits: int = 10,
        *,
        test_size: float | None = None,
        random_state: int | None = None,
    ) -> None: ...
    def split(
        self, X: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class KFold:
    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None: ...
    def split(
        self, X: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class GroupKFold:
    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None: ...
    def split(
        self, X: ArrayLike, y: ArrayLike, groups: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...

class TimeSeriesSplit:
    def __init__(
        self,
        n_splits: int = 5,
        *,
        max_train_size: int | None = None,
        test_size: int | None = None,
        gap: int = 0,
    ) -> None: ...
    def split(
        self, X: ArrayLike
    ) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]: ...
