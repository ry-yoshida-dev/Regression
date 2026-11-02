import numpy as np
from numpy.typing import ArrayLike, NDArray

class Pipeline:
    steps: list[tuple[str, object]]
    named_steps: dict[str, object]

    def __init__(self, steps: list[tuple[str, object]]) -> None: ...
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Pipeline: ...
    def predict(self, X: ArrayLike) -> NDArray[np.floating]: ...
