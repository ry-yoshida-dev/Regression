class RandomForestRegressor:
    n_estimators: int
    max_features: str | None
    bootstrap: bool

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | None = None,
        bootstrap: bool = True,
        n_jobs: int | None = None,
        random_state: int | None = None,
    ) -> None: ...

class HistGradientBoostingRegressor:
    loss: str
    learning_rate: float
    early_stopping: bool

    def __init__(
        self,
        loss: str = "squared_error",
        *,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        max_leaf_nodes: int | None = 31,
        max_depth: int | None = None,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        max_bins: int = 255,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-7,
        random_state: int | None = None,
    ) -> None: ...
