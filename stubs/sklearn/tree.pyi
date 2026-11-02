class DecisionTreeRegressor:
    criterion: str
    max_depth: int | None

    def __init__(
        self,
        *,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int | None = None,
        max_leaf_nodes: int | None = None,
        ccp_alpha: float = 0.0,
    ) -> None: ...
