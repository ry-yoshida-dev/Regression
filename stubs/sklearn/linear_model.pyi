class LinearRegression:
    fit_intercept: bool
    positive: bool

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        n_jobs: int | None = None,
        positive: bool = False,
    ) -> None: ...

class Ridge:
    alpha: float

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        fit_intercept: bool = True,
        max_iter: int | None = None,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None: ...

class Lasso:
    alpha: float

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        positive: bool = False,
    ) -> None: ...

class ElasticNet:
    alpha: float
    l1_ratio: float

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        positive: bool = False,
    ) -> None: ...
