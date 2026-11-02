from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from regression.types import NumericArray


def linear_data(
    *,
    num_samples: int = 120,
    noise_scale: float = 0.05,
    seed: int = 0,
) -> tuple[NumericArray, NDArray[np.float64]]:
    """Return 3-D features and a linear target with small Gaussian noise."""
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((num_samples, 3))
    targets = (
        features @ np.array([2.0, -1.0, 0.5])
        + 3.0
        + noise_scale * rng.standard_normal(num_samples)
    )
    return features, targets


def noise_data(
    *,
    num_samples: int = 200,
    seed: int = 0,
) -> tuple[NumericArray, NDArray[np.float64]]:
    """Return features independent of the target."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_samples, 3)), rng.standard_normal(num_samples)
