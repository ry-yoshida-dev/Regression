"""Shared type aliases for regression data."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

type NumericArray = NDArray[np.integer | np.floating]
type FloatArray = NDArray[np.floating]
type TargetArray = NDArray[np.integer | np.floating]
type IndexArray = NDArray[np.intp]
type GroupArray = NDArray[np.integer | np.str_]
