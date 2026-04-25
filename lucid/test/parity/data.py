from typing import Sequence

import numpy as np

_Shape = Sequence[int]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_floats(
    shape: _Shape,
    *,
    seed: int,
    low: float = -2.0,
    high: float = 2.0,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    return _rng(seed).uniform(low, high, size=shape).astype(dtype)


def pos_floats(
    shape: _Shape,
    *,
    seed: int,
    low: float = 0.25,
    high: float = 2.0,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    return _rng(seed).uniform(low, high, size=shape).astype(dtype)


def nonzero_floats(
    shape: _Shape,
    *,
    seed: int,
    min_abs: float = 0.25,
    high: float = 2.0,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    rng = _rng(seed)
    mag = rng.uniform(min_abs, high, size=shape)
    sign = rng.choice([-1.0, 1.0], size=shape)
    return (mag * sign).astype(dtype)


def unit_bounded(
    shape: _Shape, *, seed: int, margin: float = 0.05, dtype: np.dtype = np.float64
) -> np.ndarray:
    high = 1.0 - margin
    return _rng(seed).uniform(-high, high, size=shape).astype(dtype)


def prob_simplex(
    shape: _Shape, *, seed: int, dtype: np.dtype = np.float64
) -> np.ndarray:
    raw = np.abs(_rng(seed).standard_normal(size=shape)) + 0.001
    return (raw / raw.sum(axis=-1, keepdims=True)).astype(dtype)


def logits(shape: _Shape, *, seed: int, dtype: np.dtype = np.float64) -> np.ndarray:
    return _rng(seed).standard_normal(size=shape).astype(dtype)


def int_array(
    shape: _Shape,
    *,
    seed: int,
    low: int = -5,
    high: int = 5,
    dtype: np.dtype = np.int64,
) -> np.ndarray:
    return _rng(seed).integers(low, high, size=shape).astype(dtype)


def bool_array(shape: _Shape, *, seed: int) -> np.ndarray:
    return _rng(seed).integers(0, 2, size=shape).astype(bool)


def perm_indices(n: int, *, seed: int) -> np.ndarray:
    return _rng(seed).permutation(n).astype(np.int64)


def image(
    batch: int,
    channels: int,
    height: int,
    width: int,
    *,
    seed: int,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    return (
        _rng(seed).standard_normal(size=(batch, channels, height, width)).astype(dtype)
    )


def seq(
    batch: int, length: int, features: int, *, seed: int, dtype: np.dtype = np.float64
) -> np.ndarray:
    return _rng(seed).standard_normal(size=(batch, length, features)).astype(dtype)


def cat_indices(batch: int, num_classes: int, *, seed: int) -> np.ndarray:
    return _rng(seed).integers(0, num_classes, size=(batch,)).astype(np.int64)


def scalar(value: float = 1.7, *, dtype: np.dtype = np.float64) -> np.ndarray:
    return np.array(value, dtype=dtype)
