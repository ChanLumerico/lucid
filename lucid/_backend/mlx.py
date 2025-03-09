"""
Lucid's backend suppport for MLX array acceleration in order to 
seamlessly incorporate `mlx` into `lucid.

NOTE: alpha build (wip)
"""

from typing import Any

from lucid.types import _ArrayOrScalar, _NumPyArray

try:
    import mlx.core as mx
except ModuleNotFoundError as e:
    print(f"mlx library not installed. Try 'pip install mlx'.")


def to_mlx_array(data: _ArrayOrScalar | _NumPyArray, dtype: Any = None) -> mx.array:
    return mx.array(val=data, dtype=dtype)
