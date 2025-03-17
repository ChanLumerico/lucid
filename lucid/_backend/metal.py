from typing import Any
import numpy as np

from lucid.types import _NumPyArray

try:
    import mlx.core as mx
except ModuleNotFoundError as e:
    print(f"mlx library not installed. Try 'pip install mlx'.")

_MLXArray = mx.array


def is_cpu_op(*tensor_or_any) -> bool:
    return any([getattr(t, "device", None) == "cpu" for t in tensor_or_any])


def is_gpu_op(*tensor_or_any) -> bool:
    return any([getattr(t, "device", None) == "gpu" for t in tensor_or_any])


# Beta
def parse_mlx_indexing(index: Any) -> Any:
    if isinstance(index, np.ndarray):
        raise TypeError(
            "MLX does not support NumPy arrays as indices. Convert to MLX arrays."
        )

    if isinstance(index, tuple):
        parsed = []
        for id in index:
            if isinstance(id, bool):
                parsed.append(1 if id else 0)  # Ensure boolean converts to int
            elif isinstance(id, mx.array) and id.dtype == bool:
                parsed.append(
                    mx.array(np.where(id.tolist())[0], dtype=mx.int32)
                )  # FIXED: Full NumPy preprocessing
            elif isinstance(id, list) and all(isinstance(i, bool) for i in id):
                mask = mx.array(id, dtype=mx.bool_)
                parsed.append(
                    mx.array(np.where(mask.tolist())[0], dtype=mx.int32)
                )  # FIXED: Fully integer-based indexing
            elif isinstance(id, list):
                parsed.append(mx.array(id, dtype=mx.int32))
            else:
                parsed.append(id)
        return tuple(parsed)

    elif isinstance(index, bool):
        return 1 if index else 0  # Ensure boolean converts to int

    elif isinstance(index, mx.array) and index.dtype == bool:
        return mx.array(
            np.where(index.tolist())[0], dtype=mx.int32
        )  # FIXED: Ensures boolean indices are precomputed

    elif isinstance(index, list) and all(isinstance(i, bool) for i in index):
        mask = mx.array(index, dtype=mx.bool_)
        return mx.array(
            np.where(mask.tolist())[0], dtype=mx.int32
        )  # FIXED: Fully integer-based indexing

    elif isinstance(index, list):
        return mx.array(index, dtype=mx.int32)

    return index
