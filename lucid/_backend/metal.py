from typing import Any
import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as e:
    print(f"mlx library not installed. Try 'pip install mlx'.")


def is_cpu_op(*tensor_or_any) -> bool:
    return all([t.device == "cpu" for t in tensor_or_any if hasattr(t, "device")])


def is_gpu_op(*tensor_or_any) -> bool:
    return any([t.device == "gpu" for t in tensor_or_any if hasattr(t, "device")])


def parse_mlx_indexing(index: Any) -> Any:
    if isinstance(index, np.ndarray):
        raise TypeError(
            "GPU tensors do not support CPU tensor or NumPy array indexing. "
            + "Convert to GPU tensors."
        )

    if isinstance(index, tuple):
        parsed = []
        for i, idx in enumerate(index):
            if isinstance(idx, np.ndarray):
                raise ValueError(f"NumPy array indexing found at {i}-th index.")

            if isinstance(idx, bool):
                parsed.append(1 if idx else 0)

            elif isinstance(idx, mx.array) and idx.dtype == mx.bool_:
                parsed.append(mx.array(np.flatnonzero(idx.tolist()), dtype=mx.int32))

            elif isinstance(idx, list) and all(isinstance(i, bool) for i in idx):
                mask = mx.array(idx, dtype=mx.bool_)
                parsed.append(mx.array(np.flatnonzero(mask.tolist()), dtype=mx.int32))

            elif isinstance(idx, list):
                parsed.append(mx.array(idx, dtype=mx.int32))

            else:
                parsed.append(idx)

        return tuple(parsed)

    elif isinstance(index, bool):
        return 1 if index else 0

    elif isinstance(index, mx.array) and index.dtype == mx.bool_:
        return mx.array(np.flatnonzero(index.tolist()), dtype=mx.int32)

    elif isinstance(index, list) and all(isinstance(i, bool) for i in index):
        mask = mx.array(index, dtype=mx.bool_)
        return mx.array(np.flatnonzero(mask.tolist()), dtype=mx.int32)

    elif isinstance(index, list):
        return mx.array(index, dtype=mx.int32)

    return index
