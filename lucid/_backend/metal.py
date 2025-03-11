from typing import Any

try:
    import mlx.core as mx
except ModuleNotFoundError as e:
    print(f"mlx library not installed. Try 'pip install mlx'.")

_MLXArray = mx.array


def is_cpu_op(*tensor_or_any) -> bool:
    return not is_gpu_op(*tensor_or_any)


def is_gpu_op(*tensor_or_any) -> bool:
    return any([getattr(t, "device", None) == "gpu" for t in tensor_or_any])
