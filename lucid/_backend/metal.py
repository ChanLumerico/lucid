from typing import Any, ClassVar, Mapping
import warnings
import platform

import numpy as np

from lucid.types import _TensorLike

try:
    import mlx.core as mx
except ImportError:
    class _MetalFallback:
        @staticmethod
        def is_available() -> bool:
            return False

    class _RandomFallback:
        @staticmethod
        def seed(_seed):
            return None

    class _MXFallback:
        array = np.ndarray
        bool_ = np.bool_
        int32 = np.int32
        metal = _MetalFallback()
        random = _RandomFallback()

        @staticmethod
        def eval(x):
            return x

        @staticmethod
        def stop_gradient(x):
            return x

    mx = _MXFallback()


class MetalNotSupportedWarning(UserWarning):
    _has_warned: ClassVar[bool] = False

    def __init__(self, message=None):
        system = platform.system()
        machine = platform.machine()
        arch = platform.processor() or machine

        default_message = (
            f"Metal GPU acceleration is not supported on this system "
            f"({system}, {arch}). Falling back to CPU, which may be slower "
            f"than native CPU due to lazy execution overhead."
        )
        super().__init__(message or default_message)


def check_metal_availability() -> None:
    if MetalNotSupportedWarning._has_warned:
        return
    if not mx.metal.is_available():
        MetalNotSupportedWarning._has_warned = True
        warnings.warn(MetalNotSupportedWarning(), stacklevel=2)


def is_cpu_op(*tensor_or_any) -> bool:
    for t in tensor_or_any:
        device = getattr(t, "device", None)
        if device is None:
            if isinstance(t, mx.array):
                return False
        else:
            if device == "gpu":
                return False
    return True


def is_gpu_op(*tensor_or_any) -> bool:
    for t in tensor_or_any:
        device = getattr(t, "device", None)
        if device is None:
            if isinstance(t, mx.array):
                return True
        else:
            if device == "gpu":
                return True
    return False


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


def post_step_eval(
    param: _TensorLike | Any, state: Mapping[str, Any] | None = None
) -> None:
    is_gpu = False
    if hasattr(param, "is_gpu"):
        try:
            is_gpu = bool(param.is_gpu())
        except Exception:
            is_gpu = False

    if not is_gpu:
        return

    data = getattr(param, "data", None)
    if data is not None:
        mx.eval(data)
        stopped = mx.stop_gradient(data)
        if stopped is not None:
            param.data = stopped

    grad = getattr(param, "grad", None)
    if grad is not None:
        mx.eval(grad)
        stopped = mx.stop_gradient(grad)
        if stopped is not None:
            param.grad = stopped

    if not state:
        return

    for key, value in state.items():
        if isinstance(value, mx.array):
            mx.eval(value)
            stopped = mx.stop_gradient(value)
            if stopped is not None:
                state[key] = stopped
