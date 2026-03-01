from __future__ import annotations

import sys
import types
from enum import Enum

import numpy as np


if "mlx.core" not in sys.modules:
    mlx_module = types.ModuleType("mlx")
    core_module = types.ModuleType("mlx.core")

    class FakeArray(np.ndarray):
        def __new__(cls, input_array, dtype=None):
            arr = np.array(input_array, dtype=dtype)
            return arr.view(cls)

    class DeviceType(Enum):
        cpu = "cpu"
        gpu = "gpu"

    class _Device:
        type = DeviceType.cpu

    class Dtype:
        def __init__(self, np_dtype):
            self._np_dtype = np.dtype(np_dtype)
            self.size = self._np_dtype.itemsize

    class _Metal:
        @staticmethod
        def is_available() -> bool:
            return False

    def array(x, dtype=None):
        return FakeArray(x, dtype=dtype)

    core_module.array = FakeArray
    core_module.DeviceType = DeviceType
    core_module.Dtype = Dtype
    core_module.default_device = lambda: _Device()

    # Dtypes
    core_module.bool_ = np.bool_
    core_module.int8 = np.int8
    core_module.int16 = np.int16
    core_module.int32 = np.int32
    core_module.int64 = np.int64
    core_module.float16 = np.float16
    core_module.float32 = np.float32
    core_module.float64 = np.float64
    core_module.complex64 = np.complex64

    # Array ops used by code paths
    core_module.add = np.add
    core_module.subtract = np.subtract
    core_module.multiply = np.multiply
    core_module.divide = np.divide
    core_module.minimum = np.minimum
    core_module.maximum = np.maximum
    core_module.power = np.power
    core_module.matmul = np.matmul
    core_module.dot = np.dot
    core_module.inner = np.inner
    core_module.outer = np.outer
    core_module.tensordot = np.tensordot
    core_module.exp = np.exp
    core_module.log = np.log
    core_module.log2 = np.log2
    core_module.sqrt = np.sqrt
    core_module.sin = np.sin
    core_module.cos = np.cos
    core_module.tan = np.tan
    core_module.arcsin = np.arcsin
    core_module.arccos = np.arccos
    core_module.arctan = np.arctan
    core_module.sinh = np.sinh
    core_module.cosh = np.cosh
    core_module.tanh = np.tanh
    core_module.clip = np.clip
    core_module.abs = np.abs
    core_module.sign = np.sign
    core_module.reciprocal = np.reciprocal
    core_module.square = np.square
    core_module.floor = np.floor
    core_module.ceil = np.ceil
    core_module.round = np.round
    core_module.sum = np.sum
    core_module.mean = np.mean
    core_module.var = np.var
    core_module.min = np.min
    core_module.max = np.max
    core_module.swapaxes = np.swapaxes
    core_module.transpose = np.transpose
    core_module.cumprod = np.cumprod
    core_module.cumsum = np.cumsum
    core_module.broadcast_to = np.broadcast_to
    core_module.repeat = np.repeat
    core_module.eval = lambda x: x
    core_module.stop_gradient = lambda x: x
    core_module.metal = _Metal()

    mlx_module.core = core_module
    sys.modules["mlx"] = mlx_module
    sys.modules["mlx.core"] = core_module
