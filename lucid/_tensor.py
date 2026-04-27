"""
lucid._tensor — Tensor: thin Python handle around `_C_engine.TensorImpl`.

Every public op (`lucid.add`, `lucid.matmul`, `nn.Linear.forward`, ...)
unwraps the `_impl` field, dispatches to the C++ engine, and re-wraps the
returned `TensorImpl`. Python-side state is therefore minimal: every
tensor's storage, shape, dtype, device, version counter, requires_grad
flag, autograd graph node, and gradient buffer all live in C++.

Subclasses (`FloatTensor`, `IntTensor`, …) are dtype-fixed convenience
constructors mirroring the legacy API.
"""

from __future__ import annotations

from typing import Any, ClassVar, Iterator, Optional, Sequence

import numpy as np

from lucid._C import engine as _C_engine
from lucid._bridge import (
    to_engine_dtype,
    to_engine_device,
    from_engine_dtype,
    from_engine_device,
    impl_of,
)
from lucid.error import UnknownDeviceError
from lucid.types import (
    Numeric,
    _ArrayOrScalar,
    _BuiltinNumeric,
    _DeviceType,
    _ShapeLike,
)


__all__ = [
    "Tensor",
    "FloatTensor", "DoubleTensor", "HalfTensor",
    "CharTensor", "ShortTensor", "IntTensor", "LongTensor",
    "BoolTensor",
]


def _to_numpy(data: _ArrayOrScalar, *, dtype: Numeric | None) -> np.ndarray:
    """Resolve any user-flavored input to a numpy array suitable for upload
    into the C++ engine."""
    if isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, (int, float, complex, bool)):
        arr = np.asarray(data)
    elif isinstance(data, (list, tuple)):
        arr = np.asarray(data)
    else:
        # MLX array, torch tensor, etc. — try numpy() conversion
        if hasattr(data, "numpy"):
            arr = np.asarray(data.numpy())
        else:
            try:
                arr = np.asarray(data)
            except Exception as e:
                raise TypeError(
                    f"Cannot construct Tensor from {type(data).__name__}: {e}"
                )
    if dtype is not None:
        target = dtype.cpu
        if target is not None and arr.dtype != target:
            arr = arr.astype(target)
    return np.ascontiguousarray(arr)


class Tensor:
    """Lucid tensor — Python handle to a C++ engine `TensorImpl`."""

    # Subclasses set this to fix the dtype at construction.
    _fixed_dtype: ClassVar[Numeric | None] = None

    # ----- construction ----------------------------------------------------
    def __init__(
        self,
        data: _ArrayOrScalar | "Tensor" | "_C_engine.TensorImpl",
        requires_grad: bool = False,
        keep_grad: bool = False,
        dtype: _BuiltinNumeric | Numeric | None = None,
        device: _DeviceType = "cpu",
    ) -> None:
        # Subclass dtype takes precedence
        if self._fixed_dtype is not None:
            dtype = self._fixed_dtype

        # Coerce builtin types (int/float/complex/bool) to Numeric
        if isinstance(dtype, type) and dtype in (int, float, complex, bool):
            from lucid.types import Int64, Float32, Complex64, Bool
            dtype = {int: Int64, float: Float32, complex: Complex64,
                     bool: Bool}[dtype]
        if dtype is not None and not isinstance(dtype, Numeric):
            raise TypeError(f"dtype must be Numeric or builtin type, got {dtype!r}")

        # Already a TensorImpl: wrap directly
        if isinstance(data, _C_engine.TensorImpl):
            self._impl = data
            self._keep_grad = keep_grad
            return

        # Already a Tensor: share or copy
        if isinstance(data, Tensor):
            self._impl = data._impl
            self._keep_grad = keep_grad
            if requires_grad and not self._impl.requires_grad:
                # Cannot retroactively flip requires_grad on the same impl.
                # Fall through to a fresh copy with the new flag.
                arr = np.array(self._impl.data_as_python())
                self._impl = _C_engine.TensorImpl(
                    arr, to_engine_device(device), True
                )
            return

        # Convert input to numpy
        arr = _to_numpy(data, dtype=dtype)

        eng_device = to_engine_device(device)
        self._impl = _C_engine.TensorImpl(arr, eng_device, requires_grad)
        self._keep_grad = keep_grad

    @classmethod
    def _wrap(cls, impl: "_C_engine.TensorImpl") -> "Tensor":
        """Fast-path constructor: wrap a TensorImpl without re-validating."""
        t = cls.__new__(cls)
        t._impl = impl
        t._keep_grad = False
        return t

    @staticmethod
    def copy_data(data: Any) -> np.ndarray:
        """Return a numpy copy of the underlying data — utility used by some
        legacy callers (e.g., to clone a tensor's data into a new device)."""
        if isinstance(data, np.ndarray):
            return data.copy()
        if hasattr(data, "data_as_python"):
            return np.array(data.data_as_python()).copy()
        return np.asarray(data).copy()

    # ----- metadata --------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._impl.shape)

    @property
    def ndim(self) -> int:
        return len(self._impl.shape)

    @property
    def size(self) -> int:
        return self._impl.numel()

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of 0-d tensor")
        return self.shape[0]

    @property
    def dtype(self) -> Numeric:
        return from_engine_dtype(self._impl.dtype)

    @property
    def device(self) -> _DeviceType:
        return from_engine_device(self._impl.device)

    @property
    def requires_grad(self) -> bool:
        return self._impl.requires_grad

    def requires_grad_(self, value: bool = True) -> "Tensor":
        # Cannot toggle in-place on the same impl; re-allocate if needed.
        if self._impl.requires_grad == value:
            return self
        arr = np.array(self._impl.data_as_python())
        self._impl = _C_engine.TensorImpl(arr, self._impl.device, value)
        return self

    @property
    def is_leaf(self) -> bool:
        return self._impl.is_leaf

    @property
    def version(self) -> int:
        return self._impl.version

    @property
    def keep_grad(self) -> bool:
        return getattr(self, "_keep_grad", False)

    @property
    def is_free(self) -> bool:
        return False  # legacy concept; new engine manages lifetime via shared_ptr

    # ----- data access -----------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        """Zero-copy numpy view of the tensor's storage (CPU only).
        For GPU tensors this triggers a download to host."""
        return np.array(self._impl.data_as_python())

    @property
    def grad(self) -> Optional[np.ndarray]:
        g = self._impl.grad_as_python()
        if g is None:
            return None
        return np.array(g)

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> Any:
        if self.size != 1:
            raise ValueError("only scalar tensors support .item()")
        return self.data.flatten()[0].item()

    def tolist(self) -> Any:
        return self.data.tolist()

    # ----- device movement -------------------------------------------------
    def to(self, device: _DeviceType | str) -> "Tensor":
        target = to_engine_device(device)
        if self._impl.device == target:
            return self
        # Round-trip via numpy (cheap on Apple unified memory).
        arr = np.array(self._impl.data_as_python())
        new_impl = _C_engine.TensorImpl(arr, target, self._impl.requires_grad)
        return Tensor._wrap(new_impl)

    def cpu(self) -> "Tensor":
        return self.to("cpu")

    def gpu(self) -> "Tensor":
        return self.to("gpu")

    def is_cpu(self) -> bool:
        return self.device == "cpu"

    def is_gpu(self) -> bool:
        return self.device == "gpu"

    # ----- dtype --------------------------------------------------------
    def astype(self, dtype: _BuiltinNumeric | Numeric) -> "Tensor":
        if isinstance(dtype, type) and dtype in (int, float, complex, bool):
            from lucid.types import Int64, Float32, Complex64, Bool
            dtype = {int: Int64, float: Float32, complex: Complex64,
                     bool: Bool}[dtype]
        target_eng = to_engine_dtype(dtype)
        if self._impl.dtype == target_eng:
            return self
        # Re-cast via numpy (cheap; engine cast op TBD).
        arr = np.array(self._impl.data_as_python())
        np_target = dtype.cpu if isinstance(dtype, Numeric) and dtype.cpu else None
        if np_target is not None:
            arr = arr.astype(np_target)
        new_impl = _C_engine.TensorImpl(arr, self._impl.device,
                                        self._impl.requires_grad)
        return Tensor._wrap(new_impl)

    # ----- autograd -------------------------------------------------------
    def backward(
        self, gradient: Optional["Tensor"] = None, *,
        retain_graph: bool = False, retain_grad: bool = False,
    ) -> None:
        if not self._impl.requires_grad:
            raise RuntimeError("backward() called on tensor without requires_grad")
        # The engine's implicit-ones-seed path is taken when gradient is None.
        # (gradient-arg path TBD — currently engine seed is always implicit.)
        _C_engine.engine_backward(self._impl, retain_graph)

    def zero_grad(self) -> None:
        self._impl.zero_grad()

    def detach(self) -> "Tensor":
        arr = np.array(self._impl.data_as_python())
        new_impl = _C_engine.TensorImpl(arr, self._impl.device, False)
        return Tensor._wrap(new_impl)

    def free(self) -> None:
        # Legacy hook; engine releases via shared_ptr automatically.
        pass

    def new_tensor(self) -> "Tensor":
        return self.detach()

    def clear_node(self, clear_op: bool = True) -> None:
        # Legacy autograd graph hook; new engine has no equivalent surface.
        pass

    # ----- common shape methods (defined fully in lucid.ops, here as forwarders)
    def reshape(self, *shape) -> "Tensor":
        from lucid.ops.utils import reshape
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            return reshape(self, shape[0])
        return reshape(self, list(shape))

    def view(self, *shape) -> "Tensor":
        return self.reshape(*shape)

    def squeeze(self, axis=None) -> "Tensor":
        from lucid.ops.utils import squeeze
        return squeeze(self, axis)

    def unsqueeze(self, axis: int) -> "Tensor":
        from lucid.ops.utils import unsqueeze
        return unsqueeze(self, axis)

    def flatten(self, start_axis: int = 0, end_axis: int = -1) -> "Tensor":
        from lucid.ops.utils import flatten
        return flatten(self, start_axis, end_axis)

    def ravel(self) -> "Tensor":
        from lucid.ops.utils import ravel
        return ravel(self)

    def expand(self, *sizes) -> "Tensor":
        from lucid.ops.utils import expand
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
            sizes = sizes[0]
        return expand(self, *sizes)

    def transpose(self, axes: list[int] | None = None) -> "Tensor":
        from lucid.ops import transpose
        return transpose(self, axes)

    def swapaxes(self, axis1: int, axis2: int) -> "Tensor":
        from lucid.ops import swapaxes
        return swapaxes(self, axis1, axis2)

    @property
    def T(self) -> "Tensor":
        from lucid.ops import T as T_op
        return T_op(self)

    @property
    def mT(self) -> "Tensor":
        from lucid.ops import mT as mT_op
        return mT_op(self)

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        from lucid.ops import sum as _sum
        return _sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        from lucid.ops import mean as _mean
        return _mean(self, axis, keepdims)

    def var(self, axis=None, keepdims: bool = False) -> "Tensor":
        from lucid.ops import var as _var
        return _var(self, axis, keepdims)

    def matmul(self, other: "Tensor") -> "Tensor":
        from lucid.ops import matmul
        return matmul(self, other)

    def dot(self, other: "Tensor") -> "Tensor":
        from lucid.ops import dot
        return dot(self, other)

    def clip(self, min_value=None, max_value=None) -> "Tensor":
        from lucid.ops import clip
        return clip(self, min_value, max_value)

    # ----- repr -----------------------------------------------------------
    def __repr__(self) -> str:
        try:
            arr = np.array(self._impl.data_as_python())
            preview = np.array2string(arr, precision=4, suppress_small=True,
                                       threshold=20, edgeitems=2)
        except Exception:
            preview = "<unprintable>"
        flag = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({preview}, dtype={self.dtype}, device='{self.device}'{flag})"

    __str__ = __repr__


# --------------------------------------------------------------------------- #
# Dtype-fixed subclasses (mirror legacy lucid.FloatTensor / IntTensor / ...)
# --------------------------------------------------------------------------- #

from lucid.types import (
    Float16 as _F16, Float32 as _F32, Float64 as _F64,
    Int8 as _I8, Int16 as _I16, Int32 as _I32, Int64 as _I64,
    Bool as _Bool,
)


class FloatTensor(Tensor):  _fixed_dtype = _F32
class DoubleTensor(Tensor): _fixed_dtype = _F64
class HalfTensor(Tensor):   _fixed_dtype = _F16
class CharTensor(Tensor):   _fixed_dtype = _I8
class ShortTensor(Tensor):  _fixed_dtype = _I16
class IntTensor(Tensor):    _fixed_dtype = _I32
class LongTensor(Tensor):   _fixed_dtype = _I64
class BoolTensor(Tensor):   _fixed_dtype = _Bool
