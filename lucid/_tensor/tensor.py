from typing import TYPE_CHECKING, Self, Any, Iterator

import numpy as np

from lucid._C import engine as _C_engine
from lucid._dtype import dtype, _ENGINE_TO_DTYPE, float16, float32, float64, bfloat16, complex64
from lucid._device import device, _device_from_engine

if TYPE_CHECKING:
    from lucid.nn.module import Module
    from lucid.nn.parameter import Parameter


class Tensor:
    """
    Core data structure of the Lucid framework.

    Wraps TensorImpl via composition. All public API goes through this class.
    """

    _is_parameter: bool = False
    __lucid_function__: None = None

    def __init__(
        self,
        data: Any,
        *,
        dtype: dtype | _C_engine.Dtype | str | None = None,
        device: device | str | None = None,
        requires_grad: bool = False,
    ) -> None:
        from lucid._factories.converters import _to_impl
        self._impl: _C_engine.TensorImpl = _to_impl(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )

    @classmethod
    def __new_from_impl__(cls, impl: _C_engine.TensorImpl) -> Self:
        """Internal factory: wrap an existing TensorImpl without copying."""
        obj = object.__new__(cls)
        obj._impl = impl
        return obj

    @property
    def impl(self) -> _C_engine.TensorImpl:
        """C++ TensorImpl accessor (used by the C++ autograd engine)."""
        return self._impl

    # ── metadata ─────────────────────────────────────────────────────────────

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._impl.shape)

    @property
    def dtype(self) -> dtype:
        return _ENGINE_TO_DTYPE[self._impl.dtype]

    @property
    def device(self) -> device:
        return _device_from_engine(self._impl.device)

    @property
    def ndim(self) -> int:
        return len(self._impl.shape)

    @property
    def T(self) -> Self:
        """Return a tensor with all dimensions reversed (like numpy .T)."""
        return Tensor.__new_from_impl__(_C_engine.T(self._impl))  # type: ignore[return-value]

    @property
    def mT(self) -> Self:
        """Return a tensor with the last two dimensions transposed."""
        return Tensor.__new_from_impl__(_C_engine.mT(self._impl))  # type: ignore[return-value]

    @property
    def is_metal(self) -> bool:
        return self._impl.device == _C_engine.Device.GPU

    @property
    def is_leaf(self) -> bool:
        return self._impl.is_leaf

    @property
    def requires_grad(self) -> bool:
        return self._impl.requires_grad

    @requires_grad.setter
    def requires_grad(self, v: bool) -> None:
        from lucid._dispatch import _impl_with_grad
        self._impl = _impl_with_grad(self._impl, v)

    def numel(self) -> int:
        """Return the total number of elements."""
        return int(self._impl.numel())

    def dim(self) -> int:
        """Return the number of dimensions."""
        return len(self._impl.shape)

    def size(self, dim: int | None = None) -> int | tuple[int, ...]:
        """Return size of a specific dimension, or all dimensions as a tuple."""
        s = tuple(self._impl.shape)
        if dim is not None:
            return s[dim]
        return s

    def is_contiguous(self) -> bool:
        """Return True if the tensor is stored contiguously in memory."""
        return self._impl.is_contiguous()

    # ── autograd ─────────────────────────────────────────────────────────────

    @property
    def grad(self) -> Self | None:
        g = self._impl.grad_as_python()
        if g is None:
            return None
        import numpy as np
        arr = np.asarray(g)
        impl = _C_engine.TensorImpl(arr, self._impl.device, False)
        return Tensor.__new_from_impl__(impl)  # type: ignore[return-value]

    @grad.setter
    def grad(self, v: Tensor | None) -> None:
        if v is None:
            self._impl.zero_grad()
        # TensorImpl has no set_grad — assignment sets the underlying grad buffer
        # through engine_backward only; manual grad setting is not supported

    @property
    def grad_fn(self) -> _C_engine.Node | None:
        return getattr(self._impl, "grad_fn", None)

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Set requires_grad in-place and return self."""
        from lucid._dispatch import _impl_with_grad
        self._impl = _impl_with_grad(self._impl, requires_grad)
        return self

    def retain_grad(self) -> None:
        """Retain gradient on this non-leaf tensor after backward."""
        if hasattr(self._impl, "retain_grad"):
            self._impl.retain_grad()

    def backward(
        self,
        gradient: Tensor | None = None,
        retain_graph: bool = False,
        create_graph: bool = False,
    ) -> None:
        """Compute gradients by backpropagating from this tensor.

        For scalar outputs, gradient defaults to ones.
        For non-scalar outputs, gradient must be provided with the same shape;
        gradients are then computed as if the effective scalar were (self * gradient).sum().
        """
        if gradient is not None:
            if self._impl.shape != gradient._impl.shape:
                raise RuntimeError(
                    f"backward(): gradient shape {tuple(gradient._impl.shape)} does not "
                    f"match tensor shape {tuple(self._impl.shape)}"
                )
            # Compute VJP: (self * g.detach()).sum() then backprop that scalar
            g_impl = gradient.detach()._impl
            scaled = _C_engine.mul(self._impl, g_impl)
            root = _C_engine.sum(scaled)
            _C_engine.engine_backward(root, retain_graph=retain_graph)
        else:
            if self._impl.shape and self._impl.numel() != 1:
                raise RuntimeError(
                    "grad can be implicitly created only for scalar outputs; "
                    "call backward(gradient=...) for non-scalar tensors"
                )
            _C_engine.engine_backward(self._impl, retain_graph=retain_graph)

    def detach(self) -> Self:
        """Return a new Tensor detached from the autograd graph."""
        from lucid._dispatch import _wrap, _impl_with_grad
        import numpy as np
        arr = np.ascontiguousarray(np.asarray(self._impl.data_as_python()))
        impl = _C_engine.TensorImpl(arr, self._impl.device, False)
        return _wrap(impl)  # type: ignore[return-value]

    def detach_(self) -> Self:
        """Detach in-place from the autograd graph."""
        from lucid._dispatch import _impl_with_grad
        self._impl = _impl_with_grad(self._impl, False)
        return self

    def clone(self) -> Self:
        """Return a copy of this tensor, preserving autograd history."""
        from lucid._dispatch import _wrap
        impl = _C_engine.contiguous(self._impl)
        return _wrap(impl)  # type: ignore[return-value]

    # ── conversion ───────────────────────────────────────────────────────────

    def item(self) -> float | int | bool:
        """Return the value of a single-element tensor as a Python scalar."""
        if self._impl.numel() != 1:
            raise RuntimeError(
                "item() can only be called on a tensor with one element"
            )
        arr = self._impl.data_as_python()
        import numpy as np
        val = np.asarray(arr).flat[0]
        return val.item()

    def numpy(self) -> np.ndarray:  # type: ignore[type-arg]
        """Return the tensor as a NumPy array (CPU only)."""
        import numpy as np
        raw = self._impl.data_as_python()
        return np.asarray(raw)

    def tolist(self) -> Any:
        """Return the tensor as a nested Python list."""
        return self.numpy().tolist()

    def contiguous(self) -> Self:
        """Return a contiguous copy of this tensor."""
        from lucid._dispatch import _wrap
        return _wrap(_C_engine.contiguous(self._impl))  # type: ignore[return-value]

    @property
    def data(self) -> Self:
        """Return this tensor's data without gradient tracking."""
        from lucid._dispatch import _impl_with_grad
        return Tensor.__new_from_impl__(_impl_with_grad(self._impl, False))  # type: ignore[return-value]

    # ── device/dtype conversion ───────────────────────────────────────────────
    # Injected by _tensor/_to.py after class definition.

    # ── shape helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        if not self._impl.shape:
            raise TypeError("len() of a 0-d tensor")
        return self._impl.shape[0]

    def __bool__(self) -> bool:
        if self._impl.numel() != 1:
            raise RuntimeError(
                "Boolean value of Tensor with more than one element is ambiguous"
            )
        return bool(self.item())

    def __repr__(self) -> str:
        from lucid._tensor._repr import tensor_repr
        return tensor_repr(self)

    # hash: identity-based so tensors can be used as dict keys
    __hash__ = object.__hash__  # type: ignore[assignment]

    # ── iteration & formatting ────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Self]:
        """Iterate over the first dimension (raises for 0-d tensors)."""
        if not self._impl.shape:
            raise TypeError("iteration over a 0-d tensor")
        for i in range(self._impl.shape[0]):
            yield self[i]  # type: ignore[misc]

    def __format__(self, format_spec: str) -> str:
        if self._impl.numel() == 1:
            return format(self.item(), format_spec)
        return repr(self)

    # ── new_* convenience constructors ────────────────────────────────────────

    def new_empty(self, *size: int, dtype: dtype | None = None,
                  device: device | str | None = None,
                  requires_grad: bool = False) -> Self:
        """Return an uninitialized tensor with the given size, inheriting dtype/device."""
        from lucid._factories.creation import empty as _empty
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _empty(*size, dtype=_dtype, device=_device,  # type: ignore[return-value]
                      requires_grad=requires_grad)

    def new_zeros(self, *size: int, dtype: dtype | None = None,
                  device: device | str | None = None,
                  requires_grad: bool = False) -> Self:
        """Return a zeros tensor with the given size, inheriting dtype/device."""
        from lucid._factories.creation import zeros as _zeros
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _zeros(*size, dtype=_dtype, device=_device,  # type: ignore[return-value]
                      requires_grad=requires_grad)

    def new_ones(self, *size: int, dtype: dtype | None = None,
                 device: device | str | None = None,
                 requires_grad: bool = False) -> Self:
        """Return an all-ones tensor with the given size, inheriting dtype/device."""
        from lucid._factories.creation import ones as _ones
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _ones(*size, dtype=_dtype, device=_device,  # type: ignore[return-value]
                     requires_grad=requires_grad)

    def new_full(self, size: tuple[int, ...], fill_value: float,
                 dtype: dtype | None = None,
                 device: device | str | None = None,
                 requires_grad: bool = False) -> Self:
        """Return a tensor filled with fill_value, inheriting dtype/device."""
        from lucid._factories.creation import full as _full
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _full(size, fill_value, dtype=_dtype, device=_device,  # type: ignore[return-value]
                     requires_grad=requires_grad)

    def new_tensor(self, data: Any, dtype: dtype | None = None,
                   device: device | str | None = None,
                   requires_grad: bool = False) -> Self:
        """Return a new tensor from data, inheriting dtype/device."""
        from lucid._factories.converters import tensor as _tensor
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _tensor(data, dtype=_dtype, device=_device,  # type: ignore[return-value]
                       requires_grad=requires_grad)

    # ── size / element info ───────────────────────────────────────────────────

    def element_size(self) -> int:
        """Return the size of each element in bytes."""
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        """Total number of bytes consumed by the tensor data."""
        return self._impl.numel() * self.dtype.itemsize

    def is_floating_point(self) -> bool:
        """Return True if the dtype is a floating-point type."""
        return self.dtype in (float16, float32, float64, bfloat16)

    def is_complex(self) -> bool:
        """Return True if the dtype is complex."""
        return self.dtype is complex64

    def share_memory_(self) -> Self:
        """No-op: Apple Silicon uses unified memory — all tensors share memory."""
        return self

    # ── zero_() helper ───────────────────────────────────────────────────────

    def zero_(self) -> Self:
        """Fill this tensor with zeros in-place."""
        from lucid._dispatch import _unwrap
        result = _C_engine.mul_(self._impl, _C_engine.zeros(
            self._impl.shape, self._impl.dtype, self._impl.device
        ))
        self._impl = result
        return self


# ── inject dunders and methods after class definition ────────────────────────
from lucid._tensor._dunders import _inject_dunders  # noqa: E402
from lucid._tensor._methods import _inject_methods  # noqa: E402
from lucid._tensor._to import _inject_to  # noqa: E402
_inject_dunders(Tensor)
_inject_methods(Tensor)
_inject_to(Tensor)
