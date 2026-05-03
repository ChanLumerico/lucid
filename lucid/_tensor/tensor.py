from typing import TYPE_CHECKING, Self, Any, Iterator

import numpy as np

from lucid._C import engine as _C_engine
from lucid._dtype import dtype, _ENGINE_TO_DTYPE, float16, float32, float64, bfloat16, complex64
from lucid._device import device, _device_from_engine

if TYPE_CHECKING:
    from lucid.nn.module import Module
    from lucid.nn.parameter import Parameter


class Tensor:
    """Multi-dimensional array with automatic differentiation support.

    The central data structure of the Lucid framework. Wraps a C++ ``TensorImpl``
    via composition. Tensors live on either ``cpu`` (Apple Accelerate) or
    ``metal`` (Apple Metal GPU) devices.

    Parameters
    ----------
    data : array_like
        Input data. Accepts nested Python lists, NumPy arrays, or scalars.
    dtype : lucid.dtype, optional
        Desired data type. Defaults to ``lucid.float32``.
    device : str or lucid.device, optional
        Target device (``"cpu"`` or ``"metal"``). Defaults to the global default.
    requires_grad : bool, optional
        If ``True``, operations on this tensor are recorded for autograd.
        Default is ``False``.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.Tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> x.shape
    (2, 2)
    >>> x.dtype
    lucid.float32
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
        """Compute gradients by back-propagating from this tensor.

        Accumulates gradients into the ``.grad`` attribute of all leaf tensors
        that have ``requires_grad=True``.

        Parameters
        ----------
        gradient : Tensor, optional
            Seed gradient of the same shape as ``self``. Required when ``self``
            is not a scalar. Omitting it for non-scalars raises ``RuntimeError``.
        retain_graph : bool, optional
            If ``False`` (default), the computation graph is freed after the
            backward pass. Set to ``True`` when calling backward multiple times.
        create_graph : bool, optional
            Not yet supported. Reserved for higher-order gradients.

        Raises
        ------
        RuntimeError
            If ``self`` is not a scalar and ``gradient`` is not provided, or if
            ``gradient.shape != self.shape``.

        Examples
        --------
        >>> x = lucid.randn(3)
        >>> x.requires_grad_(True)
        >>> y = (x * x).sum()
        >>> y.backward()
        >>> x.grad          # 2 * x
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

    # ── Phase N convenience methods ───────────────────────────────────────────

    def fill_(self, value: float) -> Self:
        """Fill this tensor in-place with a scalar value."""
        raw = self._impl.data_as_python()
        raw[:] = value
        return self

    def copy_(self, other: Self) -> Self:
        """Copy data from other into this tensor in-place."""
        raw = self._impl.data_as_python()
        import numpy as np
        raw[:] = np.asarray(other._impl.data_as_python())
        return self

    def flip(self, dims: int | list[int]) -> Self:
        """Reverse the tensor along the given dimension(s)."""
        import numpy as np
        dims_list = [dims] if isinstance(dims, int) else list(dims)
        arr = np.ascontiguousarray(np.flip(np.asarray(self._impl.data_as_python()), axis=dims_list))
        impl = _C_engine.TensorImpl(arr, self._impl.device, False)
        return Tensor.__new_from_impl__(impl)  # type: ignore[return-value]

    def fliplr(self) -> Self:
        """Reverse the tensor along dimension 1 (left-right)."""
        return self.flip(1)

    def flipud(self) -> Self:
        """Reverse the tensor along dimension 0 (up-down)."""
        return self.flip(0)

    def index_select(self, dim: int, index: Self) -> Self:
        """Select elements along dim using integer index tensor."""
        # Build broadcast index for gather
        import numpy as np
        out_shape = list(self._impl.shape)
        out_shape[dim] = index._impl.shape[0]
        idx_1d = np.asarray(index._impl.data_as_python()).flatten().astype(np.int32)
        bcast_shape = [1] * len(out_shape)
        bcast_shape[dim] = len(idx_1d)
        idx_nd = np.broadcast_to(idx_1d.reshape(bcast_shape), out_shape).copy()
        idx_impl = _C_engine.TensorImpl(np.ascontiguousarray(idx_nd), self._impl.device, False)
        return Tensor.__new_from_impl__(_C_engine.gather(self._impl, idx_impl, dim))  # type: ignore[return-value]

    def masked_select(self, mask: Self) -> Self:
        """Return a 1-D tensor of elements where mask is True."""
        import numpy as np
        arr = np.asarray(self._impl.data_as_python())
        m = np.asarray(mask._impl.data_as_python()).astype(bool)
        selected = np.ascontiguousarray(arr[m].astype(arr.dtype))
        impl = _C_engine.TensorImpl(selected, self._impl.device, False)
        return Tensor.__new_from_impl__(impl)  # type: ignore[return-value]

    def expand_as(self, other: Self) -> Self:
        """Expand this tensor to the shape of other."""
        return Tensor.__new_from_impl__(  # type: ignore[return-value]
            _C_engine.broadcast_to(self._impl, list(other._impl.shape))
        )

    def view_as(self, other: Self) -> Self:
        """Return a tensor with the same data but reshaped to other.shape."""
        return Tensor.__new_from_impl__(  # type: ignore[return-value]
            _C_engine.reshape(self._impl, list(other._impl.shape))
        )

    def type_as(self, other: Self) -> Self:
        """Return a tensor cast to the same dtype as other."""
        return self.to(other.dtype)

    def lerp(self, end: Self, weight: float | Self) -> Self:
        """Linear interpolation: self + weight * (end - self)."""
        diff = Tensor.__new_from_impl__(  # type: ignore[return-value]
            _C_engine.sub(end._impl, self._impl)
        )
        if isinstance(weight, Tensor):
            scaled = Tensor.__new_from_impl__(  # type: ignore[return-value]
                _C_engine.mul(weight._impl, diff._impl)
            )
        else:
            import numpy as np
            w_arr = np.full(diff._impl.shape, weight, dtype=np.float32)
            w_impl = _C_engine.TensorImpl(w_arr, self._impl.device, False)
            scaled = Tensor.__new_from_impl__(  # type: ignore[return-value]
                _C_engine.mul(w_impl, diff._impl)
            )
        return Tensor.__new_from_impl__(_C_engine.add(self._impl, scaled._impl))  # type: ignore[return-value]

    def where(self, condition: Self, other: Self | float) -> Self:
        """Return elements from self where condition is True, else from other."""
        if not isinstance(other, Tensor):
            import numpy as np
            arr = np.full(self._impl.shape, float(other), dtype=np.float32)
            other_impl = _C_engine.TensorImpl(arr, self._impl.device, False)
        else:
            other_impl = other._impl
        return Tensor.__new_from_impl__(  # type: ignore[return-value]
            _C_engine.where(condition._impl, self._impl, other_impl)
        )

    def diff(self, n: int = 1, dim: int = -1) -> Self:
        """Compute n-th order discrete difference along dim."""
        result: Self = self  # type: ignore[assignment]
        for _ in range(n):
            ndim = len(result._impl.shape)
            d = dim % ndim
            length = result._impl.shape[d]
            # result[1:] - result[:-1] along dim
            from lucid._tensor._indexing import _select_slice as _ss
            a_impl = _ss(result._impl, d, slice(1, length, 1))
            b_impl = _ss(result._impl, d, slice(0, length - 1, 1))
            result = Tensor.__new_from_impl__(_C_engine.sub(a_impl, b_impl))  # type: ignore[assignment]
        return result

    def addmm(self, mat1: Self, mat2: Self,
              beta: float = 1.0, alpha: float = 1.0) -> Self:
        """beta * self + alpha * mat1 @ mat2."""
        mm = Tensor.__new_from_impl__(_C_engine.matmul(mat1._impl, mat2._impl))  # type: ignore[return-value]
        import numpy as np
        if alpha != 1.0:
            a_arr = np.full(mm._impl.shape, alpha, dtype=np.float32)
            a_impl = _C_engine.TensorImpl(a_arr, mm._impl.device, False)
            mm_impl = _C_engine.mul(a_impl, mm._impl)
        else:
            mm_impl = mm._impl
        if beta != 1.0:
            b_arr = np.full(self._impl.shape, beta, dtype=np.float32)
            b_impl = _C_engine.TensorImpl(b_arr, self._impl.device, False)
            self_scaled = _C_engine.mul(b_impl, self._impl)
        else:
            self_scaled = self._impl
        return Tensor.__new_from_impl__(_C_engine.add(self_scaled, mm_impl))  # type: ignore[return-value]

    def bmm(self, mat2: Self) -> Self:
        """Batch matrix multiplication: (B,n,m) @ (B,m,p) → (B,n,p)."""
        return Tensor.__new_from_impl__(_C_engine.matmul(self._impl, mat2._impl))  # type: ignore[return-value]

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
