from typing import TYPE_CHECKING, Callable, ClassVar, Final, Self, Iterator, overload

if TYPE_CHECKING:
    import numpy as np

from lucid._C import engine as _C_engine
from lucid._dtype import (
    dtype,
    _ENGINE_TO_DTYPE,
    bool_,
    int8,
    int16,
    int32,
    int64,
    float16,
    float32,
    float64,
    bfloat16,
    complex64,
)
from lucid._device import device, _device_from_engine
from lucid._dispatch import _wrap, _impl_with_grad
from lucid._factories.creation import (
    zeros as _zeros,
    ones as _ones,
    empty as _empty,
    full as _full,
)
from lucid._factories.converters import tensor as _tensor_fn, _to_impl

if TYPE_CHECKING:
    from lucid.nn.module import Module
    from lucid.nn.parameter import Parameter
    from lucid.autograd._hooks import RemovableHandle as _RemovableHandle


class Tensor[DT: dtype, DV: device]:
    """Multi-dimensional array with automatic differentiation support.

    The central data structure of the Lucid framework. Wraps a C++ ``TensorImpl``
    via composition. Tensors live on either ``cpu`` (Apple Accelerate) or
    ``metal`` (Apple Metal GPU) devices.

    Type Parameters
    ---------------
    DT : dtype
        The element dtype of this tensor, e.g. ``lucid.float32``.
        Use ``Tensor[float32, device]`` in type hints to communicate dtype.
    DV : device
        The device this tensor resides on, e.g. ``lucid.device("cpu")``.
        Use ``Tensor[dtype, device("metal")]`` to communicate device.

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

    _is_parameter: ClassVar[bool] = False
    __lucid_function__: ClassVar[None] = None

    def __init__(
        self,
        data: np.ndarray | list[object] | int | float | bool | Tensor,
        *,
        dtype: dtype | _C_engine.Dtype | str | None = None,
        device: device | str | None = None,
        requires_grad: bool = False,
    ) -> None:
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
    def dtype(self) -> DT:
        return _ENGINE_TO_DTYPE[self._impl.dtype]  # type: ignore[return-value]

    @property
    def device(self) -> DV:
        return _device_from_engine(self._impl.device)  # type: ignore[return-value]

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
    def is_shared(self) -> bool:
        """True when backed by a Metal MTLResourceStorageModeShared buffer.

        Shared-memory tensors live in Apple Silicon unified DRAM and are
        simultaneously accessible from CPU and GPU without a memcpy.
        Create them with ``lucid.metal.shared_tensor()`` or promote an
        existing tensor with ``lucid.metal.to_shared()``.
        """
        return self._impl.is_metal_shared

    @property
    def is_leaf(self) -> bool:
        return self._impl.is_leaf

    @property
    def requires_grad(self) -> bool:
        return self._impl.requires_grad

    @requires_grad.setter
    def requires_grad(self, v: bool) -> None:
        self._impl = _impl_with_grad(self._impl, v)

    def numel(self) -> int:
        """Return the total number of elements."""
        return int(self._impl.numel())

    def dim(self) -> int:
        """Return the number of dimensions."""
        return len(self._impl.shape)

    @overload
    def size(self, dim: int) -> int: ...
    @overload
    def size(self, dim: None = ...) -> tuple[int, ...]: ...
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
        # Prefer graph-mode gradient (set when backward was run with create_graph=True).
        g_impl = self._impl.grad_as_impl()
        if g_impl is not None:
            return Tensor.__new_from_impl__(g_impl)  # type: ignore[return-value]
        # Normal backward pass gradient: wrap the grad Storage as a TensorImpl
        # via the engine's ``grad_to_tensor`` helper — no numpy round-trip.
        g_impl = self._impl.grad_to_tensor()
        if g_impl is None:
            return None
        return Tensor.__new_from_impl__(g_impl)  # type: ignore[return-value]

    @grad.setter
    def grad(self, v: Tensor | None) -> None:
        if v is None:
            self._impl.zero_grad()
        else:
            from lucid._dispatch import _unwrap

            self._impl.set_grad(_unwrap(v))

    @property
    def grad_fn(self) -> _C_engine.Node | None:
        return getattr(self._impl, "grad_fn", None)

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Set requires_grad in-place and return self."""
        self._impl = _impl_with_grad(self._impl, requires_grad)
        return self

    def retain_grad(self) -> None:
        """Retain gradient on this non-leaf tensor after backward."""
        if hasattr(self._impl, "retain_grad_"):
            self._impl.retain_grad_()

    def register_hook(
        self, hook: Callable[[Tensor], Tensor | None]
    ) -> _RemovableHandle:
        """Register a hook that fires when this tensor's gradient is computed.

        The hook receives the accumulated gradient tensor.  If it returns a
        non-``None`` :class:`~lucid.Tensor`, that value replaces the gradient.

        Parameters
        ----------
        hook : callable
            ``hook(grad: Tensor) -> Tensor | None``

        Returns
        -------
        RemovableHandle
            Call ``.remove()`` to de-register the hook, or use it as a
            context manager.

        Notes
        -----
        * For leaf tensors the hook fires after :meth:`backward` accumulates
          the gradient, which is the common use case (gradient clipping,
          logging).
        * For non-leaf tensors, call :meth:`retain_grad` before the forward
          pass so the gradient is preserved and available when hooks fire.
        * The hook must be registered **before** the forward computation for
          non-leaf tensors; for leaf tensors any timing works.

        Examples
        --------
        >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> grads = []
        >>> h = x.register_hook(lambda g: grads.append(g.clone()))
        >>> (x * 2).sum().backward()
        >>> grads[0]          # tensor([2., 2., 2.])
        >>> h.remove()        # de-register
        """
        if not self.requires_grad:
            raise RuntimeError(
                "register_hook called on a tensor that does not require grad. "
                "Only tensors with requires_grad=True can have gradient hooks."
            )
        from lucid.autograd._hooks import _register_tensor_hook

        return _register_tensor_hook(self, hook)

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
        # On Metal, flush the forward computation graph before backward so
        # that MLX evaluates two small graphs (forward, then backward+step)
        # instead of one large fused graph — roughly 2× faster in practice.
        # Implemented in C++ (TensorImpl::eval) — no Python-level mlx import.
        self._impl.eval()

        if gradient is not None:
            if self._impl.shape != gradient._impl.shape:
                raise RuntimeError(
                    f"backward(): gradient shape {tuple(gradient._impl.shape)} does not "
                    f"match tensor shape {tuple(self._impl.shape)}"
                )
            g_impl = gradient.detach()._impl
            scaled = _C_engine.mul(self._impl, g_impl)
            root = _C_engine.sum(scaled)
            _C_engine.engine_backward(
                root, retain_graph=retain_graph, create_graph=create_graph
            )
        else:
            if self._impl.shape and self._impl.numel() != 1:
                raise RuntimeError(
                    "grad can be implicitly created only for scalar outputs; "
                    "call backward(gradient=...) for non-scalar tensors"
                )
            _C_engine.engine_backward(
                self._impl, retain_graph=retain_graph, create_graph=create_graph
            )

        # Fire any tensor-level gradient hooks registered via register_hook().
        # Import lazily to avoid circular imports at module load time.
        from lucid.autograd._hooks import _dispatch_tensor_grad_hooks

        _dispatch_tensor_grad_hooks()

    def detach(self) -> Self:
        """Return a new Tensor detached from the autograd graph."""
        return Tensor.__new_from_impl__(
            _impl_with_grad(  # type: ignore[return-value]
                _C_engine.contiguous(self._impl), False
            )
        )

    def detach_(self) -> Self:
        """Detach in-place from the autograd graph."""
        self._impl = _impl_with_grad(self._impl, False)
        return self

    def clamp_(
        self,
        min: float | None = None,
        max: float | None = None,
    ) -> Self:
        """Clamp all elements in-place to [min, max]."""
        lo = min if min is not None else float("-inf")
        hi = max if max is not None else float("inf")
        self._impl = _C_engine.clip_(self._impl, lo, hi)
        return self

    def clamp_min_(self, min: float) -> Self:
        """Clamp all elements in-place to a minimum value."""
        return self.clamp_(min=min)

    def clamp_max_(self, max: float) -> Self:
        """Clamp all elements in-place to a maximum value."""
        return self.clamp_(max=max)

    def clone(self) -> Self:
        """Return a copy of this tensor, preserving autograd history."""
        impl = _C_engine.contiguous(self._impl)
        return _wrap(impl)  # type: ignore[return-value]

    # ── conversion ───────────────────────────────────────────────────────────

    def item(self) -> float | int | bool:
        """Return the value of a single-element tensor as a Python scalar.

        Delegates to the engine's ``TensorImpl::item`` which performs the
        single-element extraction (including IEEE-754 binary16 → float
        decoding) without going through numpy.
        """
        return self._impl.item()

    def numpy(self) -> np.ndarray:  # type: ignore[type-arg]
        """Return the tensor as a NumPy array (CPU only).

        Imports numpy lazily — the rest of Lucid stays numpy-free unless
        the user explicitly bridges through this method.  When numpy is
        not installed, raises an ImportError pointing at
        ``pip install lucid[numpy]``.
        """
        from lucid._factories.converters import _require_numpy

        np = _require_numpy("Tensor.numpy()")
        raw = self._impl.data_as_python()
        return np.asarray(raw)

    # ── DLPack protocol ──────────────────────────────────────────────────
    #
    # Both methods route through ``self.numpy()`` so any DLPack-aware
    # consumer (reference framework / JAX / etc) can ingest a Lucid tensor directly:
    #
    #     >>> any_dlpack_consumer.from_dlpack(lucid.zeros(3))
    #
    # Metal tensors silently round-trip through CPU (numpy() already
    # does that).  A native engine-side DLPack export is filed as
    # future work.

    def __dlpack__(self, stream: object | None = None) -> object:
        """Return a DLPack PyCapsule view of this tensor (CPU memory)."""
        return (
            self.numpy().__dlpack__(stream=stream)
            if stream is not None
            else self.numpy().__dlpack__()
        )

    def __dlpack_device__(self) -> tuple[int, int]:
        """Return ``(device_type, device_id)`` per the DLPack spec.

        Always reports CPU because the export goes through NumPy.
        ``1 == kDLCPU``.
        """
        return (1, 0)

    def tolist(self) -> list[object] | int | float | bool:
        """Return the tensor as a nested Python list."""
        return self.numpy().tolist()

    def contiguous(self) -> Self:
        """Return a contiguous copy of this tensor."""
        return _wrap(_C_engine.contiguous(self._impl))  # type: ignore[return-value]

    def unfold(self, dimension: int, size: int, step: int) -> Tensor:
        """Return a view with an extra dimension for sliding-window slices.

        Each slice along *dimension* has *size* elements with stride *step*.
        Output shape: (..., L, size) where L = (dim_size - size) // step + 1.
        Backed by C++ engine.
        """
        return _wrap(_C_engine.unfold_dim(self._impl, dimension, size, step))  # type: ignore[return-value]

    def scatter_add(self, dim: int, index: Tensor, src: Tensor) -> Tensor:
        """Out-of-place scatter-add along *dim*. Returns a new tensor."""
        from lucid._ops import scatter_add as _sa

        return _sa(self, dim, index, src)  # type: ignore[return-value]

    @property
    def data(self) -> Self:
        """Return this tensor's data without gradient tracking."""
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
        return _tensor_repr(self)

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

    def new_empty(
        self,
        *size: int,
        dtype: dtype | None = None,
        device: device | str | None = None,
        requires_grad: bool = False,
    ) -> Self:
        """Return an uninitialized tensor with the given size, inheriting dtype/device."""
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _empty(
            *size,
            dtype=_dtype,
            device=_device,  # type: ignore[return-value]
            requires_grad=requires_grad,
        )

    def new_zeros(
        self,
        *size: int,
        dtype: dtype | None = None,
        device: device | str | None = None,
        requires_grad: bool = False,
    ) -> Self:
        """Return a zeros tensor with the given size, inheriting dtype/device."""
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _zeros(
            *size,
            dtype=_dtype,
            device=_device,  # type: ignore[return-value]
            requires_grad=requires_grad,
        )

    def new_ones(
        self,
        *size: int,
        dtype: dtype | None = None,
        device: device | str | None = None,
        requires_grad: bool = False,
    ) -> Self:
        """Return an all-ones tensor with the given size, inheriting dtype/device."""
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _ones(
            *size,
            dtype=_dtype,
            device=_device,  # type: ignore[return-value]
            requires_grad=requires_grad,
        )

    def new_full(
        self,
        size: tuple[int, ...],
        fill_value: float,
        dtype: dtype | None = None,
        device: device | str | None = None,
        requires_grad: bool = False,
    ) -> Self:
        """Return a tensor filled with fill_value, inheriting dtype/device."""
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _full(
            size,
            fill_value,
            dtype=_dtype,
            device=_device,  # type: ignore[return-value]
            requires_grad=requires_grad,
        )

    def new_tensor(
        self,
        data: np.ndarray | list[object] | int | float | bool | Tensor,
        dtype: dtype | None = None,
        device: device | str | None = None,
        requires_grad: bool = False,
    ) -> Self:
        """Return a new tensor from data, inheriting dtype/device."""
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _tensor_fn(
            data,
            dtype=_dtype,
            device=_device,  # type: ignore[return-value]
            requires_grad=requires_grad,
        )

    # ── size / element info ───────────────────────────────────────────────────

    def element_size(self) -> int:
        """Return the size of each element in bytes."""
        return self.dtype.itemsize

    @property
    def itemsize(self) -> int:
        """Alias for :meth:`element_size` — bytes per element."""
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        """Total number of bytes consumed by the tensor data."""
        return self._impl.numel() * self.dtype.itemsize

    def stride(self, dim: int | None = None) -> tuple[int, ...] | int:
        """Return the strides of the tensor in *element* counts.

        Parameters
        ----------
        dim : int, optional
            If given, return the stride along that dimension only.

        Returns
        -------
        tuple[int, ...] or int
            Element-count strides (same semantics as the reference framework).
        """
        byte_strides: list[int] = list(self._impl.stride)
        itemsz: int = self.dtype.itemsize
        elem_strides = tuple(s // itemsz for s in byte_strides)
        if dim is None:
            return elem_strides
        return elem_strides[dim]

    def data_ptr(self) -> int:
        """Return the address of the first element as an integer.

        On Apple Silicon the tensor lives in unified memory; this method
        returns a best-effort identifier derived from the storage object.
        Use :meth:`numpy` + ``ndarray.ctypes.data`` for interop that
        requires the actual pointer.
        """
        # id() of the impl object is a stable, process-unique identifier
        # suitable for equality checks (e.g. detecting aliasing) even if not
        # the raw memory address.
        return id(self._impl)

    def storage_offset(self) -> int:
        """Return the offset (in elements) of the first element in storage.

        Contiguous tensors always return ``0``.  Non-contiguous view tensors
        may return a non-zero offset; Lucid currently represents all tensors
        as contiguous so this always returns ``0``.
        """
        return 0

    @property
    def H(self) -> Tensor:
        """Conjugate transpose.

        For real tensors this is identical to :attr:`mT`.  For complex
        tensors the elements are conjugated before transposing.
        """
        if self.is_complex():
            from lucid._ops.composite import conj as _conj

            return _conj(self).mT  # type: ignore[return-value]
        return self.mT  # type: ignore[return-value]

    def type(self, dtype: str | None = None) -> str | Tensor:
        """Return or cast the tensor type.

        * ``t.type()`` — return a string like ``'lucid.FloatTensor'``.
        * ``t.type('lucid.DoubleTensor')`` — cast and return the new tensor.

        Supported type strings: ``FloatTensor``, ``DoubleTensor``,
        ``HalfTensor``, ``IntTensor``, ``LongTensor``, ``BoolTensor``,
        ``ShortTensor``, ``ByteTensor``.
        """
        _DTYPE_STR: dict[str, object] = {
            "lucid.FloatTensor": float32,
            "lucid.DoubleTensor": float64,
            "lucid.HalfTensor": float16,
            "lucid.IntTensor": int32,
            "lucid.LongTensor": int64,
            "lucid.BoolTensor": bool_,
            "lucid.ShortTensor": int16,
            "lucid.ByteTensor": int8,
        }
        _DTYPE_TO_STR: dict[object, str] = {v: k for k, v in _DTYPE_STR.items()}
        if dtype is None:
            return _DTYPE_TO_STR.get(self.dtype, f"lucid.{self.dtype}")
        if dtype not in _DTYPE_STR:
            raise TypeError(
                f"Tensor.type(): unrecognised type string '{dtype}'. "
                f"Valid values: {list(_DTYPE_STR)}"
            )
        return self.to(_DTYPE_STR[dtype])  # type: ignore[arg-type]

    def get_device(self) -> int:
        """Return the device index.

        Returns ``0`` for Metal (GPU) tensors and ``-1`` for CPU tensors,
        following the reference framework's convention.
        """
        return 0 if self.is_metal else -1

    def pin_memory(self, device: object = None) -> Tensor:
        """No-op on Apple Silicon.

        Apple Silicon uses unified memory — all tensors are directly accessible
        by both CPU and GPU without explicit pinning.  Returns ``self``.
        """
        return self  # type: ignore[return-value]

    def is_pinned(self, device: object = None) -> bool:
        """Return ``False`` — pinned memory is not applicable on Apple Silicon."""
        return False

    @property
    def is_cuda(self) -> bool:
        """``False`` — Lucid targets Apple Silicon (Metal/MLX); this device class is not available."""
        return False

    def reshape_as(self, other: Tensor) -> Tensor:
        """Return a tensor with the same data reshaped to ``other.shape``."""
        return Tensor.__new_from_impl__(  # type: ignore[return-value]
            _C_engine.reshape(self._impl, list(other._impl.shape))
        )

    class _UntypedStorage:
        """Minimal storage object returned by :meth:`Tensor.untyped_storage`."""

        def __init__(self, tensor: Tensor) -> None:
            self._tensor = tensor

        def data_ptr(self) -> int:
            """Pointer-like identifier for the underlying storage."""
            return self._tensor.data_ptr()

        def size(self) -> int:
            """Total size in bytes."""
            return self._tensor.nbytes

        def nbytes(self) -> int:
            """Alias for :meth:`size`."""
            return self.size()

        def __len__(self) -> int:
            return self.size()

        def __repr__(self) -> str:
            return (
                f"UntypedStorage(nbytes={self.size()}, "
                f"device={self._tensor.device})"
            )

    def untyped_storage(self) -> _UntypedStorage:
        """Return the underlying storage object.

        The returned object exposes :meth:`data_ptr`, :meth:`size`,
        and :meth:`nbytes` — the subset needed for common introspection
        patterns.
        """
        return Tensor._UntypedStorage(self)

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
        filled = _C_engine.full(
            list(self._impl.shape), value, self._impl.dtype, self._impl.device
        )
        self._impl.copy_from(filled)
        return self

    def copy_(self, other: Self) -> Self:
        """Copy data from other into this tensor in-place."""
        src = _C_engine.contiguous(other._impl)
        self._impl.copy_from(src)
        return self

    # ``flip`` / ``fliplr`` / ``flipud`` are auto-injected from the registry
    # (see ``_ops/_registry.py``); the previous explicit definitions
    # duplicated that path.

    def index_select(self, dim: int, index: Self) -> Self:
        """Select elements along dim using integer index tensor."""
        out_shape = list(self._impl.shape)
        k = index._impl.shape[0]
        out_shape[dim] = k
        # Reshape 1-D index to broadcast over all other dims via engine ops.
        bcast_shape = [1] * len(out_shape)
        bcast_shape[dim] = k
        idx_rs = _C_engine.reshape(index._impl, bcast_shape)
        idx_bc = _C_engine.broadcast_to(idx_rs, out_shape)
        return Tensor.__new_from_impl__(  # type: ignore[return-value]
            _C_engine.gather(self._impl, idx_bc, dim)
        )

    def masked_select(self, mask: Self) -> Self:
        """Return a 1-D tensor of elements where mask is True."""
        return Tensor.__new_from_impl__(  # type: ignore[return-value]
            _C_engine.masked_select(self._impl, mask._impl)
        )

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
            w_impl = _C_engine.full(
                list(diff._impl.shape),
                float(weight),
                diff._impl.dtype,
                diff._impl.device,
            )
            scaled = Tensor.__new_from_impl__(  # type: ignore[return-value]
                _C_engine.mul(w_impl, diff._impl)
            )
        return Tensor.__new_from_impl__(_C_engine.add(self._impl, scaled._impl))  # type: ignore[return-value]

    def where(self, condition: Self, other: Self | float) -> Self:
        """Return elements from self where condition is True, else from other."""
        if not isinstance(other, Tensor):
            other_impl = _C_engine.full(
                list(self._impl.shape),
                float(other),
                self._impl.dtype,
                self._impl.device,
            )
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
            a_impl = _ss(result._impl, d, slice(1, length, 1))
            b_impl = _ss(result._impl, d, slice(0, length - 1, 1))
            result = Tensor.__new_from_impl__(_C_engine.sub(a_impl, b_impl))  # type: ignore[assignment]
        return result

    def addmm(
        self, mat1: Self, mat2: Self, beta: float = 1.0, alpha: float = 1.0
    ) -> Self:
        """beta * self + alpha * mat1 @ mat2."""
        mm_impl = _C_engine.matmul(mat1._impl, mat2._impl)
        if alpha != 1.0:
            a_impl = _C_engine.full(
                list(mm_impl.shape), alpha, mm_impl.dtype, mm_impl.device
            )
            mm_impl = _C_engine.mul(a_impl, mm_impl)
        self_impl = self._impl
        if beta != 1.0:
            b_impl = _C_engine.full(
                list(self_impl.shape), beta, self_impl.dtype, self_impl.device
            )
            self_impl = _C_engine.mul(b_impl, self_impl)
        return Tensor.__new_from_impl__(_C_engine.add(self_impl, mm_impl))  # type: ignore[return-value]

    def bmm(self, mat2: Self) -> Self:
        """Batch matrix multiplication: (B,n,m) @ (B,m,p) → (B,n,p)."""
        return Tensor.__new_from_impl__(_C_engine.matmul(self._impl, mat2._impl))  # type: ignore[return-value]

    # ── zero_() helper ───────────────────────────────────────────────────────

    def zero_(self) -> Self:
        """Fill this tensor with zeros in-place."""
        result = _C_engine.mul_(
            self._impl,
            _C_engine.zeros(self._impl.shape, self._impl.dtype, self._impl.device),
        )
        self._impl = result
        return self

    # ── pickling support (required for multiprocessing DataLoader) ────────────

    def __reduce__(self) -> tuple:
        # Wire format mirrors the lucid.serialization v3 contract — raw
        # bytes + (shape, dtype name, device) — so multiprocessing pickle
        # is numpy-free.
        return (
            _tensor_unpickle,
            (
                self._impl.to_bytes(),
                list(self._impl.shape),
                self.dtype._name,
                self._impl.device,
                self._impl.requires_grad,
            ),
        )


def _tensor_unpickle(
    raw_bytes: bytes,
    shape: list[int],
    dtype_name: str,
    device: object,
    requires_grad: bool,
) -> Tensor:
    """Top-level helper so multiprocessing (spawn) can pickle/unpickle Tensor."""
    from lucid._dtype import _resolve_dtype_name, to_engine_dtype

    eng_dtype = to_engine_dtype(_resolve_dtype_name(dtype_name))
    impl = _C_engine.TensorImpl.from_bytes(
        raw_bytes, list(shape), eng_dtype, device, requires_grad
    )
    return Tensor.__new_from_impl__(impl)  # type: ignore[return-value]


# ── inject dunders and methods after class definition ────────────────────────
from lucid._tensor._repr import tensor_repr as _tensor_repr  # noqa: E402
from lucid._tensor._indexing import _select_slice as _ss  # noqa: E402
from lucid._tensor._dunders import _inject_dunders  # noqa: E402
from lucid._tensor._methods import _inject_methods  # noqa: E402
from lucid._tensor._to import _inject_to  # noqa: E402

_inject_dunders(Tensor)
_inject_methods(Tensor)
_inject_to(Tensor)
