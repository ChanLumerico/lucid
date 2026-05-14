from typing import TYPE_CHECKING, Callable, ClassVar, Self, Iterator, overload

if TYPE_CHECKING:
    import numpy as np

from lucid._C import engine as _C_engine
from lucid._dtype import (
    dtype,
    dtype as _dtype_cls,
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
from lucid._device import device, device as _device_cls, _device_from_engine
from lucid._dispatch import _wrap, _impl_with_grad
from lucid._factories.creation import (
    zeros as _zeros,
    ones as _ones,
    empty as _empty,
    full as _full,
)
from lucid._factories.converters import tensor as _tensor_fn, _to_impl

if TYPE_CHECKING:
    from lucid.autograd._hooks import RemovableHandle as _RemovableHandle


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
        r"""Construct a Tensor from Python data, a NumPy array, or another Tensor.

        The data is funnelled through :func:`lucid._factories.converters._to_impl`,
        the canonical bridge between the outside world and Lucid's C++
        ``TensorImpl``. Python scalars become 0-d tensors; nested lists are
        recursively flattened with shape inference; NumPy arrays cross the
        single sanctioned host-to-engine boundary; and existing ``Tensor``
        sources are rewrapped (optionally with cast and/or device transfer).

        Parameters
        ----------
        data : ndarray | list | int | float | bool | Tensor
            Source data. Python scalars become 0-d tensors; lists are
            recursively converted; NumPy arrays use the bridge boundary
            documented in :mod:`lucid._factories.converters`; another Tensor
            is shallow-copied to a new Tensor with possibly different
            ``dtype`` / ``device`` / ``requires_grad``.
        dtype : dtype | str | None, optional
            Element type. If ``None``, inferred from ``data``.
        device : device | str | None, optional
            Target device (``"cpu"`` or ``"metal"``). If ``None``, defaults
            to the source device or ``"cpu"`` for host data.
        requires_grad : bool, optional
            Whether to record operations on this tensor for autograd.

        Returns
        -------
        Tensor
            A freshly constructed tensor whose storage is owned by the
            engine.

        Notes
        -----
        Constructor is one of the six sanctioned host-to-engine bridge
        boundaries (rule **H4**). NumPy arrays cross the bridge exactly
        once here; afterwards the data is owned by Lucid's engine. The
        resulting layout is C-contiguous row-major:

        .. math::

            \text{stride}[i] = e \cdot \prod_{j=i+1}^{d-1} s_j

        where :math:`e` is the element size in bytes and :math:`s_j` are
        the dimension sizes.

        Examples
        --------
        >>> import lucid
        >>> lucid.Tensor([1.0, 2.0, 3.0])
        tensor([1., 2., 3.])
        >>> lucid.Tensor([[1, 2], [3, 4]], dtype=lucid.int64).shape
        (2, 2)
        >>> x = lucid.Tensor(3.14, requires_grad=True)
        >>> x.requires_grad
        True
        """
        self._impl: _C_engine.TensorImpl = _to_impl(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )

    @classmethod
    def __new_from_impl__(cls, impl: _C_engine.TensorImpl) -> Self:
        r"""Wrap an existing ``TensorImpl`` as a ``Tensor`` with zero copy.

        Internal factory used pervasively by ops, autograd, and dispatch to
        promote a freshly-produced C++ ``TensorImpl`` into a Python
        :class:`Tensor` without re-running the full constructor (which would
        re-validate / re-convert the data). Bypasses :meth:`__init__`
        entirely by calling :func:`object.__new__` and binding ``_impl``
        directly.

        Parameters
        ----------
        impl : lucid._C.engine.TensorImpl
            Engine-side tensor object produced by a C++ kernel, autograd
            node, or factory. Ownership transfers to the new ``Tensor``.

        Returns
        -------
        Tensor
            Python-side wrapper aliasing ``impl``; no data is copied.

        Notes
        -----
        This is an **internal API**. End users should construct tensors via
        :class:`Tensor` (the public constructor) or factory functions like
        :func:`lucid.zeros` / :func:`lucid.tensor`. The factory exists so
        that the per-op overhead is one ``object.__new__`` plus one attribute
        assignment, rather than the full ``_to_impl`` validation pipeline.

        On the hot dispatch path the asymptotic cost is
        :math:`\mathcal{O}(1)` independent of ``numel``.

        Examples
        --------
        >>> import lucid
        >>> from lucid._C import engine as _C_engine
        >>> impl = _C_engine.zeros([3], _C_engine.Dtype.Float32, _C_engine.Device.CPU)
        >>> t = lucid.Tensor.__new_from_impl__(impl)
        >>> t.shape
        (3,)
        """
        obj = object.__new__(cls)
        obj._impl = impl
        return obj

    @property
    def impl(self) -> _C_engine.TensorImpl:
        r"""Access the underlying C++ ``TensorImpl`` object.

        Provides read-only access to the engine-side tensor that backs this
        Python wrapper. Used by ops and the autograd engine to fetch the
        native handle without going through Python-level conversions.

        Returns
        -------
        lucid._C.engine.TensorImpl
            The engine tensor object. Lifetime is tied to ``self``; do not
            retain references beyond ``self``'s lifetime.

        Notes
        -----
        This is an internal accessor exposed for advanced use cases such as
        writing custom ops that bind directly against the engine. Public
        APIs should compose existing ``Tensor`` operations instead.
        Logically the identity projection
        :math:`\pi: \text{Tensor} \to \text{TensorImpl}` with
        :math:`\pi(t) = t.\_impl`.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> type(x.impl).__name__
        'TensorImpl'
        """
        return self._impl

    # ── metadata ─────────────────────────────────────────────────────────────

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Shape of the tensor as a tuple of integers.

        Each element gives the size of the corresponding dimension, with
        ``shape[0]`` being the outermost (batch) dimension.  A scalar tensor
        (0-d) has ``shape == ()``.

        Returns
        -------
        tuple[int, ...]
            Immutable tuple of dimension sizes.

        Notes
        -----
        For a tensor with :math:`N` elements arranged in :math:`d` dimensions
        :math:`(s_0, s_1, \ldots, s_{d-1})`, the total element count satisfies:

        .. math::

            \prod_{i=0}^{d-1} s_i = N

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3, 4, 5)
        >>> x.shape
        (3, 4, 5)
        >>> lucid.tensor(1.0).shape
        ()
        """
        return tuple(self._impl.shape)

    @property
    def dtype(self) -> dtype:
        r"""Data type of the tensor elements.

        Reflects the numeric format used to store each element in memory.
        Lucid supports the following dtypes:

        * ``lucid.float32`` (default for floating-point)
        * ``lucid.float64``
        * ``lucid.float16``
        * ``lucid.bfloat16``
        * ``lucid.int8``, ``lucid.int16``, ``lucid.int32``, ``lucid.int64``
        * ``lucid.bool_``
        * ``lucid.complex64``

        Returns
        -------
        lucid.dtype
            The element type of this tensor.

        Notes
        -----
        The memory footprint of a tensor is :math:`N \times e` bytes, where
        :math:`N` is the total number of elements and :math:`e` is
        ``dtype.itemsize``.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3).dtype
        lucid.float32
        >>> lucid.zeros(3, dtype=lucid.int64).dtype
        lucid.int64
        """
        return _ENGINE_TO_DTYPE[self._impl.dtype]  # type: ignore[return-value]

    @property
    def device(self) -> device:
        r"""Device on which this tensor is stored.

        Lucid tensors reside on one of two devices:

        * ``cpu`` — Apple Accelerate (vDSP / BNNS / BLAS / LAPACK).
        * ``metal`` — Apple Metal GPU via the MLX backend.

        Returns
        -------
        lucid.device
            The device object for this tensor.

        Notes
        -----
        On Apple Silicon the CPU and GPU share the same physical DRAM
        (unified memory architecture).  Moving a tensor between devices
        copies the *logical* dispatch target, not physical bytes, unless
        the tensor is in a non-shared Metal buffer. The dispatch routing
        rule is

        .. math::

            \text{backend}(t) = \begin{cases}
                \text{Accelerate} & \text{if } t.\text{device} = \text{cpu} \\
                \text{MLX/Metal} & \text{if } t.\text{device} = \text{metal}.
            \end{cases}

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> x.device
        device(type='cpu')
        >>> x.metal().device
        device(type='metal')
        """
        return _device_from_engine(self._impl.device)  # type: ignore[return-value]

    @property
    def ndim(self) -> int:
        r"""Number of dimensions (rank) of the tensor.

        Equivalent to ``len(tensor.shape)``.  A scalar tensor has ``ndim == 0``,
        a vector has ``ndim == 1``, a matrix has ``ndim == 2``, and so on.

        Returns
        -------
        int
            The number of dimensions.

        Examples
        --------
        >>> import lucid
        >>> lucid.tensor(3.14).ndim
        0
        >>> lucid.zeros(5).ndim
        1
        >>> lucid.zeros(2, 3).ndim
        2

        Notes
        -----
        For a shape tuple :math:`(s_0, \ldots, s_{d-1})` the rank is simply

        .. math::

            \text{ndim} = d = |\text{shape}|.
        """
        return len(self._impl.shape)

    @property
    def T(self) -> Self:
        r"""Tensor with all dimensions reversed.

        For a 2-D tensor this is the standard matrix transpose.  For tensors
        with more than two dimensions, the axis order is fully reversed:
        axis ``i`` maps to axis ``ndim - 1 - i``.

        Returns
        -------
        Tensor
            A view (or copy) with reversed dimension order.

        Notes
        -----
        For a tensor of shape :math:`(s_0, s_1, \ldots, s_{d-1})`, the
        transposed tensor has shape :math:`(s_{d-1}, \ldots, s_1, s_0)`.
        This differs from :attr:`mT`, which only transposes the final two axes.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.arange(6).reshape(2, 3)
        >>> x.shape
        (2, 3)
        >>> x.T.shape
        (3, 2)
        >>> lucid.arange(24).reshape(2, 3, 4).T.shape
        (4, 3, 2)
        """
        return Tensor.__new_from_impl__(_C_engine.T(self._impl))  # type: ignore[return-value]

    @property
    def mT(self) -> Self:
        r"""Tensor with the last two dimensions transposed.

        Equivalent to calling ``lucid.swapaxes(x, -2, -1)``.  Useful for
        batched linear-algebra operations where the batch dimensions should
        remain untouched.

        Returns
        -------
        Tensor
            A view (or copy) with axes ``-2`` and ``-1`` swapped.

        Notes
        -----
        For a tensor of shape :math:`(\ldots, m, n)` the result has shape
        :math:`(\ldots, n, m)`.  The leading batch dimensions are unchanged.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(4, 3, 5)
        >>> x.mT.shape
        (4, 5, 3)
        """
        return Tensor.__new_from_impl__(_C_engine.mT(self._impl))  # type: ignore[return-value]

    @property
    def is_metal(self) -> bool:
        r"""Whether this tensor resides on the Metal (GPU) device.

        On Apple Silicon the GPU backend is Apple Metal accessed through the
        MLX library.  CPU tensors use Apple Accelerate instead.

        Returns
        -------
        bool
            ``True`` if the tensor is on the Metal device, ``False`` for CPU.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> x.is_metal
        False
        >>> x.metal().is_metal
        True

        Notes
        -----
        Equivalent predicate :math:`\text{is\_metal}(t) = (t.\text{device} = \text{GPU})`.
        On Apple Silicon the Metal backend dispatches via MLX, which lazily
        builds a graph and evaluates on the GPU when results are observed.
        """
        return self._impl.device == _C_engine.Device.GPU

    @property
    def is_shared(self) -> bool:
        r"""True when backed by a Metal ``MTLResourceStorageModeShared`` buffer.

        Shared-memory tensors live in Apple Silicon unified DRAM and are
        simultaneously accessible from CPU and GPU without a ``memcpy``.
        Create them with ``lucid.metal.shared_tensor()`` or promote an
        existing tensor with ``lucid.metal.to_shared()``.

        Returns
        -------
        bool
            ``True`` if the tensor's storage uses Metal's shared storage
            mode, ``False`` otherwise (including all pure-CPU tensors and
            private-mode Metal buffers).

        Notes
        -----
        On Apple Silicon, shared storage permits zero-copy hand-off between
        CPU and GPU code paths because both engines see the same physical
        DRAM page. The trade-off is that Metal's GPU caches cannot be used
        as aggressively as with private storage; for compute-bound GPU
        kernels, private mode is often preferable. Predicate:
        :math:`\text{is\_shared}(t) = \mathbf{1}\{\text{storage\_mode}(t) = \text{Shared}\}`.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> x.is_shared
        False
        >>> y = lucid.metal.shared_tensor((3,))  # doctest: +SKIP
        >>> y.is_shared  # doctest: +SKIP
        True
        """
        return self._impl.is_metal_shared

    @property
    def is_leaf(self) -> bool:
        r"""Whether this tensor is a leaf in the autograd computation graph.

        A tensor is a leaf if it was created directly by the user (not as the
        result of an operation) or if it does not require gradients.  Only
        leaf tensors accumulate gradients into their ``.grad`` attribute during
        :meth:`backward`.

        Returns
        -------
        bool
            ``True`` for leaf tensors, ``False`` for intermediate results.

        Notes
        -----
        In the computation graph, leaf nodes are the "inputs" to the
        forward pass.  All tensors created with ``requires_grad=False``
        are leaves by definition.  Tensors created with
        ``requires_grad=True`` directly by the user are also leaves.
        Tensors produced by differentiable operations on
        ``requires_grad=True`` inputs are *not* leaves — they are
        intermediate nodes and their ``.grad`` is not retained unless
        :meth:`retain_grad` is called. Predicate:

        .. math::

            \text{is\_leaf}(t) = \mathbf{1}\{\text{grad\_fn}(t) = \bot\}.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
        >>> x.is_leaf
        True
        >>> y = x * 2          # result of an op — not a leaf
        >>> y.is_leaf
        False
        """
        return self._impl.is_leaf

    @property
    def requires_grad(self) -> bool:
        """Whether gradient computation is enabled for this tensor.

        When ``True``, operations involving this tensor are recorded in the
        computation graph so that :meth:`backward` can propagate gradients
        back through them.

        Setting this attribute on a leaf tensor promotes or demotes it from
        the autograd graph.  Calling ``requires_grad = True`` on an
        intermediate (non-leaf) tensor raises a ``RuntimeError`` because
        those tensors are not user-controlled inputs.

        Returns
        -------
        bool
            ``True`` if gradients are tracked for this tensor.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0])
        >>> x.requires_grad
        False
        >>> x.requires_grad = True
        >>> x.requires_grad
        True

        Notes
        -----
        Tensors with ``requires_grad=True`` become nodes in the autograd DAG.
        Operations consuming them produce non-leaf tensors that also require
        gradients (transitive closure of the flag along the forward graph).
        """
        return self._impl.requires_grad

    @requires_grad.setter
    def requires_grad(self, v: bool) -> None:
        """Set whether this tensor participates in autograd, in place.

        Replaces the underlying ``TensorImpl`` with one whose
        gradient-tracking flag is set to ``v``. The storage is shared, so
        only the flag (and any owning autograd node bookkeeping) changes.

        Parameters
        ----------
        v : bool
            New value for the ``requires_grad`` flag. ``True`` enrolls the
            tensor in the autograd graph; ``False`` removes it.

        Raises
        ------
        RuntimeError
            If ``v`` is ``True`` and ``self`` is not a leaf — non-leaf
            (intermediate) tensors inherit ``requires_grad`` from their
            inputs and cannot be flipped on directly.

        Notes
        -----
        For chaining, prefer :meth:`requires_grad_` which returns ``self``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> x.requires_grad = True
        >>> x.requires_grad
        True
        """
        self._impl = _impl_with_grad(self._impl, v)

    def numel(self) -> int:
        r"""Return the total number of elements in the tensor.

        Equivalent to the product of all dimension sizes, i.e. the result of
        evaluating :math:`\prod_{i} s_i` over the shape tuple.  For a scalar
        (0-d) tensor this is ``1``; for an empty tensor it is ``0``.

        Returns
        -------
        int
            Total element count.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3, 4).numel()
        12
        >>> lucid.tensor(5.0).numel()
        1

        Notes
        -----
        Used by the engine to allocate storage of size
        :math:`N \cdot e` bytes, where :math:`e = \text{dtype.itemsize}`.
        Convenient invariant: ``tensor.numel() == tensor.nbytes // tensor.itemsize``.
        """
        return int(self._impl.numel())

    def dim(self) -> int:
        r"""Return the number of dimensions (rank) of the tensor.

        This is identical to the :attr:`ndim` property.  It exists as a
        method for API compatibility with code that calls ``tensor.dim()``.

        Returns
        -------
        int
            Number of dimensions.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(2, 3, 4).dim()
        3
        >>> lucid.tensor(1.0).dim()
        0

        Notes
        -----
        Equivalent to ``len(tensor.shape)``; matches the conventional rank
        :math:`d` such that :math:`\text{shape} \in \mathbb{N}^d`.
        """
        return len(self._impl.shape)

    @overload
    def size(self, dim: int) -> int:
        """Overload: with an integer ``dim``, returns the size of that axis as ``int``."""
        ...

    @overload
    def size(self, dim: None = ...) -> tuple[int, ...]:
        """Overload: with no argument, returns the full shape tuple."""
        ...

    def size(self, dim: int | None = None) -> int | tuple[int, ...]:
        r"""Return the size of a specific dimension, or the full shape tuple.

        Parameters
        ----------
        dim : int, optional
            Dimension index to query.  Supports negative indexing.
            If omitted, the full shape tuple is returned.

        Returns
        -------
        int
            Size of the requested dimension (when ``dim`` is given).
        tuple[int, ...]
            Full shape tuple (when ``dim`` is omitted).

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3, 4, 5)
        >>> x.size()
        (3, 4, 5)
        >>> x.size(0)
        3
        >>> x.size(-1)
        5

        Notes
        -----
        The full shape and the element count are related by

        .. math::

            \text{numel} = \prod_{i=0}^{d-1} \text{size}(i).
        """
        s = tuple(self._impl.shape)
        if dim is not None:
            return s[dim]
        return s

    def is_contiguous(self) -> bool:
        r"""Return ``True`` if the tensor's data is stored contiguously in memory.

        A contiguous tensor stores its elements in a single unbroken block of
        memory in C (row-major) order — i.e. the stride of each dimension equals
        the product of all *later* dimension sizes times the element size.

        Non-contiguous tensors can arise from operations such as slicing,
        transposing, or permuting axes.  Many C++ kernel paths require
        contiguous input; call :meth:`contiguous` to get a contiguous copy
        when needed.

        Returns
        -------
        bool
            ``True`` if the tensor uses a single contiguous memory block in
            C-order, ``False`` otherwise.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3, 4)
        >>> x.is_contiguous()
        True
        >>> x.T.is_contiguous()   # transpose is not contiguous
        False

        Notes
        -----
        A tensor of shape :math:`(s_0, \ldots, s_{d-1})` is contiguous iff
        its strides satisfy the C-order recurrence

        .. math::

            \text{stride}[d-1] = 1, \quad
            \text{stride}[i] = \text{stride}[i+1] \cdot s_{i+1}

        (in element units). Many Accelerate/MLX kernels require contiguous
        inputs; if not, call :meth:`contiguous` first.
        """
        return self._impl.is_contiguous()

    # ── autograd ─────────────────────────────────────────────────────────────

    @property
    def grad(self) -> Self | None:
        r"""Accumulated gradient tensor, or ``None`` if not yet computed.

        After calling :meth:`backward`, this attribute holds the gradient of
        the scalar loss with respect to this tensor — i.e.
        :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{x}}` where
        :math:`\mathbf{x}` is this tensor.

        Gradients are **accumulated** (added) across multiple :meth:`backward`
        calls.  Zero the gradient before each optimisation step with
        ``tensor.grad = None`` or ``tensor.grad.zero_()``, or use an optimiser
        that calls ``zero_grad()`` automatically.

        Only **leaf** tensors (those created directly by user code with
        ``requires_grad=True``) populate this attribute by default.
        Non-leaf (intermediate) tensors discard their gradient after the
        backward pass unless :meth:`retain_grad` was called.

        Returns
        -------
        Tensor or None
            The accumulated gradient, or ``None`` if :meth:`backward` has not
            been called yet or if this tensor does not require gradients.

        Notes
        -----
        The gradient has the same shape as the tensor itself:

        .. math::

            \text{grad.shape} = \text{self.shape}

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = (x ** 2).sum()
        >>> y.backward()
        >>> x.grad          # d(sum(x^2))/dx = 2x
        tensor([2., 4., 6.])
        """
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
        """Assign or clear the accumulated gradient buffer.

        Allows external code (optimisers, gradient-clipping utilities,
        custom training loops) to write directly into the autograd
        ``.grad`` slot. Assigning ``None`` zeroes the buffer; assigning a
        ``Tensor`` overwrites its contents.

        Parameters
        ----------
        v : Tensor or None
            New gradient value. If a ``Tensor``, its shape and dtype must
            match ``self``; the value is unwrapped to its engine impl and
            installed in place. If ``None``, the engine's ``zero_grad`` is
            called, releasing the gradient storage.

        Notes
        -----
        Common training-loop idiom — clear gradients before each step:

        .. code-block:: python

            for p in model.parameters():
                p.grad = None

        Assigning ``None`` is preferred over ``p.grad.zero_()`` because it
        frees the gradient buffer entirely (smaller memory footprint
        between steps) and avoids an unnecessary write to zero-fill.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
        >>> (x ** 2).sum().backward()
        >>> x.grad is not None
        True
        >>> x.grad = None
        >>> x.grad is None
        True
        """
        if v is None:
            self._impl.zero_grad()
        else:
            from lucid._dispatch import _unwrap

            self._impl.set_grad(_unwrap(v))

    @property
    def grad_fn(self) -> _C_engine.Node | None:
        r"""The autograd graph node that created this tensor, or ``None``.

        Every tensor produced by a differentiable operation holds a reference
        to the C++ ``Node`` (gradient function) that can propagate gradients
        back through that operation.  The graph is a directed acyclic graph
        (DAG) of such nodes; :meth:`backward` traverses it in reverse
        topological order.

        Leaf tensors (created directly, not via ops) always have
        ``grad_fn = None``.

        Returns
        -------
        lucid._C.engine.Node or None
            The gradient-computing node, or ``None`` for leaf tensors or
            tensors that do not require gradients.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
        >>> x.grad_fn is None      # leaf — no grad_fn
        True
        >>> y = x * 3
        >>> y.grad_fn is None      # result of an op — has a grad_fn
        False

        Notes
        -----
        The graph forms the chain-rule factorisation used by :meth:`backward`:

        .. math::

            \frac{\partial \mathcal{L}}{\partial \mathbf{x}}
            = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}
              \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}},

        where :math:`\mathbf{y} = f(\mathbf{x})` and ``grad_fn`` carries the
        Jacobian-vector product :math:`v \mapsto v \cdot J_f`.
        """
        return getattr(self._impl, "grad_fn", None)

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        r"""Enable or disable gradient tracking for this tensor, in-place.

        Unlike the :attr:`requires_grad` property setter, this method
        returns ``self`` so it can be chained inline:

        .. code-block:: python

            x = lucid.randn(3, 4).requires_grad_(True)

        Parameters
        ----------
        requires_grad : bool, optional
            New value for gradient tracking.  Defaults to ``True``.

        Returns
        -------
        Tensor
            ``self`` with the updated ``requires_grad`` flag.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> x.requires_grad_(True).requires_grad
        True

        Notes
        -----
        In-place flag flip: the underlying storage is preserved, but the
        ``TensorImpl`` is replaced with one whose autograd flag is
        :math:`\text{requires\_grad} \in \{\text{True}, \text{False}\}`.
        Only valid on **leaf** tensors; non-leaf tensors inherit the flag
        from their producing op and cannot be flipped on directly.
        """
        self._impl = _impl_with_grad(self._impl, requires_grad)
        return self

    def retain_grad(self) -> None:
        r"""Retain the gradient on this non-leaf tensor after :meth:`backward`.

        By default, gradients are only stored on **leaf** tensors.  Intermediate
        (non-leaf) results in the computation graph have their ``.grad``
        discarded after the backward pass to save memory.  Calling
        ``retain_grad()`` on an intermediate tensor before the forward pass
        instructs the engine to keep that gradient so it can be inspected
        afterwards.

        Notes
        -----
        This method must be called **before** the forward computation whose
        gradient you want to inspect.  Calling it after :meth:`backward` has
        no effect. Conceptually retains
        :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{y}}` for
        the intermediate node :math:`\mathbf{y}` instead of discarding it
        after its parents have consumed it.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = x * 2          # intermediate — grad normally discarded
        >>> y.retain_grad()
        >>> y.sum().backward()
        >>> y.grad             # now available: d(sum(y))/dy = [1., 1., 1.]
        tensor([1., 1., 1.])
        """
        if hasattr(self._impl, "retain_grad_"):
            self._impl.retain_grad_()

    def register_hook(
        self, hook: Callable[[Tensor], Tensor | None]
    ) -> _RemovableHandle:
        r"""Register a hook that fires when this tensor's gradient is computed.

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

        Chain-rule effect: if the hook returns a tensor :math:`\tilde g`,
        the engine substitutes :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{x}}
        \leftarrow \tilde g` before continuing backward propagation.
        A returned ``None`` leaves the gradient untouched.

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
        r"""Compute gradients via reverse-mode automatic differentiation.

        Traverses the computation graph in reverse topological order, applying
        the chain rule at each node to accumulate
        :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{x}}` into the
        ``.grad`` attribute of every leaf tensor that has
        ``requires_grad=True``.

        Parameters
        ----------
        gradient : Tensor, optional
            The seed gradient — the gradient of some external scalar with
            respect to ``self``.  Must have the same shape as ``self``.

            * For **scalar** tensors (``numel() == 1``) this may be omitted;
              Lucid uses an implicit seed of ``1.0``.
            * For **non-scalar** tensors it is **required**.  Pass the upstream
              gradient explicitly (e.g. when differentiating through a loss
              that is itself a vector).
        retain_graph : bool, optional
            If ``False`` (default), intermediate tensors and gradient functions
            stored in the computation graph are freed immediately after the
            backward pass to reclaim memory.  Set to ``True`` if you need to
            call ``backward()`` again on the same graph (e.g. to compute
            multiple gradient signals or to inspect intermediate values).
        create_graph : bool, optional
            Reserved for higher-order differentiation.  When ``True`` the
            backward pass itself is differentiable, enabling gradients of
            gradients.  Not yet fully supported; accepted for API
            compatibility.

        Raises
        ------
        RuntimeError
            If ``self`` has more than one element and ``gradient`` is not
            provided.
        RuntimeError
            If ``gradient.shape != self.shape``.

        Notes
        -----
        **The chain rule and reverse-mode AD**

        Given a scalar loss :math:`\mathcal{L}` and a sequence of operations
        :math:`\mathbf{z} = f_n(\cdots f_2(f_1(\mathbf{x})) \cdots)`,
        the chain rule gives:

        .. math::

            \frac{\partial \mathcal{L}}{\partial \mathbf{x}}
            = \frac{\partial \mathcal{L}}{\partial \mathbf{z}}
              \cdot \frac{\partial \mathbf{z}}{\partial \mathbf{x}}
            = \frac{\partial \mathcal{L}}{\partial \mathbf{z}}
              \cdot J_{f_n} \cdots J_{f_1}

        where :math:`J_{f_i}` is the Jacobian of the :math:`i`-th operation.
        Reverse-mode AD (backpropagation) evaluates this product right-to-left,
        starting from the scalar output with seed gradient
        :math:`\frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1`, so
        it computes gradients for *all* inputs in a single backward pass —
        :math:`\mathcal{O}(1)` passes regardless of the number of parameters.

        **Gradient accumulation**

        Gradients are *added* to ``tensor.grad`` rather than overwritten.
        This is intentional: it supports patterns like accumulated gradient
        steps.  Zero gradients explicitly before each optimisation step:

        .. code-block:: python

            for param in model.parameters():
                param.grad = None   # or param.grad.zero_()

        **Metal flush**

        On Metal devices Lucid calls ``TensorImpl::eval()`` before the
        backward pass.  This forces MLX to evaluate the forward graph eagerly
        so that the backward kernel sees concrete values rather than deferred
        MLX computations.  In practice this yields roughly 2× faster
        backward passes for typical model sizes.

        Examples
        --------
        Scalar output — no explicit gradient seed needed:

        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> loss = (x ** 2).sum()   # scalar
        >>> loss.backward()
        >>> x.grad                  # d(sum(x^2))/dx = 2x
        tensor([2., 4., 6.])

        Non-scalar output — must supply a gradient seed:

        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
        >>> y = x * 3               # shape (2,) — not a scalar
        >>> y.backward(lucid.ones(2))
        >>> x.grad                  # d(3x)/dx = 3 for each element
        tensor([3., 3.])

        Multiple backward passes with ``retain_graph=True``:

        >>> import lucid
        >>> x = lucid.tensor([2.0], requires_grad=True)
        >>> y = x ** 3
        >>> y.backward(retain_graph=True)   # first pass
        >>> y.backward()                    # second pass — accumulates
        >>> x.grad                          # 3*x^2 + 3*x^2 = 2 * 12 = 24
        tensor([24.])
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
        r"""Return a new tensor that shares data but is detached from the autograd graph.

        The returned tensor has the same values as ``self`` but
        ``requires_grad=False`` and no ``grad_fn``.  It is treated as a
        constant by subsequent operations — gradients will not flow through it.

        Unlike :meth:`detach_`, this method does **not** modify ``self``; it
        returns a new view (made contiguous if necessary).

        Returns
        -------
        Tensor
            A detached, contiguous tensor with the same data.

        Notes
        -----
        Common use cases:

        * Stopping gradient flow when computing a target value for a loss
          (e.g. target networks in reinforcement learning).
        * Safely converting to NumPy without triggering autograd errors.
        * Logging or visualising intermediate activations without affecting
          the graph.

        Mathematically, detach is the identity on values but injects a
        zero Jacobian into the backward pass:

        .. math::

            \text{detach}(\mathbf{x}) = \mathbf{x}, \quad
            \frac{\partial \text{detach}(\mathbf{x})}{\partial \mathbf{x}} = \mathbf{0}.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
        >>> y = x * 3
        >>> z = y.detach()
        >>> z.requires_grad
        False
        >>> z.grad_fn is None
        True
        """
        return Tensor.__new_from_impl__(
            _impl_with_grad(  # type: ignore[return-value]
                _C_engine.contiguous(self._impl), False
            )
        )

    def detach_(self) -> Self:
        r"""Detach this tensor from the autograd graph in-place.

        Clears ``requires_grad`` and removes the ``grad_fn`` so that future
        operations on ``self`` are not tracked.  Returns ``self`` for chaining.

        Unlike :meth:`detach`, this modifies the tensor in-place and does
        **not** create a new object.

        Returns
        -------
        Tensor
            ``self`` after detaching.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
        >>> y = x * 3
        >>> _ = y.detach_()
        >>> y.requires_grad
        False

        Notes
        -----
        In-place counterpart of :meth:`detach`. Mutates ``self`` so future
        ops are not tracked; equivalent to setting
        :math:`\text{requires\_grad} \leftarrow \text{False}` and
        clearing ``grad_fn``. Returns ``self`` to permit fluent chaining.
        """
        self._impl = _impl_with_grad(self._impl, False)
        return self

    def clamp_(
        self,
        min: float | None = None,
        max: float | None = None,
    ) -> Self:
        r"""Clamp all elements of this tensor to ``[min, max]``, in-place.

        Each element :math:`x_i` is replaced by:

        .. math::

            x_i \leftarrow \max(\text{min},\, \min(\text{max},\, x_i))

        Parameters
        ----------
        min : float, optional
            Lower bound.  Elements below this value are raised to ``min``.
            Defaults to :math:`-\infty` (no lower clamp).
        max : float, optional
            Upper bound.  Elements above this value are lowered to ``max``.
            Defaults to :math:`+\infty` (no upper clamp).

        Returns
        -------
        Tensor
            ``self`` after clamping.

        Notes
        -----
        At least one of ``min`` or ``max`` should be specified.  If both are
        ``None`` the call is a no-op.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([-2.0, 0.5, 3.0])
        >>> x.clamp_(min=-1.0, max=2.0)
        tensor([-1., 0.5, 2.])
        """
        lo = min if min is not None else float("-inf")
        hi = max if max is not None else float("inf")
        self._impl = _C_engine.clip_(self._impl, lo, hi)
        return self

    def clamp_min_(self, min: float) -> Self:
        r"""Raise all elements below ``min`` to ``min``, in-place.

        Shorthand for ``clamp_(min=min)``.

        Parameters
        ----------
        min : float
            Lower bound.  Elements strictly less than ``min`` are set to
            ``min``.

        Returns
        -------
        Tensor
            ``self`` after clamping.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([-1.0, 0.0, 1.0])
        >>> x.clamp_min_(0.0)
        tensor([0., 0., 1.])

        Notes
        -----
        Pointwise rectification:

        .. math::

            x_i \leftarrow \max(\text{min}, x_i).

        The non-smooth point at :math:`x = \text{min}` has subgradient
        :math:`[0, 1]`; autograd selects ``1`` for :math:`x > \text{min}`
        and ``0`` otherwise.
        """
        return self.clamp_(min=min)

    def clamp_max_(self, max: float) -> Self:
        r"""Lower all elements above ``max`` to ``max``, in-place.

        Shorthand for ``clamp_(max=max)``.

        Parameters
        ----------
        max : float
            Upper bound.  Elements strictly greater than ``max`` are set to
            ``max``.

        Returns
        -------
        Tensor
            ``self`` after clamping.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([0.5, 1.5, 2.5])
        >>> x.clamp_max_(2.0)
        tensor([0.5, 1.5, 2. ])

        Notes
        -----
        Pointwise saturation:

        .. math::

            x_i \leftarrow \min(\text{max}, x_i).

        The gradient is the indicator :math:`\mathbf{1}\{x_i < \text{max}\}`.
        """
        return self.clamp_(max=max)

    def clone(self) -> Self:
        r"""Return a deep copy of this tensor, preserving autograd history.

        Unlike :meth:`detach`, the returned tensor **remains connected** to
        the computation graph.  Gradients flow through :meth:`clone` as if
        it were the identity operation, so it can be used to safely duplicate
        a tensor while keeping the backward pass intact.

        Returns
        -------
        Tensor
            A new tensor with a contiguous copy of the same data, connected
            to the same computation graph.

        Notes
        -----
        The clone operation inserts a trivial node into the graph whose
        backward pass propagates the upstream gradient unchanged:

        .. math::

            \text{clone}(\mathbf{x}) = \mathbf{x}, \quad
            \frac{\partial \text{clone}(\mathbf{x})}{\partial \mathbf{x}} = \mathbf{I}.

        Use :meth:`detach` (or ``clone().detach()``) when you want a copy
        that is *not* connected to the graph.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = x.clone()
        >>> y.requires_grad
        True
        >>> y.data_ptr() != x.data_ptr()   # different storage
        True
        """
        impl = _C_engine.contiguous(self._impl)
        return _wrap(impl)  # type: ignore[return-value]

    # ── conversion ───────────────────────────────────────────────────────────

    def item(self) -> float | int | bool:
        r"""Return the value of a single-element tensor as a Python scalar.

        Delegates to the engine's ``TensorImpl::item`` which performs the
        single-element extraction (including IEEE-754 binary16 to float
        decoding) without going through numpy.

        Returns
        -------
        float or int or bool
            The unboxed Python scalar. Floating-point dtypes (including
            ``float16`` / ``bfloat16``) return ``float``; integer dtypes
            return ``int``; ``bool_`` returns ``bool``.

        Raises
        ------
        RuntimeError
            If the tensor has more than one element (``numel() != 1``).

        Notes
        -----
        Triggers a device-to-host synchronisation when called on a Metal
        tensor — the value cannot be inspected until any pending MLX
        computation has completed. Avoid calling in tight loops on GPU
        tensors; prefer ``tensor.cpu().tolist()`` for batch extraction.

        Defined only when :math:`\text{numel} = \prod_i s_i = 1`. ``item``
        is one of the sanctioned engine-to-host bridge points (rule **H4**)
        and detaches from autograd: the returned Python scalar carries no
        gradient information.

        Examples
        --------
        >>> import lucid
        >>> lucid.tensor(3.14).item()
        3.140000104904175
        >>> lucid.tensor(7, dtype=lucid.int64).item()
        7
        """
        return self._impl.item()

    def numpy(self) -> np.ndarray:  # type: ignore[type-arg]
        r"""Return the tensor as a NumPy array (CPU only).

        Imports numpy lazily — the rest of Lucid stays numpy-free unless
        the user explicitly bridges through this method. When numpy is
        not installed, raises an ``ImportError`` pointing at
        ``pip install lucid[numpy]``.

        Returns
        -------
        numpy.ndarray
            A NumPy view (or copy) of the tensor's data. Shape and dtype
            mirror the source; the array lives in host memory.

        Raises
        ------
        ImportError
            If NumPy is not installed.
        RuntimeError
            If the tensor lives on the Metal device and cannot be evaluated
            to host memory.

        Notes
        -----
        This is one of the sanctioned bridge points between Lucid and the
        outside world (see project rule **H4**). The returned ``ndarray``
        does **not** participate in autograd; downstream NumPy operations
        will not produce gradients.

        Layout is C-contiguous; for a tensor of shape :math:`(s_0, \ldots, s_{d-1})`
        and itemsize :math:`e`, the NumPy strides equal

        .. math::

            \text{nstride}[i] = e \cdot \prod_{j=i+1}^{d-1} s_j.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> arr = x.numpy()
        >>> arr.shape
        (2, 2)
        >>> arr.dtype
        dtype('float32')
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
        r"""Export this tensor as a DLPack ``PyCapsule`` for zero-copy interop.

        DLPack is the open cross-framework tensor exchange specification —
        any consumer implementing the protocol (the reference framework,
        JAX, CuPy, TVM, and many others) can ingest a Lucid tensor without
        a Python-level data copy by calling ``their_framework.from_dlpack(t)``.

        The current implementation routes through :meth:`numpy`, so the
        export always lands on host memory (DLPack device ``kDLCPU``).
        Metal tensors silently round-trip through CPU; a future native
        engine-side DLPack export will avoid the trip.

        Parameters
        ----------
        stream : object, optional
            Stream / queue handle for synchronisation, per the DLPack v0.8+
            protocol. Forwarded to NumPy's ``__dlpack__``. ``None`` means
            no explicit synchronisation is required (the default on CPU).

        Returns
        -------
        PyCapsule
            A capsule wrapping a ``DLManagedTensor`` struct. The capsule
            must be consumed (renamed to ``"used_dltensor"``) by the
            receiver; otherwise its destructor releases the underlying
            buffer.

        Notes
        -----
        See the DLPack specification at
        https://dmlc.github.io/dlpack/latest/ for the precise wire format
        and version semantics. Lucid follows the v0.8 protocol revision.

        Bridge boundary (rule **H4**): no autograd information crosses the
        capsule. Logically the export is the identity on values,
        :math:`\text{dlpack}(\mathbf{x}) \equiv \mathbf{x}`, with the
        consumer responsible for interpreting strides/dtype correctly.

        Examples
        --------
        >>> import lucid
        >>> import numpy as np
        >>> x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> arr = np.from_dlpack(x)
        >>> arr.shape
        (2, 2)
        """
        return (
            self.numpy().__dlpack__(stream=stream)
            if stream is not None
            else self.numpy().__dlpack__()
        )

    def __dlpack_device__(self) -> tuple[int, int]:
        r"""Return ``(device_type, device_id)`` per the DLPack specification.

        Companion to :meth:`__dlpack__` — DLPack consumers query this method
        before calling ``__dlpack__`` so they know how to synchronise and
        which memory space the capsule will reference.

        Returns
        -------
        tuple[int, int]
            ``(device_type, device_id)``. Always reports ``(1, 0)``, where
            ``1 == kDLCPU`` in the DLPack device-type enum, because the
            export currently goes through NumPy (host memory). The
            ``device_id`` of ``0`` is conventional for non-indexed devices.

        Notes
        -----
        DLPack device types of interest:

        * ``1`` — ``kDLCPU`` (this implementation)
        * ``2`` — ``kDLCUDA``
        * ``8`` — ``kDLMetal``

        A future native Metal-side export will return ``(8, 0)`` when the
        tensor resides on the GPU, avoiding the CPU round-trip. The
        device tag is therefore the constant pair
        :math:`(\texttt{kDLCPU}, 0) = (1, 0)` for all current tensors.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3).__dlpack_device__()
        (1, 0)
        """
        return (1, 0)

    def tolist(self) -> list[object] | int | float | bool:
        r"""Return the tensor contents as a nested Python list (or scalar).

        Converts the tensor to a standard Python object:

        * 0-d tensor → a Python scalar (``int``, ``float``, or ``bool``).
        * 1-d tensor → a flat ``list``.
        * N-d tensor → a nested ``list`` of depth ``N``.

        The conversion routes through :meth:`numpy` and then calls
        ``ndarray.tolist()``, so the element types follow NumPy's
        Python-type promotion (e.g. ``float32`` → Python ``float``).

        Returns
        -------
        list or int or float or bool
            Nested Python representation of the tensor data.

        Examples
        --------
        >>> import lucid
        >>> lucid.tensor([[1, 2], [3, 4]]).tolist()
        [[1, 2], [3, 4]]
        >>> lucid.tensor(3.14).tolist()
        3.14

        Notes
        -----
        Routes through the NumPy bridge (rule **H4**) and forces a
        device-to-host synchronisation for Metal tensors. The nested-list
        depth equals the tensor rank :math:`d`; the total number of
        leaves equals :math:`\prod_i s_i`. Autograd information is
        dropped — the returned Python objects are pure value copies.
        """
        return self.numpy().tolist()

    def contiguous(self) -> Self:
        r"""Return a tensor whose data is stored in contiguous C-order memory.

        If ``self`` is already contiguous (``self.is_contiguous() == True``),
        this may return a view or a copy depending on the backend; the result
        is always safe to pass to kernels that require contiguous input.

        A tensor can become non-contiguous after operations like
        transposing, permuting, or slicing with non-unit strides.  Making
        it contiguous rewrites the data into a fresh buffer with strides
        matching C row-major layout:

        .. math::

            \text{stride}[i] = \prod_{j=i+1}^{d-1} \text{shape}[j]

        Returns
        -------
        Tensor
            A contiguous tensor with the same values and shape.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3, 4).T    # transposed — not contiguous
        >>> x.is_contiguous()
        False
        >>> y = x.contiguous()
        >>> y.is_contiguous()
        True

        Notes
        -----
        Idempotent: ``x.contiguous().contiguous()`` performs at most one
        copy. After the call ``is_contiguous() == True`` and strides match
        the C-order recurrence shown above.
        """
        return _wrap(_C_engine.contiguous(self._impl))  # type: ignore[return-value]

    def unfold(self, dimension: int, size: int, step: int) -> Tensor:
        r"""Return a view with an extra dimension containing sliding-window slices.

        Extracts non-overlapping or overlapping windows of length ``size``
        along ``dimension``, advancing by ``step`` elements between windows.
        The output has one extra trailing dimension compared to the input.

        Parameters
        ----------
        dimension : int
            The axis along which to slide the window.
        size : int
            Number of elements in each window.
        step : int
            Gap between the start positions of successive windows.

        Returns
        -------
        Tensor
            Tensor of shape
            :math:`(\ldots,\, L,\, \text{size})` where
            :math:`L = \lfloor (s_{\text{dim}} - \text{size}) /
            \text{step} \rfloor + 1`
            and :math:`s_{\text{dim}}` is the original size along
            ``dimension``.

        Notes
        -----
        Unfold is the fundamental primitive behind 1-D convolution and
        sliding-window aggregations.  The windows may overlap when
        ``step < size``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.arange(8, dtype=lucid.float32)
        >>> x.unfold(0, size=3, step=2).shape
        (3, 3)
        >>> # windows: [0,1,2], [2,3,4], [4,5,6]
        """
        return _wrap(_C_engine.unfold_dim(self._impl, dimension, size, step))  # type: ignore[return-value]

    def scatter_add(self, dim: int, index: Tensor, src: Tensor) -> Tensor:
        r"""Out-of-place scatter-add: add ``src`` values into ``self`` at positions given by ``index``.

        Returns a **new** tensor equal to ``self`` with ``src`` values added
        at the indices specified by ``index`` along ``dim``.  The original
        tensor is unchanged.

        Parameters
        ----------
        dim : int
            The axis along which to scatter.
        index : Tensor
            Integer tensor of indices, same shape as ``src``.  Each value
            selects a position along ``dim`` in the output.
        src : Tensor
            Values to scatter-add.  Must have the same shape as ``index``.

        Returns
        -------
        Tensor
            A new tensor of the same shape as ``self`` with the accumulated
            additions applied.

        Notes
        -----
        The output satisfies:

        .. math::

            \text{out}[\ldots, \text{index}[i,j,k], \ldots]
            \mathrel{+}= \text{src}[i, j, k]

        where the free indices iterate over all dimensions other than ``dim``.
        Multiple ``src`` entries may map to the same output position; their
        contributions are summed.

        Examples
        --------
        >>> import lucid
        >>> base = lucid.zeros(4)
        >>> idx  = lucid.tensor([0, 1, 1, 3])
        >>> src  = lucid.ones(4)
        >>> base.scatter_add(0, idx, src)
        tensor([1., 2., 0., 1.])
        """
        from lucid._ops import scatter_add as _sa

        return _sa(self, dim, index, src)  # type: ignore[return-value]

    @property
    def data(self) -> Self:
        r"""The tensor's underlying data, detached from gradient tracking.

        Returns a view of the same storage as ``self`` but with
        ``requires_grad=False`` and no ``grad_fn``.  Assigning to
        ``tensor.data`` replaces the underlying storage in-place without
        affecting the autograd graph — useful for in-place weight updates
        that should not be tracked.

        Returns
        -------
        Tensor
            A non-differentiable view of the same data.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
        >>> x.data.requires_grad
        False
        >>> x.data
        tensor([1., 2.])

        Notes
        -----
        Aliases the same storage as ``self`` with the gradient flag
        cleared. Mathematically the same identity as :meth:`detach` —
        equal values, zero Jacobian — but unlike :meth:`detach` writes
        through ``data`` propagate back to ``self``'s underlying buffer
        without participating in autograd. Prefer :meth:`detach` for new
        code; ``data`` is retained for API compatibility.
        """
        return Tensor.__new_from_impl__(_impl_with_grad(self._impl, False))  # type: ignore[return-value]

    # ── device/dtype conversion ───────────────────────────────────────────────
    # Injected by _tensor/_to.py after class definition.

    # ── shape helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        r"""Return the size of the first dimension (``shape[0]``).

        Implements the standard Python ``len()`` protocol. Mirrors
        sequence semantics: the length is the number of top-level elements
        (rows) the tensor iterates over.

        Returns
        -------
        int
            ``self.shape[0]`` — the size of the outermost dimension.

        Raises
        ------
        TypeError
            If the tensor is 0-d (scalar). Python forbids ``len()`` on
            sized-zero scalar objects.

        Examples
        --------
        >>> import lucid
        >>> len(lucid.zeros(5, 3))
        5
        >>> len(lucid.zeros(7))
        7

        Notes
        -----
        Mirrors NumPy/sequence semantics: :math:`\text{len}(t) = s_0`.
        Iterating with :meth:`__iter__` yields exactly :math:`s_0` slices.
        """
        if not self._impl.shape:
            raise TypeError("len() of a 0-d tensor")
        return self._impl.shape[0]

    def __bool__(self) -> bool:
        r"""Convert a single-element tensor to a Python ``bool``.

        Implements the truthiness protocol used by ``if t:``, ``while t:``,
        and ``bool(t)``. Only defined for single-element tensors because
        the truth value of a multi-element tensor is ambiguous: "is any
        element true?" (``.any()``) versus "are all elements true?"
        (``.all()``) are different reductions.

        Returns
        -------
        bool
            The unboxed value of the single element interpreted as a
            Python ``bool`` (non-zero numerics map to ``True``).

        Raises
        ------
        RuntimeError
            If ``numel() != 1``. Use :meth:`any` / :meth:`all` explicitly
            on multi-element tensors.

        Notes
        -----
        This matches the convention adopted by every mainstream tensor
        framework. The error guards against subtle bugs such as
        ``if pred_tensor:`` silently truncating to the first element.

        Examples
        --------
        >>> import lucid
        >>> bool(lucid.tensor(1.0))
        True
        >>> bool(lucid.tensor(0))
        False
        Mathematically: defined only when :math:`\prod_i s_i = 1`. The
        single element :math:`x` maps to :math:`\text{bool}(x) = (x \neq 0)`.

        >>> bool(lucid.tensor([1.0, 0.0]))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        RuntimeError: Boolean value of Tensor with more than one element is ambiguous
        """
        if self._impl.numel() != 1:
            raise RuntimeError(
                "Boolean value of Tensor with more than one element is ambiguous"
            )
        return bool(self.item())

    def __repr__(self) -> str:
        r"""Return a string representation suitable for debugging.

        Delegates to the formatted printer in :mod:`lucid._tensor._repr`,
        which renders shape, dtype, and (truncated) element values in a
        readable layout.

        Returns
        -------
        str
            Multi-line repr showing element values with appropriate
            precision, plus shape and dtype metadata when non-default.

        Examples
        --------
        >>> import lucid
        >>> repr(lucid.zeros(2, 3))  # doctest: +SKIP
        'tensor([[0., 0., 0.],\n        [0., 0., 0.]])'

        Notes
        -----
        Display only — does **not** participate in autograd and is one of
        the bridge boundaries permitted by rule **H4** (``_repr.py``).
        For Metal-resident tensors this forces a host synchronisation to
        materialise element values for printing.
        """
        return _tensor_repr(self)

    # hash: identity-based so tensors can be used as dict keys
    __hash__ = object.__hash__  # type: ignore[assignment]

    # ── iteration & formatting ────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Self]:
        r"""Iterate over the first dimension of the tensor.

        Yields successive slices ``self[i]`` for ``i`` in ``range(shape[0])``,
        so iterating an :math:`N`-d tensor produces :math:`(N-1)`-d slices.
        Implements the standard Python iterator protocol.

        Yields
        ------
        Tensor
            The ``i``-th slice along axis 0.

        Raises
        ------
        TypeError
            If the tensor is 0-d (scalar). Scalars have no first dimension
            to iterate over.

        Examples
        --------
        >>> import lucid
        >>> for row in lucid.arange(6).reshape(3, 2):
        ...     print(row.tolist())
        [0, 1]
        [2, 3]
        [4, 5]

        Notes
        -----
        Yields :math:`s_0` slices each of shape :math:`(s_1, \ldots, s_{d-1})`.
        For very large leading axes prefer chunked iteration via
        :func:`lucid.split` to amortise per-slice overhead.
        """
        if not self._impl.shape:
            raise TypeError("iteration over a 0-d tensor")
        for i in range(self._impl.shape[0]):
            yield self[i]  # type: ignore[misc]

    def __format__(self, format_spec: str) -> str:
        """Apply a Python format spec, supporting single-element tensors.

        For 1-element tensors, the spec is applied to the unboxed Python
        scalar via :meth:`item` — useful inside f-strings to control
        precision. For multi-element tensors the spec is ignored and the
        regular :meth:`__repr__` output is returned.

        Parameters
        ----------
        format_spec : str
            Standard Python format spec (e.g. ``".3f"``, ``"+.2e"``).

        Returns
        -------
        str
            Formatted scalar value or the full repr.

        Examples
        --------
        >>> import lucid
        >>> f"{lucid.tensor(3.14159):.2f}"
        '3.14'
        >>> f"{lucid.tensor(2.5):+.1e}"
        '+2.5e+00'

        Notes
        -----
        For single-element tensors the call is equivalent to
        ``format(t.item(), spec)``. For multi-element tensors the spec is
        ignored to avoid ambiguous element-wise formatting semantics.
        """
        if self._impl.numel() == 1:
            return format(self.item(), format_spec)
        return repr(self)

    # ── new_* convenience constructors ────────────────────────────────────────

    def new_empty(
        self,
        *size: int,
        dtype: _dtype_cls | None = None,
        device: _device_cls | str | None = None,
        requires_grad: bool = False,
    ) -> Self:
        r"""Return an uninitialized tensor of the given shape.

        The returned tensor inherits this tensor's ``dtype`` and ``device``
        unless overridden.  The contents are **undefined** — do not read
        values without first writing them.

        Parameters
        ----------
        *size : int
            Dimensions of the output tensor.
        dtype : lucid.dtype, optional
            Element type.  Defaults to ``self.dtype``.
        device : str or lucid.device, optional
            Target device.  Defaults to ``self.device``.
        requires_grad : bool, optional
            Enable gradient tracking on the result.  Default ``False``.

        Returns
        -------
        Tensor
            Uninitialized tensor of shape ``size``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(2, 3, dtype=lucid.float64)
        >>> y = x.new_empty(4, 5)
        >>> y.shape, y.dtype
        ((4, 5), lucid.float64)

        Notes
        -----
        Allocates :math:`\prod_i s_i \cdot e` bytes of uninitialised
        memory where :math:`e` is ``dtype.itemsize``. Faster than
        :meth:`new_zeros` because the engine skips the zero-fill kernel.
        Must be followed by a write before any read.
        """
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
        dtype: _dtype_cls | None = None,
        device: _device_cls | str | None = None,
        requires_grad: bool = False,
    ) -> Self:
        """Return a zero-filled tensor of the given shape.

        The returned tensor inherits this tensor's ``dtype`` and ``device``
        unless overridden.  All elements are initialised to ``0``.

        Parameters
        ----------
        *size : int
            Dimensions of the output tensor.
        dtype : lucid.dtype, optional
            Element type.  Defaults to ``self.dtype``.
        device : str or lucid.device, optional
            Target device.  Defaults to ``self.device``.
        requires_grad : bool, optional
            Enable gradient tracking on the result.  Default ``False``.

        Returns
        -------
        Tensor
            Zero tensor of shape ``size``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.ones(2, dtype=lucid.int32)
        >>> x.new_zeros(3, 3)
        tensor([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])

        Notes
        -----
        Initialises every element to the additive identity :math:`0`. The
        zero-fill is delegated to a fused engine kernel — Accelerate
        ``catlas_*set`` on CPU and MLX broadcast-fill on Metal.
        """
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
        dtype: _dtype_cls | None = None,
        device: _device_cls | str | None = None,
        requires_grad: bool = False,
    ) -> Self:
        """Return an all-ones tensor of the given shape.

        The returned tensor inherits this tensor's ``dtype`` and ``device``
        unless overridden.  All elements are initialised to ``1``.

        Parameters
        ----------
        *size : int
            Dimensions of the output tensor.
        dtype : lucid.dtype, optional
            Element type.  Defaults to ``self.dtype``.
        device : str or lucid.device, optional
            Target device.  Defaults to ``self.device``.
        requires_grad : bool, optional
            Enable gradient tracking on the result.  Default ``False``.

        Returns
        -------
        Tensor
            All-ones tensor of shape ``size``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(2, dtype=lucid.float16)
        >>> x.new_ones(2, 4)
        tensor([[1., 1., 1., 1.],
                [1., 1., 1., 1.]])

        Notes
        -----
        Initialises every element to the multiplicative identity :math:`1`.
        For arbitrary fill values use :meth:`new_full`.
        """
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
        dtype: _dtype_cls | None = None,
        device: _device_cls | str | None = None,
        requires_grad: bool = False,
    ) -> Self:
        r"""Return a tensor of the given shape filled with a constant value.

        The returned tensor inherits this tensor's ``dtype`` and ``device``
        unless overridden.  Every element is set to ``fill_value``.

        Parameters
        ----------
        size : tuple[int, ...]
            Shape of the output tensor.
        fill_value : float
            Scalar value to fill the tensor with.
        dtype : lucid.dtype, optional
            Element type.  Defaults to ``self.dtype``.
        device : str or lucid.device, optional
            Target device.  Defaults to ``self.device``.
        requires_grad : bool, optional
            Enable gradient tracking on the result.  Default ``False``.

        Returns
        -------
        Tensor
            Constant-filled tensor of shape ``size``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(1)
        >>> x.new_full((2, 3), fill_value=7.0)
        tensor([[7., 7., 7.],
                [7., 7., 7.]])

        Notes
        -----
        Constant tensor :math:`f \cdot \mathbf{1}` where :math:`f` is
        ``fill_value`` and :math:`\mathbf{1}` is the all-ones tensor of
        the given shape. ``fill_value`` is promoted to ``dtype`` before
        the broadcast fill.
        """
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
        dtype: _dtype_cls | None = None,
        device: _device_cls | str | None = None,
        requires_grad: bool = False,
    ) -> Self:
        """Return a new tensor constructed from ``data``, inheriting dtype/device.

        Creates a fresh tensor from the provided data, using this tensor's
        ``dtype`` and ``device`` as defaults.  The data is always **copied**;
        the result does not share storage with ``data`` even if ``data`` is
        already a :class:`Tensor`.

        Parameters
        ----------
        data : array_like or Tensor
            Input data — nested Python lists, scalars, or an existing tensor.
        dtype : lucid.dtype, optional
            Element type.  Defaults to ``self.dtype``.
        device : str or lucid.device, optional
            Target device.  Defaults to ``self.device``.
        requires_grad : bool, optional
            Enable gradient tracking on the result.  Default ``False``.

        Returns
        -------
        Tensor
            New tensor containing a copy of ``data``.

        Examples
        --------
        >>> import lucid
        >>> proto = lucid.zeros(1, dtype=lucid.float64)
        >>> t = proto.new_tensor([[1, 2], [3, 4]])
        >>> t.dtype
        lucid.float64

        Notes
        -----
        Always copies — never aliases ``data``'s storage even when ``data``
        is already a ``Tensor``. Routes through the same
        ``_to_impl`` bridge boundary as the public constructor (rule **H4**).
        """
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
        r"""Return the size in bytes of a single element.

        Determined entirely by :attr:`dtype`.  Common values:

        * ``float32`` / ``int32`` → 4 bytes
        * ``float64`` / ``int64`` → 8 bytes
        * ``float16`` / ``bfloat16`` / ``int16`` → 2 bytes
        * ``int8`` / ``bool_`` → 1 byte
        * ``complex64`` → 8 bytes (two 32-bit floats)

        Returns
        -------
        int
            Number of bytes per element.

        Notes
        -----
        The total memory footprint of the tensor is:

        .. math::

            \text{nbytes} = \text{numel()} \times \text{element\_size()}

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3, dtype=lucid.float32).element_size()
        4
        >>> lucid.zeros(3, dtype=lucid.float64).element_size()
        8
        """
        return self.dtype.itemsize

    @property
    def itemsize(self) -> int:
        r"""Bytes per element — alias for :meth:`element_size`.

        Provided as a property (rather than a method) for NumPy-style
        attribute access:  ``tensor.itemsize`` instead of
        ``tensor.element_size()``.

        Returns
        -------
        int
            Number of bytes per element.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(5, dtype=lucid.int16).itemsize
        2

        Notes
        -----
        Total footprint of the tensor satisfies
        :math:`\text{nbytes} = \text{numel} \cdot \text{itemsize}`.
        """
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        r"""Total number of bytes occupied by the tensor's data buffer.

        Equals ``numel() * element_size()``.  Useful for estimating memory
        usage and for setting buffer sizes when interoperating with C or
        Metal shaders.

        Returns
        -------
        int
            Total byte count of the data storage.

        Notes
        -----
        .. math::

            \text{nbytes} = \prod_{i} s_i \times e

        where :math:`s_i` are the dimension sizes and :math:`e` is the
        element size in bytes.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(4, 4, dtype=lucid.float32).nbytes
        64
        >>> lucid.zeros(4, 4, dtype=lucid.float64).nbytes
        128
        """
        return self._impl.numel() * self.dtype.itemsize

    def stride(self, dim: int | None = None) -> tuple[int, ...] | int:
        r"""Return the strides of the tensor in *element* counts.

        Parameters
        ----------
        dim : int, optional
            If given, return the stride along that dimension only.

        Returns
        -------
        tuple[int, ...] or int
            Element-count strides (same semantics as the reference framework).

        Notes
        -----
        For a C-contiguous tensor of shape :math:`(s_0, \ldots, s_{d-1})`
        the element strides satisfy the row-major recurrence

        .. math::

            \text{stride}[d-1] = 1, \quad
            \text{stride}[i] = \text{stride}[i+1] \cdot s_{i+1}.

        Non-contiguous tensors (e.g. transposed or sliced views) may have
        arbitrary strides; the address of element ``[i_0, \ldots, i_{d-1}]``
        is ``base + \sum_k i_k \cdot \text{stride}[k]``.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3, 4).stride()
        (4, 1)
        >>> lucid.zeros(3, 4).stride(0)
        4
        """
        byte_strides: list[int] = list(self._impl.stride)
        itemsz: int = self.dtype.itemsize
        elem_strides = tuple(s // itemsz for s in byte_strides)
        if dim is None:
            return elem_strides
        return elem_strides[dim]

    def data_ptr(self) -> int:
        r"""Return the address of the first element as an integer.

        On Apple Silicon the tensor lives in unified memory; this method
        returns a best-effort identifier derived from the storage object.
        Use :meth:`numpy` + ``ndarray.ctypes.data`` for interop that
        requires the actual pointer.

        Returns
        -------
        int
            A stable, process-unique integer suitable for aliasing checks
            (e.g. "do these two tensors share storage?"). Not guaranteed
            to be the raw memory address.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> y = x
        >>> x.data_ptr() == y.data_ptr()
        True
        >>> z = lucid.zeros(3)
        >>> x.data_ptr() != z.data_ptr()
        True

        Notes
        -----
        On Apple Silicon CPU and GPU share unified DRAM, so the same
        :math:`\text{data\_ptr}(t)` identifies a buffer addressable from
        both backends. The value is stable for the lifetime of ``self``;
        equality is the canonical aliasing predicate
        :math:`t_1 \sim t_2 \iff \text{data\_ptr}(t_1) = \text{data\_ptr}(t_2)`.
        """
        # id() of the impl object is a stable, process-unique identifier
        # suitable for equality checks (e.g. detecting aliasing) even if not
        # the raw memory address.
        return id(self._impl)

    def storage_offset(self) -> int:
        r"""Return the offset (in elements) of the first element in storage.

        Contiguous tensors always return ``0``. Non-contiguous view tensors
        may return a non-zero offset in frameworks that support strided
        views; Lucid currently represents all tensors as contiguous so
        this always returns ``0``.

        Returns
        -------
        int
            The element offset into the underlying storage where ``self``
            begins. Always ``0`` in the current Lucid implementation.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3, 4).storage_offset()
        0

        Notes
        -----
        In Lucid every tensor owns a fresh contiguous storage, so the
        offset is identically zero: :math:`\text{offset}(t) \equiv 0`.
        Frameworks that support sub-views over a shared buffer use this
        for pointer arithmetic; for Lucid it is provided purely for API
        compatibility.
        """
        return 0

    @property
    def H(self) -> Tensor:
        r"""Conjugate (Hermitian) transpose of the last two axes.

        For real-valued tensors this is identical to :attr:`mT`. For
        complex tensors the elements are conjugated before transposing,
        producing the Hermitian adjoint :math:`A^{\mathsf{H}}` familiar
        from linear algebra:

        .. math::

            (A^{\mathsf{H}})_{ij} = \overline{A_{ji}}

        Returns
        -------
        Tensor
            A tensor of shape :math:`(\ldots, n, m)` for an input of
            shape :math:`(\ldots, m, n)`. Real inputs round-trip through
            :attr:`mT`; complex inputs are first conjugated by
            ``lucid._ops.composite.conj``.

        Notes
        -----
        Equivalent to ``x.conj().mT`` for complex tensors. For batched
        linear algebra the leading dimensions are untouched.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.arange(6).reshape(2, 3)
        >>> x.H.shape
        (3, 2)
        """
        if self.is_complex():
            from lucid._ops.composite import conj as _conj

            return _conj(self).mT  # type: ignore[return-value]
        return self.mT  # type: ignore[return-value]

    def type(self, dtype: str | None = None) -> str | Tensor:
        """Return or cast the tensor type using legacy type strings.

        * ``t.type()`` — return a string like ``'lucid.FloatTensor'``.
        * ``t.type('lucid.DoubleTensor')`` — cast and return the new tensor.

        Supported type strings: ``FloatTensor``, ``DoubleTensor``,
        ``HalfTensor``, ``IntTensor``, ``LongTensor``, ``BoolTensor``,
        ``ShortTensor``, ``ByteTensor``.

        Parameters
        ----------
        dtype : str, optional
            Lucid legacy type string. If ``None``, returns the type string
            of ``self``; otherwise casts to the corresponding dtype.

        Returns
        -------
        str
            When ``dtype is None`` — the legacy type label.
        Tensor
            When a type string is provided — a new tensor cast to that
            dtype (current device retained).

        Raises
        ------
        TypeError
            If ``dtype`` is not one of the supported legacy strings.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3).type()
        'lucid.FloatTensor'
        >>> lucid.zeros(3).type('lucid.LongTensor').dtype
        lucid.int64

        Notes
        -----
        Provided purely for API compatibility with code that uses legacy
        type strings. New code should use :attr:`dtype` for inspection and
        :meth:`to` for casting — both avoid stringly-typed dispatch.
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
        r"""Return the device index.

        Returns ``0`` for Metal (GPU) tensors and ``-1`` for CPU tensors,
        following the convention adopted by the reference framework.

        Returns
        -------
        int
            ``0`` for Metal-resident tensors; ``-1`` for CPU-resident tensors.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3).get_device()
        -1
        >>> lucid.zeros(3).metal().get_device()  # doctest: +SKIP
        0

        Notes
        -----
        Indicator-style encoding:
        :math:`\text{idx}(t) = 0 \cdot \mathbf{1}\{\text{is\_metal}\} + (-1) \cdot \mathbf{1}\{\text{is\_cpu}\}`.
        Lucid only supports a single Metal device on Apple Silicon, so a
        positive index is always ``0``.
        """
        return 0 if self.is_metal else -1

    def pin_memory(self, device: object = None) -> Tensor:
        r"""Return ``self`` — pinned memory is a no-op on Apple Silicon.

        Apple Silicon uses unified memory: CPU and GPU share the same
        physical DRAM, so the "page-lock host memory to accelerate
        host-to-device DMA" concept from discrete-GPU frameworks does not
        apply. This method exists for API compatibility and simply
        returns the tensor unchanged.

        Parameters
        ----------
        device : object, optional
            Accepted but ignored. Present for signature compatibility.

        Returns
        -------
        Tensor
            ``self``, untouched.

        Notes
        -----
        See also :meth:`is_pinned` (always ``False``) and
        :meth:`share_memory_` (also a no-op). The function is the
        identity: :math:`\text{pin\_memory}(t) \equiv t`.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> x.pin_memory() is x
        True
        """
        return self  # type: ignore[return-value]

    def is_pinned(self, device: object = None) -> bool:
        r"""Return ``False`` — pinned memory is not applicable on Apple Silicon.

        Apple Silicon's unified-memory architecture makes the distinction
        between "pageable" and "page-locked" host memory irrelevant: CPU
        and GPU already see the same DRAM. This predicate is provided for
        API compatibility and always reports ``False``.

        Parameters
        ----------
        device : object, optional
            Accepted but ignored.

        Returns
        -------
        bool
            Always ``False``.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3).is_pinned()
        False

        Notes
        -----
        Identically false: :math:`\text{is\_pinned}(t) \equiv \text{False}`.
        Unified memory makes the concept moot — all host buffers are
        already DMA-accessible to the GPU without page-locking.
        """
        return False

    @property
    def is_cuda(self) -> bool:
        r"""Return ``False`` — Lucid does not target NVIDIA GPUs.

        Lucid is Apple-Silicon-exclusive: the GPU stream is MLX-on-Metal,
        not NVIDIA's discrete GPU stack. Use :attr:`is_metal` to detect
        GPU-resident tensors. This property exists purely for API
        compatibility with code paths that probe for the legacy attribute.

        Returns
        -------
        bool
            Always ``False``.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3).is_cuda
        False

        Notes
        -----
        Identically false: :math:`\text{is\_cuda}(t) \equiv \text{False}`.
        Use :attr:`is_metal` to query GPU residency on Apple Silicon.
        """
        return False

    def reshape_as(self, other: Tensor) -> Tensor:
        r"""Return a tensor with the same data reshaped to ``other.shape``.

        Convenience wrapper around :func:`reshape` that takes the target
        shape from another tensor instead of as a tuple. Element count
        must agree:

        .. math::

            \prod_i \text{self.shape}[i] = \prod_j \text{other.shape}[j]

        Parameters
        ----------
        other : Tensor
            Tensor whose shape will be adopted. Only ``other.shape`` is
            consulted; the values and dtype of ``other`` are ignored.

        Returns
        -------
        Tensor
            A view (or copy if storage layout requires) of ``self`` with
            shape ``other.shape``.

        Raises
        ------
        RuntimeError
            If ``self.numel() != other.numel()``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.arange(12)
        >>> proto = lucid.zeros(3, 4)
        >>> x.reshape_as(proto).shape
        (3, 4)

        Notes
        -----
        Reshape preserves element order under row-major (C) traversal;
        the element at flat index :math:`k = \sum_i i_k \cdot \prod_{j>k} s_j`
        in ``self`` becomes the element at the same flat index in the
        result. A view is returned when the source is contiguous;
        otherwise a contiguous copy is materialised first.
        """
        return Tensor.__new_from_impl__(  # type: ignore[return-value]
            _C_engine.reshape(self._impl, list(other._impl.shape))
        )

    class _UntypedStorage:
        """Minimal storage object returned by :meth:`Tensor.untyped_storage`."""

        def __init__(self, tensor: Tensor) -> None:
            """Store the owning ``Tensor`` so storage queries can forward to it."""
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
            """Return the storage size in bytes (so ``len(storage) == storage.nbytes``)."""
            return self.size()

        def __repr__(self) -> str:
            return (
                f"UntypedStorage(nbytes={self.size()}, "
                f"device={self._tensor.device})"
            )

    def untyped_storage(self) -> _UntypedStorage:
        r"""Return a minimal storage view of the underlying data buffer.

        The returned object exposes :meth:`data_ptr`, :meth:`size`,
        and :meth:`nbytes` — the subset needed for common introspection
        patterns (aliasing checks, memory-footprint accounting, debug
        printing).

        Returns
        -------
        Tensor._UntypedStorage
            Lightweight storage handle whose ``__len__`` and ``size``
            report the byte count of the buffer.

        Notes
        -----
        Lucid does not currently expose a fully-featured ``Storage`` type;
        ``untyped_storage`` is the introspection-only minimum. Mutating
        through this handle is not supported. The reported size satisfies
        :math:`|\text{storage}| = \text{numel} \cdot \text{itemsize}`.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(4, dtype=lucid.float32)
        >>> s = x.untyped_storage()
        >>> s.size()
        16
        >>> len(s)
        16
        """
        return Tensor._UntypedStorage(self)

    def is_floating_point(self) -> bool:
        r"""Return ``True`` if the dtype is a floating-point type.

        Floating-point dtypes recognised by Lucid are ``float16``,
        ``float32``, ``float64``, and ``bfloat16``.

        Returns
        -------
        bool
            ``True`` for the four floating-point dtypes above; ``False``
            for integer, boolean, and complex dtypes.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3).is_floating_point()
        True
        >>> lucid.zeros(3, dtype=lucid.int64).is_floating_point()
        False

        Notes
        -----
        Set-membership predicate
        :math:`\text{is\_float}(t) = (t.\text{dtype} \in \mathcal{F})`
        with :math:`\mathcal{F} = \{\text{float16}, \text{float32}, \text{float64}, \text{bfloat16}\}`.
        Many differentiable ops are only defined on :math:`\mathcal{F}`;
        integer and boolean dtypes block autograd at the operator level.
        """
        return self.dtype in (float16, float32, float64, bfloat16)

    def is_complex(self) -> bool:
        r"""Return ``True`` if the dtype is a complex type.

        Currently Lucid supports a single complex dtype, ``complex64``
        (two 32-bit floats per element). Future complex dtypes will also
        be reported here.

        Returns
        -------
        bool
            ``True`` if ``self.dtype is lucid.complex64``; ``False``
            otherwise.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3, dtype=lucid.complex64).is_complex()
        True
        >>> lucid.zeros(3).is_complex()
        False

        Notes
        -----
        Complex tensors store interleaved real/imag pairs:
        :math:`z_i = a_i + b_i \mathrm{i}` with :math:`a_i, b_i \in \mathbb{R}`.
        Lucid currently supports a single complex dtype (``complex64``),
        backed by two 32-bit floats per element.
        """
        return self.dtype is complex64

    def share_memory_(self) -> Self:
        r"""Mark storage as shareable across processes — a no-op on Apple Silicon.

        On platforms with separate CPU and GPU address spaces this method
        moves the storage into a shared-memory segment so that worker
        processes (e.g. DataLoader workers) can read it without copying.
        Apple Silicon's unified-memory architecture makes this unnecessary:
        all tensors are already addressable from every process that holds
        a reference to the buffer. Returns ``self`` for chaining.

        Returns
        -------
        Tensor
            ``self``, unchanged.

        Notes
        -----
        Provided for API compatibility. See also :meth:`is_pinned` and
        :meth:`pin_memory`, which are no-ops for the same reason. The
        operation is the identity:
        :math:`\text{share\_memory\_}(t) \equiv t`.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> x.share_memory_() is x
        True
        """
        return self

    # ── Phase N convenience methods ───────────────────────────────────────────

    def fill_(self, value: float) -> Self:
        r"""Fill the tensor with a scalar value in-place.

        Mutates the tensor's storage so every element becomes ``value``.
        Implemented by materialising a constant tensor with
        ``_C_engine.full`` and copying it into ``self``'s storage; the
        original ``TensorImpl`` (and thus identity) is preserved.

        Parameters
        ----------
        value : float
            Scalar value to broadcast into every position. Promoted to the
            tensor's dtype on copy.

        Returns
        -------
        Tensor
            ``self`` (in-place); useful for method chaining.

        Notes
        -----
        In-place operations bypass autograd's view tracking for
        performance. Calling ``fill_`` on a leaf tensor with
        ``requires_grad=True`` may raise a runtime error from the
        autograd engine.

        Mathematically the result is the constant tensor
        :math:`v \cdot \mathbf{1}` with the same shape as ``self``;
        every entry satisfies :math:`x_i \leftarrow v`.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.empty(3)
        >>> _ = x.fill_(0.5)
        >>> x.tolist()
        [0.5, 0.5, 0.5]
        """
        filled = _C_engine.full(
            list(self._impl.shape), value, self._impl.dtype, self._impl.device
        )
        self._impl.copy_from(filled)
        return self

    def copy_(self, other: Self) -> Self:
        r"""Copy data from ``other`` into ``self`` in-place.

        Overwrites ``self``'s storage with ``other``'s values. Broadcasting
        is permitted: ``other`` may have a shape that broadcasts to
        ``self.shape``. The dtype of ``self`` is preserved; ``other``
        values are cast as necessary.

        Parameters
        ----------
        other : Tensor
            Source tensor. Made contiguous before the copy to guarantee
            a stride-compatible memory layout.

        Returns
        -------
        Tensor
            ``self`` after the copy.

        Notes
        -----
        Unlike :meth:`clone`, ``copy_`` does not allocate a new tensor —
        only ``self``'s storage is written. The autograd graph is not
        extended by this operation. Element-wise:

        .. math::

            \text{self}_i \leftarrow \text{cast}_{\text{self.dtype}}(\text{other}_i),

        with standard right-aligned broadcasting from ``other.shape`` to
        ``self.shape``.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.zeros(3)
        >>> b = lucid.tensor([1.0, 2.0, 3.0])
        >>> _ = a.copy_(b)
        >>> a.tolist()
        [1.0, 2.0, 3.0]
        """
        src = _C_engine.contiguous(other._impl)
        self._impl.copy_from(src)
        return self

    # ``flip`` / ``fliplr`` / ``flipud`` are auto-injected from the registry
    # (see ``_ops/_registry.py``); the previous explicit definitions
    # duplicated that path.

    def index_select(self, dim: int, index: Self) -> Self:
        r"""Gather values along ``dim`` using an integer index tensor.

        For each position ``i`` along ``dim``, selects ``self[..., index[i], ...]``
        and concatenates along the same axis. The output has the same
        number of dimensions as ``self``; only the size along ``dim``
        changes to ``len(index)``.

        Parameters
        ----------
        dim : int
            Axis along which to select.
        index : Tensor
            1-D integer tensor of indices into ``self`` along ``dim``.
            Values must lie in ``[0, self.shape[dim])``.

        Returns
        -------
        Tensor
            Tensor with shape equal to ``self.shape`` except that
            ``shape[dim] == len(index)``.

        Notes
        -----
        Equivalent to advanced integer indexing:

        .. math::

            \text{out}[\ldots, i, \ldots] = \text{self}[\ldots, \text{index}[i], \ldots]

        Implemented via engine-level ``gather`` after broadcasting the
        index tensor over the non-selected axes.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.arange(12).reshape(3, 4)
        >>> idx = lucid.tensor([0, 2], dtype=lucid.int64)
        >>> x.index_select(0, idx).shape
        (2, 4)
        """
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
        r"""Flatten and select elements where ``mask`` is ``True``.

        Returns a 1-D tensor containing the values of ``self`` at positions
        where ``mask`` is truthy, in row-major (C) order. The output
        length is data-dependent — equal to ``mask.sum().item()`` — so
        this op forces a host-side synchronisation on Metal tensors.

        Parameters
        ----------
        mask : Tensor
            Boolean tensor, broadcastable to ``self.shape``. Non-bool
            dtypes are treated as truthy/falsy element-wise.

        Returns
        -------
        Tensor
            1-D tensor of selected elements; dtype matches ``self``.

        Notes
        -----
        Because the output shape depends on the mask's runtime values,
        this is one of the "data-dependent output" carve-outs that may
        round-trip through CPU on Metal devices.

        Defines

        .. math::

            \text{out} = (x_i \,:\, i \in \{i \mid \text{mask}_i\})

        in row-major scan order; the output length equals
        :math:`\sum_i \mathbf{1}\{\text{mask}_i\}`.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        >>> mask = lucid.tensor([True, False, True, False])
        >>> x.masked_select(mask).tolist()
        [1.0, 3.0]
        """
        return Tensor.__new_from_impl__(  # type: ignore[return-value]
            _C_engine.masked_select(self._impl, mask._impl)
        )

    def expand_as(self, other: Self) -> Self:
        r"""Broadcast ``self`` to match ``other.shape`` without copying data.

        Convenience wrapper around ``broadcast_to`` that takes the target
        shape from another tensor. The expansion is **view-like**: the
        underlying storage is not duplicated, and the broadcast dimensions
        are realised by zero-stride entries in the resulting tensor.

        Parameters
        ----------
        other : Tensor
            Tensor whose shape will be adopted. Only ``other.shape`` is
            consulted.

        Returns
        -------
        Tensor
            A view of ``self`` with shape ``other.shape``.

        Notes
        -----
        Broadcasting rules follow the standard right-aligned semantics:
        each dimension of ``self.shape`` must either equal the
        corresponding entry of ``other.shape`` or be ``1``. Size-1 axes
        are stretched by setting the corresponding stride to zero, so the
        resulting view aliases the source storage. Formally, for each axis

        .. math::

            s'_i = \begin{cases}
                s_i & \text{if } s_i = t_i \\
                t_i & \text{if } s_i = 1
            \end{cases}

        with stride :math:`0` on stretched axes.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0]).reshape(1, 3)
        >>> proto = lucid.zeros(4, 3)
        >>> x.expand_as(proto).shape
        (4, 3)
        """
        return Tensor.__new_from_impl__(  # type: ignore[return-value]
            _C_engine.broadcast_to(self._impl, list(other._impl.shape))
        )

    def view_as(self, other: Self) -> Self:
        r"""Reinterpret strides to match ``other.shape``.

        Convenience wrapper around :func:`reshape` that adopts the shape
        of another tensor. When the source is contiguous the result is a
        true view (zero copy); otherwise the engine may have to
        materialise a contiguous copy first.

        Parameters
        ----------
        other : Tensor
            Tensor whose shape will be adopted. Only ``other.shape`` is
            consulted.

        Returns
        -------
        Tensor
            A reshaped view (or copy) of ``self`` with shape ``other.shape``.

        Raises
        ------
        RuntimeError
            If ``self.numel() != other.numel()``.

        Notes
        -----
        For a contiguous tensor the new strides are computed as

        .. math::

            \text{stride}[i] = \prod_{j > i} \text{shape}[j]

        so the layout is row-major with no data motion.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.arange(6)
        >>> proto = lucid.zeros(2, 3)
        >>> x.view_as(proto).shape
        (2, 3)
        """
        return Tensor.__new_from_impl__(  # type: ignore[return-value]
            _C_engine.reshape(self._impl, list(other._impl.shape))
        )

    def type_as(self, other: Self) -> Self:
        r"""Cast ``self`` to the dtype of ``other``.

        Convenience wrapper around :meth:`to` that adopts the dtype of
        another tensor. Useful when two tensors must have matching
        precision before a fused op.

        Parameters
        ----------
        other : Tensor
            Tensor whose ``dtype`` will be adopted.

        Returns
        -------
        Tensor
            ``self`` cast to ``other.dtype``. If the dtype already
            matches, the call is a no-op.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3, dtype=lucid.int32)
        >>> y = lucid.zeros(1, dtype=lucid.float64)
        >>> x.type_as(y).dtype
        lucid.float64

        Notes
        -----
        Values are cast element-wise:
        :math:`x_i \leftarrow \text{cast}_{\tau}(x_i)` where
        :math:`\tau = \text{other.dtype}`. Casts between floating-point
        types preserve gradients; integer→float→integer round-trips lose
        information in the integer truncation step.
        """
        return self.to(other.dtype)

    def lerp(self, end: Self, weight: float | Self) -> Self:
        r"""Linearly interpolate between ``self`` and ``end``.

        Computes

        .. math::

            \text{out} = \text{self} + \text{weight} \times (\text{end} - \text{self})

        element-wise. ``weight`` may be a Python scalar or a tensor
        broadcastable to ``end - self``; with ``weight = 0`` the result
        equals ``self``, with ``weight = 1`` it equals ``end``.

        Parameters
        ----------
        end : Tensor
            Target tensor. Must broadcast against ``self``.
        weight : float or Tensor
            Interpolation coefficient. Float values are broadcast to a
            constant tensor matching the difference's shape; tensor
            values are used as-is.

        Returns
        -------
        Tensor
            Interpolated tensor with the broadcast shape of
            ``self``, ``end``, and ``weight``.

        Notes
        -----
        Linear interpolation is affine: ``lerp(a, b, t)`` lies on the
        line segment from ``a`` to ``b``. The formula is numerically
        preferable to ``(1 - weight) * self + weight * end`` because it
        avoids loss of precision at small ``weight`` values.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([0.0, 0.0, 0.0])
        >>> b = lucid.tensor([10.0, 20.0, 30.0])
        >>> a.lerp(b, 0.5).tolist()
        [5.0, 10.0, 15.0]
        """
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
        r"""Element-wise piecewise selection between ``self`` and ``other``.

        For each position, returns the value from ``self`` if the
        corresponding entry in ``condition`` is truthy, otherwise the
        value from ``other``:

        .. math::

            \text{out}_i = \begin{cases}
                \text{self}_i  & \text{if } \text{condition}_i \\
                \text{other}_i & \text{otherwise}
            \end{cases}

        Parameters
        ----------
        condition : Tensor
            Boolean (or truthy-valued) mask, broadcastable to ``self``.
        other : Tensor or float
            Fall-back values when ``condition`` is falsy. Scalars are
            broadcast to a constant tensor matching ``self.shape``.

        Returns
        -------
        Tensor
            Tensor with the broadcast shape of the three inputs, dtype
            matching ``self``.

        Notes
        -----
        Differentiable in both branches: gradients flow into ``self``
        where ``condition`` is true and into ``other`` where it is false.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        >>> mask = lucid.tensor([True, False, True, False])
        >>> x.where(mask, 0.0).tolist()
        [1.0, 0.0, 3.0, 0.0]
        """
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
        r"""Compute the ``n``-th order discrete difference along ``dim``.

        The first-order difference operator :math:`\Delta` is defined as

        .. math::

            (\Delta x)_i = x_{i+1} - x_i,

        which reduces the length of the chosen axis by 1. Higher orders
        are obtained by composition: :math:`\Delta^n = \Delta \circ \Delta^{n-1}`,
        equivalent to

        .. math::

            (\Delta^n x)_i = \sum_{k=0}^{n} (-1)^k \binom{n}{k} x_{i+n-k}.

        Parameters
        ----------
        n : int, optional
            Order of the difference operator. Default ``1``. Must be
            non-negative; ``n = 0`` returns ``self`` unchanged.
        dim : int, optional
            Axis along which to compute the difference. Negative values
            count from the end. Default ``-1`` (last axis).

        Returns
        -------
        Tensor
            Tensor whose size along ``dim`` is ``self.shape[dim] - n`` and
            whose other axes match ``self``.

        Notes
        -----
        Useful as a discrete analogue of differentiation. For sample
        spacing :math:`h`, finite-difference derivatives are obtained by
        dividing by :math:`h^n`.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 4.0, 7.0, 11.0])
        >>> x.diff().tolist()
        [1.0, 2.0, 3.0, 4.0]
        >>> x.diff(n=2).tolist()
        [1.0, 1.0, 1.0]
        """
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
        r"""Fused matrix multiply-add.

        Computes

        .. math::

            \text{out} = \beta \cdot \text{self} + \alpha \cdot (\text{mat1} \cdot \text{mat2})

        where :math:`\cdot` is matrix multiplication. ``mat1`` must have
        shape :math:`(n, k)` and ``mat2`` shape :math:`(k, p)`; the
        product has shape :math:`(n, p)`. ``self`` must broadcast to
        :math:`(n, p)`.

        Parameters
        ----------
        mat1 : Tensor
            Left matrix of shape :math:`(n, k)`.
        mat2 : Tensor
            Right matrix of shape :math:`(k, p)`.
        beta : float, optional
            Scaling factor applied to ``self``. Default ``1.0``.
            ``beta = 0`` zeroes the additive term.
        alpha : float, optional
            Scaling factor applied to the matrix product. Default ``1.0``.

        Returns
        -------
        Tensor
            Resulting tensor of shape :math:`(n, p)`.

        Notes
        -----
        This is the same operation as the BLAS ``GEMM`` routine
        :math:`C \leftarrow \beta C + \alpha A B`. In Lucid it is
        currently decomposed as ``add(beta * self, alpha * (mat1 @ mat2))``;
        a future fused-kernel path may exploit the Accelerate / MLX
        GEMM-with-bias primitives directly.

        Examples
        --------
        >>> import lucid
        >>> bias = lucid.zeros(2, 2)
        >>> a = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> b = lucid.tensor([[1.0, 0.0], [0.0, 1.0]])
        >>> bias.addmm(a, b, alpha=2.0).tolist()
        [[2.0, 4.0], [6.0, 8.0]]
        """
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
        r"""Batched matrix multiplication.

        Performs a batch of independent matrix multiplications. ``self``
        must have shape :math:`(B, n, m)` and ``mat2`` shape
        :math:`(B, m, p)`; the result has shape :math:`(B, n, p)` and is
        computed slice-wise:

        .. math::

            \text{out}[b] = \text{self}[b] \cdot \text{mat2}[b]
                \quad \text{for } b \in \{0, \ldots, B-1\}.

        Parameters
        ----------
        mat2 : Tensor
            Right-hand batched matrix of shape :math:`(B, m, p)`. The
            batch dimension and the contracted dimension :math:`m` must
            match ``self``.

        Returns
        -------
        Tensor
            Tensor of shape :math:`(B, n, p)`.

        Notes
        -----
        Unlike :meth:`matmul`, ``bmm`` does **not** broadcast the batch
        dimension and requires exactly 3-D inputs. For broadcasted and
        higher-rank batched matmul use :meth:`matmul` (``@``).

        Examples
        --------
        >>> import lucid
        >>> a = lucid.ones(8, 3, 4)
        >>> b = lucid.ones(8, 4, 5)
        >>> a.bmm(b).shape
        (8, 3, 5)
        """
        return Tensor.__new_from_impl__(_C_engine.matmul(self._impl, mat2._impl))  # type: ignore[return-value]

    # ── zero_() helper ───────────────────────────────────────────────────────

    def zero_(self) -> Self:
        r"""Fill the tensor with zeros in-place.

        Mutates ``self``'s storage so every element becomes ``0``,
        without allocating a new tensor. Equivalent to ``self.fill_(0.0)``
        but routed through a multiply-by-zero kernel for clarity.

        Returns
        -------
        Tensor
            ``self``, after zeroing.

        Notes
        -----
        As with :meth:`fill_`, in-place mutation bypasses autograd's
        view tracking. Calling ``zero_`` on a leaf tensor with
        ``requires_grad=True`` may raise a runtime error from the
        autograd engine.

        Element-wise: :math:`x_i \leftarrow 0` for every position. The
        result is the additive identity of the tensor algebra at the
        same shape and dtype.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.ones(3)
        >>> _ = x.zero_()
        >>> x.tolist()
        [0.0, 0.0, 0.0]
        """
        result = _C_engine.mul_(
            self._impl,
            _C_engine.zeros(self._impl.shape, self._impl.dtype, self._impl.device),
        )
        self._impl = result
        return self

    # ── pickling support (required for multiprocessing DataLoader) ────────────

    def __reduce__(self) -> tuple:
        r"""Pickle hook for cross-process Tensor serialisation.

        Implements the Python pickle protocol (PEP 307). The returned
        ``(callable, args)`` pair is invoked by ``pickle.loads`` /
        ``copy.deepcopy`` to reconstruct the tensor; the unpickler
        receives raw element bytes plus enough metadata to rebuild the
        ``TensorImpl`` without going through NumPy.

        Returns
        -------
        tuple
            ``(_tensor_unpickle, (raw_bytes, shape, dtype_name, device,
            requires_grad))``. The first element is a module-level
            callable so that ``pickle`` can locate it by qualified name;
            the second element is the positional argument tuple.

        Notes
        -----
        The wire format mirrors the ``lucid.serialization`` v3 contract,
        so DataLoader workers spawned via ``multiprocessing`` (which
        defaults to the ``spawn`` start method on macOS) can transfer
        tensors without importing NumPy. The byte buffer is produced by
        ``TensorImpl::to_bytes`` and consumed by ``TensorImpl::from_bytes``.

        The pickle protocol requires the callable to be importable by
        fully qualified name — that is why :func:`_tensor_unpickle` is a
        module-level function rather than a static method. Logically
        :math:`\text{loads}(\text{dumps}(t)) = t` on values; autograd
        history is **not** preserved across the boundary.

        Examples
        --------
        >>> import pickle, lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0])
        >>> y = pickle.loads(pickle.dumps(x))
        >>> y.tolist()
        [1.0, 2.0, 3.0]
        """
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
