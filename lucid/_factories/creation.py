r"""
Tensor creation functions: zeros, ones, empty, full, eye, arange, linspace, *_like.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import normalize_factory_kwargs, _unwrap, _wrap, _impl_with_grad
from lucid._types import DeviceLike, DTypeLike

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _size_to_list(*size: int | tuple[int, ...]) -> list[int]:
    r"""Normalize size args: zeros(2,3) or zeros((2,3)) → [2, 3]."""
    if len(size) == 1 and isinstance(size[0], (list, tuple)):
        return list(size[0])
    return list(size)  # type: ignore[arg-type]


def zeros(
    *size: int | tuple[int, ...],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Return a tensor filled with the additive identity element, zero.

    Allocates a new tensor of the requested shape and fills every element
    with the scalar value $0$.  The result is the unique tensor satisfying

    .. math::

        Z_{i_1, i_2, \ldots, i_n} = 0
        \quad \forall\; (i_1, \ldots, i_n) \in \prod_k [0, s_k)

    where $(s_1, \ldots, s_n)$ is the requested shape.

    Zero-initialisation is the standard starting point for accumulation
    buffers (gradient accumulators, running statistics in batch norm,
    confusion matrices) and for masking computations where a neutral
    additive element is required.

    Parameters
    ----------
    *size : int or tuple[int, ...]
        Shape of the output tensor.  Can be passed as separate positional
        integers ``zeros(2, 3)`` or as a single tuple ``zeros((2, 3))``.
    dtype : lucid.dtype, optional
        Scalar data type of the output.  Defaults to the global default
        dtype (``lucid.float32`` unless changed with
        ``lucid.set_default_dtype``).
    device : str or lucid.device, optional
        Target device — ``"cpu"`` (Apple Accelerate) or ``"metal"``
        (Apple Metal GPU).  Defaults to the global default device.
    requires_grad : bool, optional
        If ``True``, operations on the returned tensor are recorded by the
        autograd engine.  Default: ``False``.

    Returns
    -------
    Tensor
        Tensor of shape ``size`` filled with zeros.

    Notes
    -----
    On Metal, the allocation and fill are performed on-device without a
    host round-trip.  Gradient of ``zeros`` with respect to any upstream
    variable is always zero by linearity, so it is never placed on the
    computation graph as a leaf that affects gradients — but downstream
    operations on it *are* tracked when ``requires_grad=True``.

    Examples
    --------
    >>> import lucid
    >>> lucid.zeros(3).tolist()
    [0.0, 0.0, 0.0]

    >>> lucid.zeros(2, 3).shape
    (2, 3)

    >>> lucid.zeros((4,), dtype=lucid.int32).dtype
    lucid.int32

    Accumulate squared errors manually:

    >>> acc = lucid.zeros(1)
    >>> for x in [1.0, -2.0, 3.0]:
    ...     acc = acc + lucid.tensor(x) ** 2
    """
    _dt, _dev, _rg = normalize_factory_kwargs(dtype, device, requires_grad)
    shape = _size_to_list(*size)
    impl = _C_engine.zeros(shape, _dt, _dev)
    return _wrap(_impl_with_grad(impl, _rg) if _rg else impl)


def ones(
    *size: int | tuple[int, ...],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Return a tensor filled with the multiplicative identity element, one.

    Allocates a new tensor of the requested shape and fills every element
    with the scalar value $1$.  Every entry satisfies

    .. math::

        O_{i_1, \ldots, i_n} = 1
        \quad \forall\; (i_1, \ldots, i_n) \in \prod_k [0, s_k)

    One-initialisation is the canonical starting point for multiplicative
    accumulators, scale parameters in normalisation layers (before any
    training), and homogeneous coordinate vectors in projective geometry.

    Parameters
    ----------
    *size : int or tuple[int, ...]
        Shape of the output tensor.  Accepts separate ints
        ``ones(2, 3)`` or a single tuple ``ones((2, 3))``.
    dtype : lucid.dtype, optional
        Scalar data type.  Defaults to the global default dtype
        (``lucid.float32`` unless overridden).
    device : str or lucid.device, optional
        Target device — ``"cpu"`` or ``"metal"``.
    requires_grad : bool, optional
        If ``True``, downstream operations are tracked by autograd.
        Default: ``False``.

    Returns
    -------
    Tensor
        Tensor of shape ``size`` filled with ones.

    Notes
    -----
    The all-ones vector $\mathbf{1} \in \mathbb{R}^n$ plays an important
    role in matrix algebra: multiplying a matrix $A$ on the right by
    $\mathbf{1}$ computes its row sums, $A\mathbf{1} = \text{rowsum}(A)$.
    This is useful when implementing attention masks, normalisation
    denominators, and indicator functions.

    Examples
    --------
    >>> import lucid
    >>> lucid.ones(4).tolist()
    [1.0, 1.0, 1.0, 1.0]

    >>> lucid.ones(2, 2, dtype=lucid.int8)
    Tensor([[1, 1],
            [1, 1]], dtype=int8)

    Row-sum via matrix-vector product:

    >>> A = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> row_sums = A @ lucid.ones(2)
    >>> row_sums.tolist()
    [3.0, 7.0]
    """
    _dt, _dev, _rg = normalize_factory_kwargs(dtype, device, requires_grad)
    shape = _size_to_list(*size)
    impl = _C_engine.ones(shape, _dt, _dev)
    return _wrap(_impl_with_grad(impl, _rg) if _rg else impl)


def empty(
    *size: int | tuple[int, ...],
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Return a tensor whose elements are uninitialised (undefined memory).

    Allocates the backing memory for the requested shape but performs **no
    initialisation write**.  Reading any element before overwriting it is
    undefined behaviour — values may be zero, garbage, or a previous
    tensor's data depending on the allocator's state.

    The key advantage over `zeros` or `ones` is that the initialisation
    kernel is skipped entirely, which is measurable when allocating large
    temporary buffers inside hot loops.  The typical pattern is:

    .. math::

        \text{allocate} \to \text{fill via in-place op} \to \text{read}

    Parameters
    ----------
    *size : int or tuple[int, ...]
        Shape of the output tensor.  Accepts separate ints
        ``empty(2, 3)`` or a single tuple ``empty((2, 3))``.
    dtype : lucid.dtype, optional
        Scalar data type.  Defaults to the global default dtype.
    device : str or lucid.device, optional
        Target device — ``"cpu"`` or ``"metal"``.
    requires_grad : bool, optional
        If ``True``, downstream operations are tracked by autograd.
        Default: ``False``.

    Returns
    -------
    Tensor
        Uninitialised tensor of shape ``size``.

    Notes
    -----
    Never use ``empty`` as input to a computation that reads its values
    without first writing them.  Common safe patterns:

    - ``out = lucid.empty(n); lucid.matmul(a, b, out=out)``
    - Passing as a pre-allocated output buffer to in-place operations

    On Metal the memory is allocated in the device heap; the allocator
    may reuse pages from recently freed tensors, so values are truly
    unpredictable.

    Examples
    --------
    >>> import lucid
    >>> t = lucid.empty(3, 4)
    >>> t.shape
    (3, 4)

    Safe usage — always overwrite before reading:

    >>> buf = lucid.empty(5)
    >>> buf[:] = lucid.arange(5).astype(lucid.float32)
    """
    _dt, _dev, _rg = normalize_factory_kwargs(dtype, device, requires_grad)
    shape = _size_to_list(*size)
    impl = _C_engine.empty(shape, _dt, _dev)
    return _wrap(_impl_with_grad(impl, _rg) if _rg else impl)


def full(
    size: int | list[int] | tuple[int, ...],
    fill_value: float,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Return a tensor filled with a constant scalar value.

    Every element of the output equals ``fill_value``:

    .. math::

        F_{i_1, \ldots, i_n} = c
        \quad \forall\; (i_1, \ldots, i_n) \in \prod_k [0, s_k)

    where $c$ is ``fill_value``.  This is a generalisation of `zeros`
    ($c = 0$) and `ones` ($c = 1$) to an arbitrary constant.

    Parameters
    ----------
    size : int or list[int] or tuple[int, ...]
        Shape of the output tensor.  Unlike the ``*size`` varargs of
        `zeros` / `ones`, this is a single positional argument, so
        multi-dimensional shapes must be passed as a list or tuple:
        ``full((2, 3), 7.0)``.
    fill_value : float
        The scalar constant to broadcast across all elements.
    dtype : lucid.dtype, optional
        Scalar data type.  If ``None``, inferred from ``fill_value``
        (integers → ``int64``, floats → the global default float dtype).
    device : str or lucid.device, optional
        Target device — ``"cpu"`` or ``"metal"``.
    requires_grad : bool, optional
        If ``True``, downstream operations are tracked by autograd.
        Default: ``False``.

    Returns
    -------
    Tensor
        Constant tensor of shape ``size`` filled with ``fill_value``.

    Examples
    --------
    >>> import lucid
    >>> lucid.full((2, 3), 3.14).shape
    (2, 3)

    >>> lucid.full(4, -1.0).tolist()
    [-1.0, -1.0, -1.0, -1.0]

    Mask of a fixed value (e.g. $-\infty$ for attention masking):

    >>> mask = lucid.full((4, 4), float("-inf"))
    """
    _dt, _dev, _rg = normalize_factory_kwargs(dtype, device, requires_grad)
    shape = list(size) if isinstance(size, (list, tuple)) else [size]
    impl = _C_engine.full(shape, fill_value, _dt, _dev)
    return _wrap(_impl_with_grad(impl, _rg) if _rg else impl)


def eye(
    n: int,
    m: int | None = None,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Return a 2-D matrix with ones on the main diagonal and zeros elsewhere.

    The returned matrix $E \in \mathbb{R}^{n \times m}$ is defined by the
    Kronecker delta:

    .. math::

        E_{ij} = \delta_{ij} =
        \begin{cases}
            1 & \text{if } i = j \\
            0 & \text{if } i \neq j
        \end{cases}

    When $n = m$ this is the identity matrix $I_n$, which satisfies
    $I_n A = A I_n = A$ for any $n \times n$ matrix $A$.  Rectangular
    variants ($n \neq m$) arise naturally as the pseudo-identity in
    least-squares problems and in constructing projection matrices.

    Parameters
    ----------
    n : int
        Number of rows.
    m : int, optional
        Number of columns.  Defaults to ``n``, yielding a square identity.
    dtype : lucid.dtype, optional
        Scalar data type.  Defaults to the global default dtype
        (``lucid.float32`` unless overridden).
    device : str or lucid.device, optional
        Target device — ``"cpu"`` or ``"metal"``.
    requires_grad : bool, optional
        If ``True``, downstream operations are tracked by autograd.
        Default: ``False``.

    Returns
    -------
    Tensor
        2-D tensor of shape ``(n, m)`` with $E_{ij} = \delta_{ij}$.

    Notes
    -----
    The identity matrix is its own inverse and its own transpose:
    $I_n^{-1} = I_n^\top = I_n$.  Its eigenvalues are all $1$, and it
    is simultaneously orthogonal, symmetric, idempotent ($I^2 = I$),
    and unitary.

    In deep learning, identity initialisations are used in recurrent
    networks (e.g. the IRNN, Le et al. 2015) to preserve gradient norms
    across time steps, exploiting the fact that the spectral radius of
    $I$ is exactly $1$.

    Examples
    --------
    >>> import lucid
    >>> lucid.eye(3)
    Tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])

    Rectangular variant:

    >>> lucid.eye(2, 4)
    Tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.]])

    Verify $I A = A$:

    >>> A = lucid.tensor([[1., 2.], [3., 4.]])
    >>> (lucid.eye(2) @ A - A).abs().max()
    Tensor(0.)
    """
    _dt, _dev, _rg = normalize_factory_kwargs(dtype, device, requires_grad)
    _m = m if m is not None else n
    impl = _C_engine.eye(n, _m, 0, _dt, _dev)
    return _wrap(_impl_with_grad(impl, _rg) if _rg else impl)


def arange(
    start: float,
    end: float | None = None,
    step: float = 1,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    r"""Return a 1-D tensor of evenly spaced values over a half-open interval.

    Generates the arithmetic sequence

    .. math::

        x_k = \texttt{start} + k \cdot \texttt{step},
        \quad k = 0, 1, \ldots, N-1

    where $N = \left\lfloor \dfrac{\texttt{end} - \texttt{start}}{\texttt{step}} \right\rfloor$
    is the number of elements.  The interval is **half-open**: ``start``
    is included, ``end`` is excluded, mirroring Python's built-in
    ``range``.

    When called with a single positional argument ``arange(n)``, the call
    is reinterpreted as ``arange(0, n, 1)``, yielding
    $[0, 1, \ldots, n-1]$.

    Parameters
    ----------
    start : float
        Starting value of the sequence (inclusive).  When ``end`` is
        ``None``, this argument is treated as ``end`` and ``start`` is
        set to ``0``.
    end : float, optional
        End of the interval (exclusive).  Required unless using the
        single-argument form.
    step : float, optional
        Spacing between consecutive values.  May be negative (producing
        a decreasing sequence) provided ``start > end``.  Default: ``1``.
    dtype : lucid.dtype, optional
        Scalar data type of the output.  If ``None``, inferred from
        the types of ``start``, ``end``, and ``step``: integer arguments
        yield ``int64``; any float argument yields the global float default.
    device : str or lucid.device, optional
        Target device — ``"cpu"`` or ``"metal"``.

    Returns
    -------
    Tensor
        1-D tensor containing the arithmetic sequence.

    Notes
    -----
    Due to floating-point rounding, the number of elements may differ
    from the naively expected $\lceil (\texttt{end} - \texttt{start}) /
    \texttt{step} \rceil$ by $\pm 1$.  When exact element counts matter,
    prefer `linspace` which always produces exactly ``steps`` values.

    The last element is always **strictly less than** ``end`` (for
    positive ``step``) or **strictly greater than** ``end`` (for negative
    ``step``).

    Examples
    --------
    >>> import lucid
    >>> lucid.arange(5).tolist()
    [0, 1, 2, 3, 4]

    >>> lucid.arange(1.0, 2.0, 0.25).tolist()
    [1.0, 1.25, 1.5, 1.75]

    Descending sequence:

    >>> lucid.arange(5, 0, -1).tolist()
    [5, 4, 3, 2, 1]

    Position encoding indices:

    >>> pos = lucid.arange(512, dtype=lucid.float32)
    """
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    if end is None:
        start, end = 0.0, float(start)
    return _wrap(_C_engine.arange(start, end, step, _dt, _dev))


def linspace(
    start: float,
    end: float,
    steps: int,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    r"""Return a 1-D tensor of ``steps`` equally spaced values over a closed interval.

    Generates the arithmetic sequence

    .. math::

        x_k = \texttt{start} + k \cdot \frac{\texttt{end} - \texttt{start}}{\texttt{steps} - 1},
        \quad k = 0, 1, \ldots, \texttt{steps} - 1

    Unlike `arange`, the interval is **closed** on both ends: ``start``
    and ``end`` are both included in the output, and the number of
    elements is always exactly ``steps``.

    Parameters
    ----------
    start : float
        Starting value of the sequence (inclusive).
    end : float
        Ending value of the sequence (inclusive).
    steps : int
        Number of evenly spaced samples.  Must be $\geq 1$.
        When ``steps = 1`` the output contains only ``start``.
    dtype : lucid.dtype, optional
        Scalar data type.  Defaults to the global default float dtype.
    device : str or lucid.device, optional
        Target device — ``"cpu"`` or ``"metal"``.

    Returns
    -------
    Tensor
        1-D tensor of shape ``(steps,)`` with evenly spaced values.

    Notes
    -----
    ``linspace`` is preferable to ``arange`` whenever:

    * The exact number of points is prescribed (e.g. FFT grid,
      sinusoidal position encodings, Gaussian quadrature nodes).
    * Floating-point rounding in the step size would cause off-by-one
      element counts.

    The spacing between consecutive elements is exactly
    $\Delta = (\texttt{end} - \texttt{start}) / (\texttt{steps} - 1)$,
    so the output satisfies $x_{\texttt{steps}-1} = \texttt{end}$ to
    within floating-point precision.

    For frequency grids in signal processing the common pattern is:

    .. math::

        f_k = \frac{k}{N} \cdot f_s, \quad k = 0, \ldots, N-1

    which can be constructed as ``linspace(0, fs, N, endpoint=False)``
    (use ``arange(N) / N * fs`` for the half-open variant).

    Examples
    --------
    >>> import lucid
    >>> lucid.linspace(0.0, 1.0, 5).tolist()
    [0.0, 0.25, 0.5, 0.75, 1.0]

    Sinusoidal position encoding grid:

    >>> t = lucid.linspace(0.0, 2 * 3.14159, 64)
    >>> t.shape
    (64,)

    Single-element edge case:

    >>> lucid.linspace(3.0, 7.0, 1).tolist()
    [3.0]
    """
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    return _wrap(_C_engine.linspace(start, end, steps, _dt, _dev))


def zeros_like(
    t: Tensor,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Return a zero-filled tensor with the same shape, dtype, and device as ``t``.

    Equivalent to ``lucid.zeros(t.shape, dtype=t.dtype, device=t.device)``,
    but infers the metadata from an existing tensor so the caller need not
    repeat it.  Every element satisfies

    .. math::

        Z_{i_1, \ldots, i_n} = 0
        \quad \forall\; (i_1, \ldots, i_n) \in \prod_k [0, s_k)

    where $(s_1, \ldots, s_n) = \texttt{t.shape}$.

    Parameters
    ----------
    t : Tensor
        Reference tensor.  Its ``shape``, ``dtype``, and ``device`` are
        used as defaults for the output.
    dtype : lucid.dtype, optional
        Override the data type.  When ``None`` (default), inherits
        ``t.dtype``.
    device : str or lucid.device, optional
        Override the device.  When ``None`` (default), inherits
        ``t.device``.
    requires_grad : bool, optional
        If ``True``, downstream operations are tracked by autograd.
        Default: ``False``.

    Returns
    -------
    Tensor
        Zero tensor with the same shape (and optionally dtype/device) as ``t``.

    Notes
    -----
    A common use case is zeroing out a gradient accumulator that has the
    same shape as a parameter tensor:

    .. math::

        g \leftarrow \mathbf{0}_{\text{like}(\theta)}

    This avoids hard-coding shape constants and ensures dtype consistency
    (e.g. ``float16`` parameters get ``float16`` zero gradients).

    Examples
    --------
    >>> import lucid
    >>> w = lucid.randn(3, 4, dtype=lucid.float16, device="metal")
    >>> g = lucid.zeros_like(w)
    >>> g.shape, g.dtype, g.device
    ((3, 4), lucid.float16, lucid.device('metal'))

    Override dtype on-the-fly:

    >>> mask = lucid.zeros_like(w, dtype=lucid.bool_)
    """
    _dt, _dev, _rg = normalize_factory_kwargs(
        dtype if dtype is not None else t.dtype,
        device if device is not None else t.device,
        requires_grad,
    )
    impl = _unwrap(t)
    out = _C_engine.zeros(list(impl.shape), _dt, _dev)
    return _wrap(_impl_with_grad(out, _rg) if _rg else out)


def ones_like(
    t: Tensor,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Return an all-ones tensor with the same shape, dtype, and device as ``t``.

    Equivalent to ``lucid.ones(t.shape, dtype=t.dtype, device=t.device)``,
    but infers metadata from an existing tensor.  Every element satisfies

    .. math::

        O_{i_1, \ldots, i_n} = 1
        \quad \forall\; (i_1, \ldots, i_n) \in \prod_k [0, s_k)

    Parameters
    ----------
    t : Tensor
        Reference tensor whose ``shape``, ``dtype``, and ``device`` are
        used as defaults.
    dtype : lucid.dtype, optional
        Override the data type.  Defaults to ``t.dtype``.
    device : str or lucid.device, optional
        Override the device.  Defaults to ``t.device``.
    requires_grad : bool, optional
        Enable autograd tracking on the output.  Default: ``False``.

    Returns
    -------
    Tensor
        All-ones tensor shaped like ``t``.

    Notes
    -----
    Multiplicative identity initialisation is used in layer-normalisation
    and group-normalisation to set the learnable scale parameter $\gamma$
    to $1$ at the start of training, so the network begins as a pure
    normalisation with no learned rescaling:

    .. math::

        \hat{x} = \gamma \cdot \frac{x - \mu}{\sigma} + \beta,
        \quad \gamma_0 = \mathbf{1},\; \beta_0 = \mathbf{0}

    Examples
    --------
    >>> import lucid
    >>> x = lucid.randn(2, 8)
    >>> gamma = lucid.ones_like(x)
    >>> gamma.shape
    (2, 8)

    Initialise scale parameters of a normalisation layer:

    >>> weight = lucid.ones_like(x, requires_grad=True)
    """
    _dt, _dev, _rg = normalize_factory_kwargs(
        dtype if dtype is not None else t.dtype,
        device if device is not None else t.device,
        requires_grad,
    )
    impl = _unwrap(t)
    out = _C_engine.ones(list(impl.shape), _dt, _dev)
    return _wrap(_impl_with_grad(out, _rg) if _rg else out)


def empty_like(
    t: Tensor,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
    requires_grad: bool = False,
) -> Tensor:
    r"""Return an uninitialised tensor with the same shape, dtype, and device as ``t``.

    Allocates backing memory matching ``t``'s metadata but performs no
    initialisation write.  Reading elements before overwriting them is
    undefined behaviour.  The typical use is to pre-allocate an output
    buffer for an in-place kernel call:

    .. math::

        \text{allocate}(|\texttt{t}|) \;\to\; \text{kernel\_write}(\text{buf})
        \;\to\; \text{read}(\text{buf})

    Parameters
    ----------
    t : Tensor
        Reference tensor whose ``shape``, ``dtype``, and ``device`` are
        inherited by the output.
    dtype : lucid.dtype, optional
        Override the data type.  Defaults to ``t.dtype``.
    device : str or lucid.device, optional
        Override the device.  Defaults to ``t.device``.
    requires_grad : bool, optional
        Enable autograd tracking.  Default: ``False``.

    Returns
    -------
    Tensor
        Uninitialised tensor shaped like ``t``.

    Notes
    -----
    Skipping the zero-fill write is only a win when the allocation is
    large and will be fully overwritten before any read.  For small
    tensors the branch-prediction and cache effects of the explicit
    initialisation kernel are negligible; prefer `zeros_like` in those
    cases for safety.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.randn(1024, 1024)
    >>> buf = lucid.empty_like(x)     # fast scratch space
    >>> buf.shape
    (1024, 1024)
    """
    _dt, _dev, _rg = normalize_factory_kwargs(
        dtype if dtype is not None else t.dtype,
        device if device is not None else t.device,
        requires_grad,
    )
    impl = _unwrap(t)
    out = _C_engine.empty(list(impl.shape), _dt, _dev)
    return _wrap(_impl_with_grad(out, _rg) if _rg else out)


def full_like(
    t: Tensor,
    fill_value: float,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    r"""Return a constant-filled tensor with the same shape, dtype, and device as ``t``.

    Every element of the output equals ``fill_value``:

    .. math::

        F_{i_1, \ldots, i_n} = c
        \quad \forall\; (i_1, \ldots, i_n) \in \prod_k [0, s_k)

    where $c = \texttt{fill\_value}$ and $(s_1, \ldots, s_n) = \texttt{t.shape}$.
    This is the shape-aware generalisation of `full`.

    Parameters
    ----------
    t : Tensor
        Reference tensor.  Shape, dtype, and device are inherited unless
        overridden by the keyword arguments.
    fill_value : float
        Scalar constant to broadcast across all elements.
    dtype : lucid.dtype, optional
        Override the data type.  When specified, the result is cast via
        ``astype`` after allocation, so the output dtype matches the
        override rather than ``t.dtype``.
    device : str or lucid.device, optional
        Override the device.  When specified, the result is moved via
        ``to`` after allocation.

    Returns
    -------
    Tensor
        Constant tensor shaped like ``t``.

    Notes
    -----
    A canonical application is initialising the **attention bias mask**
    before selectively unmasking positions.  Starting from
    $-\infty$ on a tensor shaped like the attention weight matrix ensures
    that masked positions become zero after softmax:

    .. math::

        \text{logit}_{ij} \leftarrow
        \begin{cases}
          \text{logit}_{ij} & \text{unmasked} \\
          -\infty           & \text{masked}
        \end{cases}
        \implies
        \text{softmax}(\text{logit})_i \;=\; 0 \text{ for masked } j

    Examples
    --------
    >>> import lucid
    >>> scores = lucid.randn(4, 8)
    >>> mask = lucid.full_like(scores, float("-inf"))
    >>> mask.shape
    (4, 8)

    Constant padding value:

    >>> x = lucid.randn(2, 3)
    >>> padded = lucid.full_like(x, -1.0)
    """
    out: Tensor = _wrap(_C_engine.full_like(_unwrap(t), fill_value, False))
    if dtype is not None and dtype is not t.dtype:
        out = out.astype(dtype)  # type: ignore[attr-defined]
    if device is not None and str(device) != str(t.device):
        out = out.to(device)
    return out


def logspace(
    start: float,
    end: float,
    steps: int,
    base: float = 10.0,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    r"""Return a 1-D tensor of ``steps`` values evenly spaced on a logarithmic scale.

    Generates the geometric sequence

    .. math::

        x_k = \texttt{base}^{y_k},
        \quad y_k = \texttt{start} + k \cdot \frac{\texttt{end} - \texttt{start}}{\texttt{steps} - 1},
        \quad k = 0, 1, \ldots, \texttt{steps} - 1

    Equivalently, the exponents $y_k$ are linearly spaced (as in
    `linspace`) and then the base is raised to those exponents.  The
    result spans the range $[\texttt{base}^\texttt{start},\,
    \texttt{base}^\texttt{end}]$ multiplicatively: consecutive elements
    satisfy $x_{k+1} / x_k = \text{const}$.

    Parameters
    ----------
    start : float
        Exponent of the first value.  The first element equals
        $\texttt{base}^\texttt{start}$.
    end : float
        Exponent of the last value.  The last element equals
        $\texttt{base}^\texttt{end}$.
    steps : int
        Number of samples.  Must be $\geq 1$.
    base : float, optional
        Logarithm base.  Common choices:

        * ``10.0`` (default) — decades, used in frequency / magnitude plots
        * ``2.0`` — octaves, used in learning-rate schedules and wavelets
        * ``math.e`` ($\approx 2.718$) — natural exponential spacing
    dtype : lucid.dtype, optional
        Scalar data type.  Defaults to the global default float dtype.
    device : str or lucid.device, optional
        Target device — ``"cpu"`` or ``"metal"``.

    Returns
    -------
    Tensor
        1-D tensor of shape ``(steps,)`` with geometrically spaced values.

    Notes
    -----
    Logarithmic spacing appears in:

    * **Learning-rate grid search** — scanning
      $[10^{-5}, 10^{-1}]$ with ``logspace(-5, -1, 9)`` covers five
      decades uniformly in log-space rather than concentrating samples
      near the upper bound as linear spacing would.
    * **Frequency analysis** — the mel and bark scales approximate
      human auditory perception and are nearly logarithmic in Hz.
    * **Sinusoidal position encodings** (Vaswani et al., 2017) use
      $\omega_k = 10000^{-2k/d}$, which is a logspace on base $10000$.

    The ratio between adjacent elements is constant:

    .. math::

        \frac{x_{k+1}}{x_k}
        = \texttt{base}^{\,(\texttt{end}-\texttt{start})\,/\,(\texttt{steps}-1)}

    Examples
    --------
    >>> import lucid
    >>> lucid.logspace(0, 3, 4).tolist()
    [1.0, 10.0, 100.0, 1000.0]

    Two-octave learning-rate sweep (base 2):

    >>> lrs = lucid.logspace(-8, -1, 8, base=2.0)
    >>> lrs.tolist()        # doctest: +SKIP
    [0.00390625, 0.0078125, ..., 0.5]

    Sinusoidal position encoding frequencies (Transformer convention):

    >>> d_model = 512
    >>> inv_freq = lucid.logspace(0, -1, d_model // 2, base=10000.0)
    """
    _dt, _dev, _ = normalize_factory_kwargs(dtype, device)
    return _wrap(_C_engine.logspace(start, end, steps, base, _dt, _dev))
