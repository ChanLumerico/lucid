"""Bijective ``Transform`` system for ``lucid.distributions``.

A :class:`Transform` is an invertible map ``T : X → Y`` with a known
log-absolute-determinant Jacobian, used to push a base distribution
through a measurable bijection (e.g. ``ExpTransform`` turns a Normal
into a LogNormal).

Composition + inversion are handled by :class:`ComposeTransform` and
:class:`Transform.__invert__` so users can build Lucid-side bijectors
without worrying about Jacobian bookkeeping.
"""

import math

import lucid
from lucid._tensor.tensor import Tensor
from lucid._types import DTypeLike, DeviceLike
from lucid.distributions._util import as_tensor as _as_tensor


class Transform:
    """Abstract bijection.  Subclasses override ``_call``, ``_inverse``
    and ``log_abs_det_jacobian``."""

    bijective: bool = True
    sign: int = 1  # +1 if monotone increasing, −1 if decreasing.
    event_dim: int = 0

    def __init__(self) -> None:
        """Initialise the transform with an empty inverse cache."""
        self._inv: Transform | None = None

    @property
    def inv(self) -> Transform:
        """Lazy inverse — caches an :class:`_InverseTransform` view."""
        if self._inv is None:
            self._inv = _InverseTransform(self)
        return self._inv

    def __invert__(self) -> Transform:
        """Return ``self.inv``. Enables the ``~transform`` syntax."""
        return self.inv

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the forward map ``T(x)`` by delegating to :meth:`_call`."""
        return self._call(x)

    def _call(self, x: Tensor) -> Tensor:
        """Forward map ``y = T(x)`` — must be overridden by subclasses."""
        raise NotImplementedError(f"{type(self).__name__}._call")

    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse map ``x = T^{-1}(y)`` — must be overridden by subclasses."""
        raise NotImplementedError(f"{type(self).__name__}._inverse")

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Log-absolute-determinant of the Jacobian ``∂T/∂x``.

        Subclasses must override and return :math:`\log|\det \partial y/\partial x|`
        broadcast over the input batch dimensions.
        """
        raise NotImplementedError(f"{type(self).__name__}.log_abs_det_jacobian")

    def __repr__(self) -> str:
        """Return a developer-facing string representation of the instance."""
        return f"{type(self).__name__}()"


class _InverseTransform(Transform):
    """View of the inverse of an underlying transform."""

    def __init__(self, base: Transform) -> None:
        """Wrap ``base`` and inherit its event-dim/sign metadata."""
        super().__init__()
        self._base = base
        self.event_dim = base.event_dim
        self.sign = base.sign
        self._inv = base

    @property
    def inv(self) -> Transform:
        """Return the wrapped base transform (inverse of this view)."""
        return self._base

    def _call(self, x: Tensor) -> Tensor:
        """Forward direction of the inverse view = ``base._inverse``."""
        return self._base._inverse(x)

    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse direction of the inverse view = ``base._call``."""
        return self._base._call(y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Jacobian of the inverse view: negate the base Jacobian with roles swapped."""
        return -self._base.log_abs_det_jacobian(y, x)


class ExpTransform(Transform):
    """``y = exp(x)`` — maps ℝ → (0, ∞)."""

    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = e^x`."""
        return x.exp()

    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: :math:`x = \\log(y)`."""
        return y.log()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r""":math:`\log|\partial y/\partial x| = x` since :math:`|dy/dx| = e^x`."""
        return x  # |dy/dx| = exp(x), so log|dy/dx| = x.


class SigmoidTransform(Transform):
    """``y = σ(x) = 1 / (1 + exp(−x))`` — maps ℝ → (0, 1)."""

    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = \\sigma(x)`."""
        return x.sigmoid()

    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: :math:`x = \\log(y/(1-y))`."""
        return y.log() - (1.0 - y).log()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Log-Jacobian :math:`\log\bigl(y\,(1-y)\bigr)` of the sigmoid map."""
        # log|y · (1 − y)|.
        return y.log() + (1.0 - y).log()


class TanhTransform(Transform):
    """``y = tanh(x)`` — maps ℝ → (−1, 1)."""

    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = \\tanh(x)`."""
        return x.tanh()

    def _inverse(self, y: Tensor) -> Tensor:
        r"""Inverse: :math:`x = \tfrac{1}{2}\log\!\bigl((1+y)/(1-y)\bigr)`."""
        return 0.5 * ((1.0 + y) / (1.0 - y)).log()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Numerically-stable :math:`\log(1 - \tanh^2(x)) = 2\bigl(\log 2 - x - \operatorname{softplus}(-2x)\bigr)`."""
        # log(1 − tanh²(x)) = 2·(log 2 − x − softplus(−2x)) — the stable
        # form used by the reference framework.
        return 2.0 * (math.log(2.0) - x - (-2.0 * x).exp().log1p())


class AffineTransform(Transform):
    """``y = loc + scale · x``."""

    def __init__(self, loc: Tensor | float, scale: Tensor | float) -> None:
        """Store the affine offset ``loc`` and multiplier ``scale``.

        The caller is responsible for ensuring ``scale > 0``; ``sign`` is
        fixed at ``+1`` here so the transform is treated as increasing.
        """
        super().__init__()
        self.loc = _as_tensor(loc)
        self.scale = _as_tensor(scale)
        self.sign = 1  # Caller is responsible for ``scale > 0``.

    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = \\text{loc} + \\text{scale} \\cdot x`."""
        return self.loc + self.scale * x

    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: :math:`x = (y - \\text{loc}) / \\text{scale}`."""
        return (y - self.loc) / self.scale

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Constant log-Jacobian :math:`\log|\text{scale}|`, broadcast to ``x``'s shape."""
        return self.scale.abs().log() + lucid.zeros(
            tuple(x.shape), dtype=x.dtype, device=x.device
        )


class PowerTransform(Transform):
    """``y = x^exponent`` (element-wise) for ``x > 0``."""

    def __init__(self, exponent: Tensor | float) -> None:
        """Store the (element-wise) power ``exponent``."""
        super().__init__()
        self.exponent = _as_tensor(exponent)

    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = x^{\\text{exponent}}`."""
        return x**self.exponent

    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: :math:`x = y^{1/\\text{exponent}}`."""
        return y ** (1.0 / self.exponent)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r""":math:`\log|\partial y/\partial x| = \log|\text{exponent}| + (\text{exponent} - 1)\log x`."""
        # |dy/dx| = exponent · x^(exponent − 1) ⇒ log|dy/dx| =
        #   log|exponent| + (exponent − 1)·log(x).
        return self.exponent.abs().log() + (self.exponent - 1.0) * x.log()


class SoftmaxTransform(Transform):
    """``y = softmax(x)`` along the last axis — pushes ℝ^K onto the
    open K-simplex.  Not a true bijection (loses 1-DOF to the
    constraint), but standard in normalising-flow stacks."""

    event_dim: int = 1

    def _call(self, x: Tensor) -> Tensor:
        """Forward: ``softmax(x, dim=-1)``."""
        from lucid.nn.functional.activations import softmax

        return softmax(x, dim=-1)

    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: ``log(y)``, anchored at the un-normalised log-probabilities.

        Any constant shift along the last axis yields the same softmax, so
        this is one canonical preimage rather than the unique inverse.
        """
        # Standard convention: take the un-normalised log-probabilities.
        # Any constant shift gives the same softmax — we anchor at log y.
        return y.log()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Pseudo-Jacobian :math:`\sum_k \log y_k` for the over-parameterised softmax."""
        # log |det J| = sum_k log y_k — pseudo-Jacobian for the
        # over-parameterised softmax, matching the reference framework.
        return y.log().sum(dim=-1)


class StickBreakingTransform(Transform):
    """Logistic stick-breaking from ℝ^(K-1) to the K-simplex.

    Maps ``x ∈ ℝ^(K-1)`` to ``y ∈ Δ^K`` by
    ``y_k = σ(x_k − log(K−k)) · ∏_{j<k} (1 − y_j)``,
    with the last component ``y_{K-1}`` taking the residual stick.
    """

    event_dim: int = 1

    def _call(self, x: Tensor) -> Tensor:
        r"""Forward stick-breaking: ``x ∈ ℝ^{K-1}`` → simplex ``y ∈ Δ^K``.

        Each :math:`y_k = \sigma(x_k - \log(K-k)) \cdot \prod_{j<k}(1 - y_j)`,
        with :math:`y_{K-1}` taking the residual stick.
        """
        # ``x`` shape (..., K-1).  We extend with a final 0 so the
        # cumulative-product step yields the residual stick automatically.
        K_minus_1: int = int(x.shape[-1])
        offsets: Tensor = lucid.arange(K_minus_1, 0, -1, dtype=x.dtype, device=x.device)
        # σ(x_k − log(K−k)) for k = 0..K-2.
        z: Tensor = (x - offsets.log()).sigmoid()
        # Stick-breaking: y_k = z_k · ∏_{j<k} (1 − z_j).
        # Compute via cumulative product of (1 − z) and pad with 1 in front.
        one_minus_z: Tensor = 1.0 - z
        # Manual prefix-product since lucid has no cumprod-from-1 op handy:
        # build [1, 1−z₀, (1−z₀)(1−z₁), …].
        prods: list[Tensor] = []
        running: Tensor = lucid.ones_like(z.narrow(-1, 0, 1)).squeeze(-1)
        prods.append(running)
        for k in range(K_minus_1 - 1):
            running = running * one_minus_z.narrow(-1, k, 1).squeeze(-1)
            prods.append(running)
        prefix: Tensor = lucid.stack(prods, dim=-1)  # shape (..., K-1)
        head: Tensor = z * prefix  # y_0 .. y_{K-2}
        # Last component is the residual: y_{K-1} = ∏_{j<K-1} (1 − z_j).
        last: Tensor = prefix.narrow(-1, K_minus_1 - 1, 1) * one_minus_z.narrow(
            -1, K_minus_1 - 1, 1
        )
        return lucid.cat([head, last], dim=-1)

    def _inverse(self, y: Tensor) -> Tensor:
        r"""Inverse stick-breaking: simplex ``y ∈ Δ^K`` → ``x ∈ ℝ^{K-1}``.

        Recovers :math:`z_k = y_k / (1 - \sum_{j<k} y_j)` and then
        :math:`x_k = \operatorname{logit}(z_k) + \log(K - k)`.
        """
        # Recover x_k from y via z_k = y_k / (1 − Σ_{j<k} y_j),
        # then x_k = logit(z_k) + log(K−k).
        K: int = int(y.shape[-1])
        K_minus_1: int = K - 1
        # Cumulative tail sums: stick remaining before drawing y_k.
        cum: Tensor = y.narrow(-1, 0, K_minus_1).cumsum(dim=-1)
        # remaining_before_k = 1 − Σ_{j<k} y_j  → shifted version.
        remaining: list[Tensor] = []
        ones: Tensor = lucid.ones_like(y.narrow(-1, 0, 1))
        remaining.append(ones)
        for k in range(1, K_minus_1):
            remaining.append((1.0 - cum.narrow(-1, k - 1, 1)))
        rem: Tensor = (
            lucid.cat(remaining, dim=-1).squeeze(-1)
            if K_minus_1 == 1
            else lucid.cat(remaining, dim=-1)
        )
        # rem has shape (..., K-1) by construction above.
        z: Tensor = y.narrow(-1, 0, K_minus_1) / rem
        offsets: Tensor = lucid.arange(K_minus_1, 0, -1, dtype=y.dtype, device=y.device)
        return (z.log() - (1.0 - z).log()) + offsets.log()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Standard simplex-to-:math:`\mathbb{R}^{K-1}` log-Jacobian.

        :math:`\log|\det J| = \sum_k \bigl[\log y_k + \log\!\bigl(1 - \sum_{j<k} y_j\bigr)\bigr]`.
        """
        # log|det J| = Σ_k log(y_k) + log(remaining stick before y_k) —
        # the standard simplex-to-ℝ^(K-1) Jacobian.
        K_minus_1: int = int(x.shape[-1])
        z: Tensor = y.narrow(-1, 0, K_minus_1)
        # remaining-before for each k.
        remaining: list[Tensor] = []
        cum: Tensor = z.cumsum(dim=-1)
        ones: Tensor = lucid.ones_like(z.narrow(-1, 0, 1))
        remaining.append(ones)
        for k in range(1, K_minus_1):
            remaining.append((1.0 - cum.narrow(-1, k - 1, 1)))
        rem: Tensor = lucid.cat(remaining, dim=-1)
        return (z.log() + rem.log() + (1.0 - z / rem).log()).sum(dim=-1)


class LowerCholeskyTransform(Transform):
    """ℝ^(D·(D+1)/2) → lower-triangular matrices with positive diagonal.

    The standard reparameterisation used to recover Cholesky factors
    from an unconstrained vector: off-diagonal entries pass through
    unchanged, diagonal entries go through ``softplus`` so they stay
    positive.
    """

    event_dim: int = 2

    def _call(self, x: Tensor) -> Tensor:
        """Forward: lower-triangular extraction with softplus on the diagonal.

        Off-diagonal entries pass through unchanged; diagonal entries are
        mapped via ``softplus`` to keep them strictly positive; the upper
        triangle is masked to zero.
        """
        from lucid.nn.functional.activations import softplus

        # ``x`` is already a ``(*batch, D, D)`` matrix; we mask out the
        # upper triangle (lucid has no native vec→tril helper, so the
        # caller pre-shapes the input).
        D: int = int(x.shape[-1])
        # Build a lower-triangular mask.
        mask: Tensor = _tril_mask(D, x.dtype, x.device)
        diag_mask: Tensor = _eye_mask(D, x.dtype, x.device)
        off_mask: Tensor = mask - diag_mask
        # Diagonal: softplus; off-diagonal: identity; rest: zero.
        diag: Tensor = softplus(x * diag_mask) * diag_mask
        off: Tensor = x * off_mask
        return diag + off

    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: invert softplus on the diagonal, leave the rest unchanged.

        Uses ``softplus^{-1}(z) = log(exp(z) - 1)`` on the diagonal entries
        of ``y`` and copies the strict lower triangle directly.
        """
        # Inverse: invert softplus on the diagonal, off-diagonal as-is,
        # zero out the upper triangle.
        D: int = int(y.shape[-1])
        diag_mask: Tensor = _eye_mask(D, y.dtype, y.device)
        tril_mask: Tensor = _tril_mask(D, y.dtype, y.device)
        off_mask: Tensor = tril_mask - diag_mask
        # ``softplus^{-1}(z) = log(exp(z) − 1)`` — stable for z > 0.
        diag_in: Tensor = (
            (y * diag_mask).exp().log1p()
            if False
            else ((y * diag_mask).exp() - 1.0).log() * diag_mask
        )
        off_in: Tensor = y * off_mask
        return diag_in + off_in

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Sum of :math:`\log \sigma(x_{ii})` over the diagonal.

        Off-diagonal entries contribute unit Jacobian; only the softplus
        applied to the diagonal contributes the non-trivial factor
        :math:`\sigma'(x) = \sigma(x)`, computed via the stable identity
        :math:`\log \sigma(x) = -\operatorname{softplus}(-x)`.
        """
        D: int = int(x.shape[-1])
        diag_mask: Tensor = _eye_mask(D, x.dtype, x.device)
        # σ(x) = 1/(1+exp(-x)) ⇒ log σ(x) = −softplus(−x).
        from lucid.nn.functional.activations import softplus

        log_sig: Tensor = -softplus(-(x * diag_mask))
        return (log_sig * diag_mask).sum(dim=(-2, -1))


def _eye_mask(D: int, dtype: DTypeLike, device: DeviceLike) -> Tensor:
    """``D×D`` identity mask as a Lucid tensor."""
    return lucid.eye(D, dtype=dtype, device=device)


def _tril_mask(D: int, dtype: DTypeLike, device: DeviceLike) -> Tensor:
    """``D×D`` lower-triangular indicator (1 on/below diag, 0 above)."""
    ones: Tensor = lucid.ones(D, D, dtype=dtype, device=device)
    return lucid.tril(ones)


class AbsTransform(Transform):
    """Element-wise absolute value ``y = |x|``.

    Not bijective (both ``x`` and ``-x`` map to the same ``y``), so this
    transform is non-invertible.  The inverse is defined as the identity
    (convention: assume the input is non-negative).

    ``log_abs_det_jacobian`` returns ``0.`` everywhere (|J| = 1).
    """

    bijective: bool = False
    sign: int = 1

    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = |x|`."""
        return x.abs()

    def _inverse(self, y: Tensor) -> Tensor:
        """Pseudo-inverse: returns ``y`` unchanged (assumes non-negative input)."""
        return y  # convention: non-negative pre-image

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Zero everywhere — :math:`|dy/dx| = 1` outside the kink at zero."""
        return lucid.zeros(tuple(x.shape), dtype=x.dtype, device=x.device)


class IndependentTransform(Transform):
    """Wraps another :class:`Transform` and reinterprets ``n`` batch dims
    as event dims.

    Useful when the inner transform operates element-wise but is applied to a
    batch of independent events that should be treated as a single vector event.

    Parameters
    ----------
    transform : Transform
        The inner bijection applied to each element.
    reinterpreted_batch_ndims : int
        Number of batch dimensions to treat as event dimensions.
    """

    def __init__(
        self,
        transform: Transform,
        reinterpreted_batch_ndims: int,
    ) -> None:
        """Wrap ``transform`` and promote ``reinterpreted_batch_ndims`` batch dims to event dims."""
        super().__init__()
        self.transform = transform
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        self.event_dim = transform.event_dim + reinterpreted_batch_ndims

    def _call(self, x: Tensor) -> Tensor:
        """Apply the wrapped transform element-wise."""
        return self.transform(x)

    def _inverse(self, y: Tensor) -> Tensor:
        """Apply the wrapped transform's inverse element-wise."""
        return self.transform._inverse(y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Sum the wrapped Jacobian over the reinterpreted trailing dims."""
        lp = self.transform.log_abs_det_jacobian(x, y)
        n = self.reinterpreted_batch_ndims - self.transform.event_dim
        if n > 0:
            lp = lp.sum(list(range(-n, 0)))
        return lp


class ReshapeTransform(Transform):
    """Reshape the event shape from ``in_shape`` to ``out_shape``.

    The total number of elements must be the same.

    Parameters
    ----------
    in_shape : tuple[int, ...]
        Event shape of the input.
    out_shape : tuple[int, ...]
        Event shape of the output.
    """

    def __init__(
        self,
        in_shape: tuple[int, ...],
        out_shape: tuple[int, ...],
    ) -> None:
        """Validate that ``in_shape`` and ``out_shape`` have the same total size and store them.

        Raises
        ------
        ValueError
            If ``prod(in_shape) != prod(out_shape)``.
        """
        super().__init__()
        import math as _math

        in_n = _math.prod(in_shape) if in_shape else 1
        out_n = _math.prod(out_shape) if out_shape else 1
        if in_n != out_n:
            raise ValueError(
                f"ReshapeTransform: in_shape {in_shape} ({in_n} elements) "
                f"!= out_shape {out_shape} ({out_n} elements)."
            )
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.event_dim = max(len(in_shape), len(out_shape))

    def _call(self, x: Tensor) -> Tensor:
        """Reshape the event tail of ``x`` from ``in_shape`` to ``out_shape``."""
        batch = x.shape[: x.dim() - len(self.in_shape)]
        return x.reshape(*batch, *self.out_shape)

    def _inverse(self, y: Tensor) -> Tensor:
        """Reshape the event tail of ``y`` from ``out_shape`` back to ``in_shape``."""
        batch = y.shape[: y.dim() - len(self.out_shape)]
        return y.reshape(*batch, *self.in_shape)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Zero Jacobian — pure reshape is volume-preserving."""
        batch = x.shape[: x.dim() - len(self.in_shape)]
        return lucid.zeros(tuple(batch), dtype=x.dtype, device=x.device)


class CorrCholeskyTransform(Transform):
    """Maps an unconstrained real vector of length ``d*(d-1)/2`` to the
    lower Cholesky factor of a correlation matrix of size ``d×d``.

    The parameterisation applies ``tanh`` to the free parameters, then
    normalises each row of the lower triangle so that each column of the
    resulting matrix has unit 2-norm (i.e. ``L Lᵀ`` has unit diagonal).

    This is the standard unconstrained parameterisation used in Stan and
    in the reference framework's ``CorrCholeskyTransform``.

    Parameters
    ----------
    dim : int
        Size of the square correlation matrix.
    """

    event_dim: int = 2

    def __init__(self, dim: int) -> None:
        """Store the correlation-matrix dimension ``dim`` (must be ``>= 2``)."""
        super().__init__()
        if dim < 2:
            raise ValueError(f"CorrCholeskyTransform: dim must be ≥ 2, got {dim}.")
        self.dim = dim

    # Dimension of the free vector
    @property
    def _free_ndim(self) -> int:
        """Number of free parameters in the unconstrained representation: :math:`d(d-1)/2`."""
        d = self.dim
        return d * (d - 1) // 2

    def _call(self, x: Tensor) -> Tensor:
        """Map unconstrained ``(..., d*(d-1)/2)`` vector to ``(..., d, d)`` Chol."""
        d = self.dim
        *batch, _ = x.shape
        z = x.tanh()  # (..., free_ndim)

        L_flat = lucid.zeros(*batch, d, d, dtype=x.dtype, device=x.device)
        # Fill strictly lower triangle column by column.
        idx = 0
        for col in range(d):
            # row starts at col+1 (zero-indexed)
            for row in range(col + 1, d):
                # Place the raw tanh value.  Normalisation happens per-row
                # after all elements are placed.
                L_flat = self._scatter_elem(L_flat, batch, row, col, z[..., idx])
                idx += 1

        # Normalise rows so L Lᵀ has unit diagonal:
        # for row i, norm² = Σ_{j=0}^{i-1} L_{ij}² + diag²  → set diag so norm=1.
        rows = []
        for row in range(d):
            if row == 0:
                # first diagonal element = 1
                diag_val = lucid.ones(*batch, 1, dtype=x.dtype, device=x.device)
                off = lucid.zeros(*batch, d, dtype=x.dtype, device=x.device)
                rows.append(off)
                # Actually build row by row using a different approach below.
            break
        # Use the cleaner vectorised approach: build the matrix row by row.
        return self._build_chol(z, batch, d, x.dtype, x.device)

    def _build_chol(
        self,
        z: Tensor,
        batch: list[int],
        d: int,
        dtype: DTypeLike,
        device: DeviceLike,
    ) -> Tensor:
        """Build the Cholesky factor row by row from ``z`` (tanh-ed free params)."""
        row_tensors: list[Tensor] = []
        idx = 0
        cumsum_sq: list[Tensor] = []  # cumulative row norms squared

        for row in range(d):
            if row == 0:
                # L[0, 0] = 1, rest of row = 0
                r = lucid.zeros(*batch, d, dtype=dtype, device=device)
                # set r[..., 0] = 1
                ones_val = lucid.ones(*batch, 1, dtype=dtype, device=device)
                r = lucid.cat(
                    [ones_val, lucid.zeros(*batch, d - 1, dtype=dtype, device=device)],
                    dim=-1,
                )
                row_tensors.append(r.unsqueeze(-2))
                cumsum_sq.append(lucid.ones(*batch, 1, dtype=dtype, device=device))
            else:
                # off-diagonal: z[..., idx:idx+row]
                n_free = row  # number of free params in this row
                zr = z[..., idx : idx + n_free]  # (..., row)
                idx += n_free

                # Scale: z_{row, col} goes into L_{row, col}; then normalise.
                # Build row with tanh values in positions 0..row-1, 0 elsewhere.
                if n_free > 0:
                    # Normalise: L[row, :row] = z_row * sqrt(cumprod(1 - z²))
                    # Actually use the standard normalisation:
                    # L[row,0] = z[0]
                    # L[row,1] = z[1]*sqrt(1-z[0]²)
                    # L[row,2] = z[2]*sqrt(1-z[0]²-z[1]²*(...))
                    # i.e. L[row, j] = z[j] * sqrt(1 - Σ_{k<j} L[row,k]²)
                    # We compute this step-by-step.
                    elems: list[Tensor] = []
                    cum_sq = lucid.zeros(*batch, dtype=dtype, device=device)
                    for j in range(n_free):
                        scale = (1.0 - cum_sq).sqrt()
                        # Clamp to avoid sqrt of negative from fp noise.
                        scale = lucid.where(
                            scale < 0.0,
                            lucid.zeros_like(scale),
                            scale,
                        )
                        elem = zr[..., j] * scale
                        elems.append(elem.unsqueeze(-1))
                        cum_sq = cum_sq + elem * elem

                    off = lucid.cat(elems, dim=-1)  # (..., row)
                    # diagonal element: sqrt(1 - ||off||²)
                    raw_diag_sq = 1.0 - (off * off).sum(dim=-1)
                    diag_sq = lucid.where(
                        raw_diag_sq < 0.0,
                        lucid.zeros_like(raw_diag_sq),
                        raw_diag_sq,
                    )
                    diag = diag_sq.sqrt()  # (...,)
                    # pad to length d: [off | diag | zeros]
                    zero_pad = lucid.zeros(
                        *batch, d - row - 1, dtype=dtype, device=device
                    )
                    full_row = lucid.cat([off, diag.unsqueeze(-1), zero_pad], dim=-1)
                else:
                    full_row = lucid.zeros(*batch, d, dtype=dtype, device=device)

                row_tensors.append(full_row.unsqueeze(-2))

        # Stack rows → (..., d, d)
        return lucid.cat(row_tensors, dim=-2)

    @staticmethod
    def _scatter_elem(
        L: Tensor,
        batch: list[int],
        row: int,
        col: int,
        val: Tensor,
    ) -> Tensor:
        """Placeholder scatter helper — retained for reference, no-op currently.

        The vectorised ``_build_chol`` path supersedes the original
        element-by-element scatter; this stub is kept so callers in older
        code paths continue to type-check.
        """
        # Helper — not used in the final path, kept for reference.
        return L

    def _inverse(self, y: Tensor) -> Tensor:
        """Extract the free parameters ``x`` from a Cholesky factor ``L``."""
        d = self.dim
        *batch, _, _ = y.shape
        # Each row i of L: first i elements are off-diagonal, element i is diag.
        # Back-solve for z from L[row, :row].
        free: list[Tensor] = []
        for row in range(1, d):
            off = y[..., row, :row]  # (..., row)
            # Recover z from the cumulative parameterisation.
            elems: list[Tensor] = []
            cum_sq = lucid.zeros(*batch, dtype=y.dtype, device=y.device)
            for j in range(row):
                raw_scale = (1.0 - cum_sq).sqrt()
                scale = lucid.where(
                    raw_scale < 1e-8,
                    lucid.full_like(raw_scale, 1e-8),
                    raw_scale,
                )
                z_raw = off[..., j] / scale
                z_j = lucid.where(
                    z_raw < -1.0 + 1e-6,
                    lucid.full_like(z_raw, -1.0 + 1e-6),
                    lucid.where(
                        z_raw > 1.0 - 1e-6,
                        lucid.full_like(z_raw, 1.0 - 1e-6),
                        z_raw,
                    ),
                )
                elems.append(z_j.unsqueeze(-1))
                cum_sq = cum_sq + (off[..., j]) ** 2

            free.append(lucid.cat(elems, dim=-1))  # (..., row)

        z = lucid.cat(free, dim=-1)  # (..., d*(d-1)/2)
        # inverse tanh: arctanh(z) = 0.5 * log((1+z)/(1-z))
        denom = lucid.where(
            (1.0 - z) < 1e-8,
            lucid.full_like(z, 1e-8),
            1.0 - z,
        )
        return 0.5 * ((1.0 + z) / denom).log()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Log-Jacobian of the unconstrained-to-Cholesky correlation map.

        Combines the tanh contribution :math:`\sum_{i>j} \log(1 - \tanh^2(x_{ij}))`
        with the closed-form Lewandowski normalisation terms expressed
        directly from the output diagonal entries of ``y``.
        """
        # log |det J| = Σ_{row>0} Σ_{j<row} log|∂L_{row,j}/∂x_{row,j}|
        # ∂tanh(x)/∂x = 1 - tanh²(x)  and  ∂(z·scale)/∂z = scale
        # Net contribution: Σ log(1-z²)/2 (tanh derivative) + scale terms.
        # We use the chain-rule result:
        # log|det J| = Σ_{i>j} log(1 - tanh²(x_{ij})) * 0.5  (tanh term)
        #            + Σ_{row} Σ_{j<row-1} log(scale_j)
        # For simplicity use the closed-form from the Stan reference:
        # log|det J| = Σ_{k=1}^{d-1} (d-k-1) * log(tanh²(x_k)) + sum log(1-tanh²)
        # This is computed from the y (output) more directly.
        d = self.dim
        # Sum over off-diagonal log-diagonal-scale terms.
        log_diags: list[Tensor] = []
        for row in range(1, d):
            diag_elem = y[..., row, row]  # L[row,row]
            # Each diagonal contributes log(L[row,row]) * (d - row) times
            # (from the normalisation chain) — use (d - row - 1) * log(L_{rr})
            # per row from the Lewandowski formula.
            log_diags.append(
                (float(d - row - 1) + 1.0) * diag_elem.clamp(min=1e-8).log()
            )
        if not log_diags:
            return lucid.zeros(tuple(x.shape[:-1]), dtype=x.dtype, device=x.device)
        # tanh contribution: Σ log(1 - tanh²(x)) = Σ log(1 - z²)
        z = x.tanh()
        raw_1mz2 = 1.0 - z * z
        clamped_1mz2 = lucid.where(
            raw_1mz2 < 1e-8,
            lucid.full_like(raw_1mz2, 1e-8),
            raw_1mz2,
        )
        log_1_minus_z2 = clamped_1mz2.log().sum(dim=-1)
        diag_sum = lucid.cat([ld.unsqueeze(-1) for ld in log_diags], dim=-1).sum(dim=-1)
        return log_1_minus_z2 + diag_sum


class CumulativeDistributionTransform(Transform):
    """Applies a distribution's CDF as a bijection ``y = F(x)``.

    The inverse is the quantile (ICDF) function.  Useful for turning an
    arbitrary continuously distributed variable into a Uniform(0,1).

    Parameters
    ----------
    distribution : Distribution
        The distribution whose CDF/ICDF pair defines the transform.
    """

    event_dim: int = 0

    def __init__(self, distribution: Distribution) -> None:
        """Store the distribution whose CDF/ICDF pair defines the transform."""
        super().__init__()
        self.distribution = distribution

    def _call(self, x: Tensor) -> Tensor:
        """Forward: :math:`y = F(x)` via ``distribution.cdf``."""
        return self.distribution.cdf(x)

    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: :math:`x = F^{-1}(y)` via ``distribution.icdf``."""
        return self.distribution.icdf(y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        r""":math:`\log|F'(x)| = \log p(x)` — the base distribution's log-density at ``x``."""
        return self.distribution.log_prob(x)


class StackTransform(Transform):
    """Apply a list of transforms to corresponding slices along ``dim``.

    ``y[..., k, ...] = transforms[k](x[..., k, ...])``

    The number of slices must equal ``len(transforms)``.

    Parameters
    ----------
    transforms : list[Transform]
        One transform per slice.
    dim : int
        Dimension along which to slice.  Default ``0``.
    """

    def __init__(self, transforms: list[Transform], dim: int = 0) -> None:
        """Store the per-slice transforms and the stacking axis ``dim``.

        Raises
        ------
        ValueError
            If ``transforms`` is empty.
        """
        super().__init__()
        if not transforms:
            raise ValueError("StackTransform: transforms cannot be empty.")
        self.transforms = list(transforms)
        self.dim = dim
        self.event_dim = max(t.event_dim for t in transforms)

    def _call(self, x: Tensor) -> Tensor:
        """Unbind ``x`` along ``dim`` and apply each transform to the matching slice.

        Raises
        ------
        ValueError
            If the number of slices differs from the number of stored transforms.
        """
        slices = x.unbind(self.dim)
        if len(slices) != len(self.transforms):
            raise ValueError(
                f"StackTransform: got {len(slices)} slices but "
                f"{len(self.transforms)} transforms."
            )
        return lucid.stack(
            [t(s) for t, s in zip(self.transforms, slices)], dim=self.dim
        )

    def _inverse(self, y: Tensor) -> Tensor:
        """Unbind ``y`` along ``dim`` and apply each transform's inverse per slice."""
        slices = y.unbind(self.dim)
        return lucid.stack(
            [t._inverse(s) for t, s in zip(self.transforms, slices)], dim=self.dim
        )

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Per-slice Jacobians stacked back along ``dim``."""
        xs = x.unbind(self.dim)
        ys = y.unbind(self.dim)
        ladjs = [
            t.log_abs_det_jacobian(xi, yi) for t, xi, yi in zip(self.transforms, xs, ys)
        ]
        return lucid.stack(ladjs, dim=self.dim)


class CatTransform(Transform):
    """Apply a list of transforms to contiguous slices along ``dim``,
    where each transform handles ``lengths[i]`` elements.

    ``y = cat([T_i(x_i) for i in range(n)], dim=dim)``

    Parameters
    ----------
    transforms : list[Transform]
        One transform per partition.
    dim : int
        Concatenation dimension.  Default ``0``.
    lengths : list[int] | None
        Length of each slice.  If ``None``, slices are equal.
    """

    def __init__(
        self,
        transforms: list[Transform],
        dim: int = 0,
        lengths: list[int] | None = None,
    ) -> None:
        """Store the per-partition transforms, concat axis, and partition lengths.

        Raises
        ------
        ValueError
            If ``transforms`` is empty.
        """
        super().__init__()
        if not transforms:
            raise ValueError("CatTransform: transforms cannot be empty.")
        self.transforms = list(transforms)
        self.dim = dim
        self.lengths = lengths
        self.event_dim = max(t.event_dim for t in transforms)

    def _split(self, x: Tensor) -> list[Tensor]:
        """Split ``x`` along ``self.dim`` according to ``self.lengths``.

        If ``self.lengths`` is ``None``, the axis size must be divisible by
        the number of transforms — partitions of equal length are used.

        Raises
        ------
        ValueError
            If the axis size is not divisible by ``len(self.transforms)``
            when ``self.lengths`` is ``None``.
        """
        if self.lengths is None:
            total = x.shape[self.dim]
            n = len(self.transforms)
            if total % n != 0:
                raise ValueError(
                    f"CatTransform: dim {self.dim} size {total} not divisible by {n}."
                )
            lengths = [total // n] * n
        else:
            lengths = self.lengths
        return list(x.split(lengths, dim=self.dim))

    def _call(self, x: Tensor) -> Tensor:
        """Split ``x`` along ``dim``, apply each transform, and concatenate."""
        parts = self._split(x)
        return lucid.cat([t(p) for t, p in zip(self.transforms, parts)], dim=self.dim)

    def _inverse(self, y: Tensor) -> Tensor:
        """Split ``y`` along ``dim``, apply each inverse, and concatenate."""
        parts = self._split(y)
        return lucid.cat(
            [t._inverse(p) for t, p in zip(self.transforms, parts)], dim=self.dim
        )

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Per-partition Jacobians concatenated back along ``dim``."""
        xs = self._split(x)
        ys = self._split(y)
        ladjs = [
            t.log_abs_det_jacobian(xi, yi) for t, xi, yi in zip(self.transforms, xs, ys)
        ]
        return lucid.cat(ladjs, dim=self.dim)


class ComposeTransform(Transform):
    """``T = T_n ∘ … ∘ T_2 ∘ T_1`` — applied left-to-right."""

    def __init__(self, parts: list[Transform]) -> None:
        """Store the ordered list of sub-transforms.

        Raises
        ------
        ValueError
            If ``parts`` is empty.
        """
        super().__init__()
        if not parts:
            raise ValueError("ComposeTransform: parts cannot be empty.")
        self.parts = list(parts)
        self.event_dim = max(p.event_dim for p in parts)

    def _call(self, x: Tensor) -> Tensor:
        """Forward: apply every sub-transform in order."""
        for p in self.parts:
            x = p(x)
        return x

    def _inverse(self, y: Tensor) -> Tensor:
        """Inverse: apply each sub-transform's inverse in reverse order."""
        for p in reversed(self.parts):
            y = p._inverse(y)
        return y

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Cumulative log-Jacobian across the composition.

        Re-evaluates the chain of forward maps to obtain the intermediate
        states needed by each sub-Jacobian, then sums the contributions.
        """
        # Need intermediate values to evaluate each Jacobian.
        total = lucid.zeros(tuple(x.shape), dtype=x.dtype, device=x.device)
        cur = x
        for p in self.parts:
            nxt = p(cur)
            total = total + p.log_abs_det_jacobian(cur, nxt)
            cur = nxt
        return total


# ── TransformedDistribution ──────────────────────────────────────────────

from lucid.distributions.distribution import Distribution


class TransformedDistribution(Distribution):
    """Push ``base_distribution`` through a (possibly composite) bijector.

    ``rsample = transform(base_dist.rsample())`` and ``log_prob`` accounts
    for the Jacobian via the change-of-variable formula.
    """

    def __init__(
        self,
        base_distribution: Distribution,
        transforms: Transform | list[Transform],
        validate_args: bool | None = None,
    ) -> None:
        """Construct a transformed distribution.

        Parameters
        ----------
        base_distribution : Distribution
            Underlying distribution whose samples will be pushed through
            ``transforms``.
        transforms : Transform | list[Transform]
            A single transform or an ordered list applied left-to-right.
        validate_args : bool | None, optional
            Forwarded to :class:`Distribution`.
        """
        if isinstance(transforms, Transform):
            self.transforms: list[Transform] = [transforms]
        else:
            self.transforms = list(transforms)
        self.base_dist = base_distribution
        super().__init__(
            batch_shape=tuple(base_distribution.batch_shape),
            event_shape=tuple(base_distribution.event_shape),
            validate_args=validate_args,
        )

    @property
    def has_rsample(self) -> bool:  # type: ignore[override]
        """Whether reparameterised sampling is available — inherited from the base distribution."""
        return self.base_dist.has_rsample

    def _push(self, x: Tensor) -> Tensor:
        """Apply every transform in ``self.transforms`` in order to ``x``."""
        for t in self.transforms:
            x = t(x)
        return x

    def rsample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Push a reparameterised base sample through the transform chain."""
        return self._push(self.base_dist.rsample(sample_shape))

    def sample(self, sample_shape: tuple[int, ...] = ()) -> Tensor:
        """Push a (non-reparameterised) base sample through the transform chain."""
        return self._push(self.base_dist.sample(sample_shape))

    def log_prob(self, value: Tensor) -> Tensor:
        r"""Evaluate the log-density of ``value`` under the transformed distribution.

        Uses the change-of-variable formula:

        .. math::

            \log p(y) = \log p_{\text{base}}(T^{-1}(y))
                - \sum_i \log\!\bigl|\det \partial T_i / \partial x_i\bigr|

        with the chain unwound by walking the transforms in reverse.
        """
        # Walk transforms in reverse, building the inverse chain and the
        # Jacobian correction.
        log_det = lucid.zeros(
            tuple(value.shape), dtype=value.dtype, device=value.device
        )
        cur = value
        for t in reversed(self.transforms):
            prev = t._inverse(cur)
            log_det = log_det + t.log_abs_det_jacobian(prev, cur)
            cur = prev
        return self.base_dist.log_prob(cur) - log_det
