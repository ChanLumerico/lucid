"""
nn.init: parameter initialization functions.
All functions operate in-place and return the tensor.
"""

import math
from typing import TYPE_CHECKING

import lucid as _lucid
from lucid._C import engine as _C_engine
from lucid._tensor.tensor import _impl_with_grad as _iwg

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _fill_from_impl(tensor: Tensor, src_impl: object) -> Tensor:
    """Replace tensor's impl with src_impl, preserving requires_grad."""
    rg = tensor._impl.requires_grad
    impl = _C_engine.reshape(src_impl, list(tensor.shape))  # type: ignore[attr-defined]
    tensor._impl = _iwg(impl, rg)
    return tensor


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    """Fill tensor in-place with values from U(a, b)."""
    return _fill_from_impl(
        tensor,
        _C_engine.uniform(
            list(tensor.shape), a, b, tensor._impl.dtype, tensor._impl.device
        ),
    )


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Fill tensor in-place with values from N(mean, std²)."""
    return _fill_from_impl(
        tensor,
        _C_engine.normal(
            list(tensor.shape), mean, std, tensor._impl.dtype, tensor._impl.device
        ),
    )


def constant_(tensor: Tensor, val: float) -> Tensor:
    """Fill tensor in-place with a constant value."""
    return _fill_from_impl(
        tensor,
        _C_engine.full(
            list(tensor.shape), val, tensor._impl.dtype, tensor._impl.device
        ),
    )


def ones_(tensor: Tensor) -> Tensor:
    """Fill tensor in-place with ones."""
    return _fill_from_impl(
        tensor,
        _C_engine.ones(list(tensor.shape), tensor._impl.dtype, tensor._impl.device),
    )


def zeros_(tensor: Tensor) -> Tensor:
    """Fill tensor in-place with zeros."""
    return _fill_from_impl(
        tensor,
        _C_engine.zeros(list(tensor.shape), tensor._impl.dtype, tensor._impl.device),
    )


def eye_(tensor: Tensor) -> Tensor:
    """Fill a 2D tensor in-place as an identity matrix."""
    if tensor.ndim != 2:
        raise ValueError("eye_() requires a 2D tensor")
    n, m = tensor.shape
    return _fill_from_impl(
        tensor, _C_engine.eye(n, m, 0, tensor._impl.dtype, tensor._impl.device)
    )


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill tensor with Xavier uniform values."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return uniform_(tensor, -a, a)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill tensor with Xavier normal values."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    """Fill tensor with Kaiming uniform values."""
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> Tensor:
    """Fill tensor with Kaiming normal values."""
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std)


def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> Tensor:
    """Fill tensor with truncated normal values via rejection sampling."""
    shape = list(tensor.shape) if tensor.shape else [1]
    total = 1
    for s in shape:
        total *= s

    dt = tensor._impl.dtype
    dev = tensor._impl.device
    filled_parts = []
    remaining = total
    while remaining > 0:
        needed = max(remaining * 4, 16)
        candidates = _C_engine.normal([needed], mean, std, dt, dev)
        # Build mask: a <= x <= b
        lo = _C_engine.full([needed], a, dt, dev)
        hi = _C_engine.full([needed], b, dt, dev)
        ge_a = _C_engine.greater_equal(candidates, lo)
        le_b = _C_engine.less_equal(candidates, hi)
        mask = _C_engine.bitwise_and(ge_a, le_b)
        valid = _C_engine.masked_select(candidates, mask)
        n_valid = int(list(valid.shape)[0])
        take = min(n_valid, remaining)
        if take > 0:
            # Slice first `take` elements via gather + arange.
            idx = _C_engine.arange(0, take, 1, _C_engine.I32, dev)
            filled_parts.append(_C_engine.gather(valid, idx, 0))
            remaining -= take

    result = _C_engine.concatenate(filled_parts, 0)
    return _fill_from_impl(tensor, result)


def orthogonal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill tensor with a (semi-)orthogonal matrix via QR.

    For tensors with rank > 2 the leading axis is flattened against the
    rest, the resulting 2-D matrix is made orthogonal, and the original
    shape is restored — matching the reference framework.
    """
    if tensor.ndim < 2:
        raise ValueError("orthogonal_() requires at least a 2D tensor")
    rows: int = int(tensor.shape[0])
    cols: int = int(tensor.numel() // rows)
    # All work on the engine — no numpy.
    flat: Tensor = _lucid.randn(rows, cols, device=tensor._impl.device)
    # Use QR on the (max-dim × min-dim) shape so we always get an
    # orthonormal Q with the right number of columns.
    if rows < cols:
        q_full, _r = _lucid.linalg.qr(flat.mT)
        q: Tensor = q_full.narrow(1, 0, rows).mT
    else:
        q_full, _r = _lucid.linalg.qr(flat)
        q = q_full.narrow(1, 0, cols)
    if gain != 1.0:
        q = q * gain
    src_t: Tensor = _lucid.tensor(
        q, dtype=tensor._impl.dtype, device=tensor._impl.device
    )
    return _fill_from_impl(tensor, src_t._impl)


def sparse_(tensor: Tensor, sparsity: float, std: float = 0.01) -> Tensor:
    """Fill a 2-D tensor in-place with a sparse matrix.

    Each column has ``floor(sparsity * rows)`` zero entries (drawn at
    random); the remaining entries are sampled from ``N(0, std²)``.
    Only 2-D tensors are supported, matching the reference framework.
    """
    if tensor.ndim != 2:
        raise ValueError("sparse_() requires a 2D tensor")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity!r}")

    rows: int = int(tensor.shape[0])
    cols: int = int(tensor.shape[1])
    n_zero: int = int(math.floor(sparsity * rows))
    dev = tensor._impl.device
    # Per-column random row selection: ``argsort`` of uniform draws gives a
    # uniform permutation; take the first ``n_zero`` rows of each column.
    src: Tensor = _lucid.normal(0.0, std, size=(rows, cols), device=dev)
    if n_zero > 0:
        # noise is shape (rows, cols); per-column argsort along dim=0.
        noise: Tensor = _lucid.rand(rows, cols, device=dev)
        perm: Tensor = noise.argsort(dim=0)  # (rows, cols) int.
        zero_rows: Tensor = perm.narrow(0, 0, n_zero)  # (n_zero, cols).
        # Build a (rows, cols) bool mask: 1 where a row was chosen.
        mask: Tensor = _lucid.zeros(rows, cols, device=dev)
        ones: Tensor = _lucid.ones(n_zero, cols, device=dev)
        mask = mask.scatter_add(0, zero_rows, ones)
        src = _lucid.where(mask > 0.0, _lucid.zeros_like(src), src)
    src_t: Tensor = _lucid.tensor(src, dtype=tensor._impl.dtype, device=dev)
    return _fill_from_impl(tensor, src_t._impl)


def dirac_(tensor: Tensor, groups: int = 1) -> Tensor:
    """Fill a 3/4/5-D tensor in-place with a Dirac-delta filter.

    Conv weights of shape ``(out_channels, in_channels // groups, *K)``
    are filled so the convolution acts as the identity (per group, with
    the kernel centred).  Useful for residual / identity initialisation.
    """
    if tensor.ndim not in (3, 4, 5):
        raise ValueError(f"dirac_() expects a 3/4/5-D tensor; got ndim={tensor.ndim}")
    out_ch: int = int(tensor.shape[0])
    in_ch_per_group: int = int(tensor.shape[1])
    if out_ch % groups != 0:
        raise ValueError(
            f"out_channels ({out_ch}) must be divisible by groups ({groups})"
        )

    out_per_group: int = out_ch // groups
    min_dim: int = min(in_ch_per_group, out_per_group)
    spatial_centres: tuple[int, ...] = tuple(int(s) // 2 for s in tensor.shape[2:])
    dev = tensor._impl.device
    shape: tuple[int, ...] = tuple(int(s) for s in tensor.shape)
    src: Tensor = _lucid.zeros(*shape, device=dev)
    # Each (out_idx, d, *centres) gets a 1.  Build per-axis index tensors of
    # length ``groups·min_dim`` and use ``index_put_`` for the in-place write.
    n_writes: int = groups * min_dim
    if n_writes > 0:
        out_idx_list: list[int] = []
        in_idx_list: list[int] = []
        for g in range(groups):
            for d in range(min_dim):
                out_idx_list.append(g * out_per_group + d)
                in_idx_list.append(d)
        idx_tensors: list[Tensor] = [
            _lucid.tensor(out_idx_list, dtype=_lucid.int64, device=dev),
            _lucid.tensor(in_idx_list, dtype=_lucid.int64, device=dev),
        ]
        for c in spatial_centres:
            idx_tensors.append(
                _lucid.tensor([c] * n_writes, dtype=_lucid.int64, device=dev)
            )
        src = _lucid.index_put(
            src,
            tuple(idx_tensors),
            _lucid.ones(n_writes, device=dev),
        )
    src_t: Tensor = _lucid.tensor(src, dtype=tensor._impl.dtype, device=dev)
    return _fill_from_impl(tensor, src_t._impl)


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    """Return the recommended gain for an activation function."""
    _gains: dict[str, float] = {
        "linear": 1.0,
        "conv1d": 1.0,
        "conv2d": 1.0,
        "conv3d": 1.0,
        "sigmoid": 1.0,
        "tanh": 5.0 / 3.0,
        "relu": math.sqrt(2.0),
        "selu": 3.0 / 4.0,
    }
    if nonlinearity == "leaky_relu":
        slope = param if param is not None else 0.01
        return math.sqrt(2.0 / (1 + slope**2))
    if nonlinearity in _gains:
        return _gains[nonlinearity]
    raise ValueError(f"Unsupported nonlinearity: {nonlinearity!r}")


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    """Compute fan_in and fan_out for Linear and Conv weight tensors."""
    ndim = tensor.ndim
    if ndim < 2:
        raise ValueError("fan_in/fan_out requires at least 2D tensor")
    receptive = 1
    if ndim > 2:
        for s in tensor.shape[2:]:
            receptive *= s
    fan_in = tensor.shape[1] * receptive
    fan_out = tensor.shape[0] * receptive
    return fan_in, fan_out


def _calculate_correct_fan(tensor: Tensor, mode: str) -> int:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        return fan_in
    if mode == "fan_out":
        return fan_out
    raise ValueError(f"Unknown mode: {mode!r}")


# Non-inplace aliases (deprecated in the reference framework but still widely used).
constant = constant_
dirac = dirac_
eye = eye_
kaiming_normal = kaiming_normal_
kaiming_uniform = kaiming_uniform_
normal = normal_
orthogonal = orthogonal_
sparse = sparse_
uniform = uniform_
xavier_normal = xavier_normal_
xavier_uniform = xavier_uniform_

__all__ = [
    "calculate_gain",
    "constant_",
    "constant",
    "dirac_",
    "dirac",
    "eye_",
    "eye",
    "kaiming_normal_",
    "kaiming_normal",
    "kaiming_uniform_",
    "kaiming_uniform",
    "normal_",
    "normal",
    "ones_",
    "orthogonal_",
    "orthogonal",
    "sparse_",
    "sparse",
    "trunc_normal_",
    "uniform_",
    "uniform",
    "xavier_normal_",
    "xavier_normal",
    "xavier_uniform_",
    "xavier_uniform",
    "zeros_",
]
