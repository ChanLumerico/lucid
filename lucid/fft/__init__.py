"""lucid.fft — discrete Fourier transform ops.

Mirrors the standard ``torch.fft`` surface: 18 transform variants
(``fft`` / ``ifft`` / ``fft2`` / ``ifft2`` / ``fftn`` / ``ifftn`` and the
``r``- and ``h``-prefixed counterparts) plus the four utility helpers
``fftfreq`` / ``rfftfreq`` / ``fftshift`` / ``ifftshift``.

Backend: every transform delegates to ``mlx::core::fft`` through the four
engine primitives ``_C_engine.fft.{fftn,ifftn,rfftn,irfftn}``.  MLX
itself wraps Apple Accelerate vDSP under the hood, so both ``Device::CPU``
and ``Device::GPU`` tensors land on the same fast CPU FFT kernel — see
``lucid/_C/ops/fft/_Detail.h`` for the carve-out rationale.

Normalisation: matches the standard convention.  ``norm='backward'`` (the
default) leaves ``fft`` unscaled and divides ``ifft`` by ``N``;
``norm='forward'`` swaps those; ``norm='ortho'`` divides both by
``sqrt(N)``.  MLX's transforms always produce the ``'backward'`` form,
so non-default norms are reached by post-multiplying the result.

Autograd: each base op is wrapped in a ``lucid.autograd.Function``
subclass.  The backward of every transform is the *dual* transform with
the *dual* normalisation (``backward ↔ forward``, ``ortho ↔ ortho``),
restricted to the original input size for ``r``/``ir`` variants.
"""

import math
from typing import Sequence, TYPE_CHECKING

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._types import DeviceLike, DTypeLike
from lucid.autograd.function import Function as _AutogradFunction
from lucid.autograd.function import FunctionCtx

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_fft = _C_engine.fft  # type: ignore[attr-defined]

# ── Normalisation ────────────────────────────────────────────────────────────

_VALID_NORMS = ("backward", "forward", "ortho")


def _check_norm(norm: str | None) -> str:
    if norm is None:
        return "backward"
    if norm not in _VALID_NORMS:
        raise ValueError(f"norm must be one of {_VALID_NORMS} or None, got {norm!r}")
    return norm


def _dual_norm(norm: str) -> str:
    if norm == "ortho":
        return "ortho"
    return "forward" if norm == "backward" else "backward"


def _scale_after_fft(N: int, norm: str) -> float:
    """Post-multiplier applied to MLX's fft (which is 'backward', i.e. unscaled)."""
    if norm == "backward":
        return 1.0
    if norm == "forward":
        return 1.0 / N
    return 1.0 / math.sqrt(N)  # ortho


def _scale_after_ifft(N: int, norm: str) -> float:
    """Post-multiplier applied to MLX's ifft (which already includes 1/N)."""
    if norm == "backward":
        return 1.0  # MLX already divides by N
    if norm == "forward":
        return float(N)  # cancel MLX's 1/N
    return math.sqrt(float(N))  # ortho: turn 1/N into 1/sqrt(N)


# ── Axis / shape canonicalisation ────────────────────────────────────────────


def _as_axis_list(
    dim: int | Sequence[int] | None, rank: int, default_all: bool
) -> list[int]:
    if dim is None:
        if default_all:
            return list(range(rank))
        return [-1]
    if isinstance(dim, int):
        return [dim]
    return [int(d) for d in dim]


def _as_size_list(s: int | Sequence[int] | None) -> list[int]:
    if s is None:
        return []
    if isinstance(s, int):
        return [int(s)]
    return [int(v) for v in s]


def _normalise_axes(axes: list[int], rank: int) -> list[int]:
    out: list[int] = []
    for ax in axes:
        a = ax + rank if ax < 0 else ax
        if a < 0 or a >= rank:
            raise IndexError(f"FFT axis {ax} out of range for rank-{rank} tensor")
        out.append(a)
    return out


def _validate_axes_and_s(axes: list[int], s: list[int], op: str) -> None:
    if s and len(s) != len(axes):
        raise ValueError(f"{op}: len(s)={len(s)} must match len(dim)={len(axes)}")


def _input_sizes_along_axes(x_shape: tuple[int, ...], axes: list[int]) -> list[int]:
    return [int(x_shape[a]) for a in axes]


def _transform_size(s: list[int], in_sizes: list[int]) -> int:
    """Total length N transformed = product of per-axis lengths."""
    sizes = s if s else in_sizes
    N = 1
    for v in sizes:
        N *= int(v)
    return N


# ── Engine wrappers (raw, no normalisation) ──────────────────────────────────


def _engine_fftn(x: Tensor, s: list[int], axes: list[int]) -> Tensor:
    return _wrap(_fft.fftn(_unwrap(x), s, axes))


def _engine_ifftn(x: Tensor, s: list[int], axes: list[int]) -> Tensor:
    return _wrap(_fft.ifftn(_unwrap(x), s, axes))


def _engine_rfftn(x: Tensor, s: list[int], axes: list[int]) -> Tensor:
    return _wrap(_fft.rfftn(_unwrap(x), s, axes))


def _engine_irfftn(x: Tensor, s: list[int], axes: list[int]) -> Tensor:
    return _wrap(_fft.irfftn(_unwrap(x), s, axes))


def _conj(x: Tensor) -> Tensor:
    return _wrap(_fft._conj_complex(_unwrap(x)))


def _scale(x: Tensor, s: float) -> Tensor:
    """Multiply by a real scalar.  Now a thin wrapper over ``Tensor * float``
    — both backends support ``full(C64)`` and ``mul(C64, C64)`` natively
    after the P2-B complex extension landed, so the previous private
    ``_C_engine.fft._scale`` helper is no longer required."""
    if s == 1.0:
        return x
    return x * s


# ── Autograd Function classes (one per base transform) ──────────────────────
#
# Backward derivation for a complex DFT y = F(x):
#   J[k,n] = exp(-2πi kn/N).  The conjugate transpose is J^H[n,k] = exp(+2πi kn/N).
#   Under norm='backward', grad_x = J^H · grad_y = N · ifft(grad_y, 'backward')
#                                                = ifft(grad_y, 'forward').
# More generally grad_x = ifft(grad_y, dual(norm)) for any norm.  The same
# argument with the role of fft/ifft swapped gives the ifft backward.
#
# For the real variants the saved ``in_size`` (length along the last
# transformed axis of the original real signal) is needed to reconstruct
# the correct backward shape because rfft → n//2+1 is not invertible from
# the gradient shape alone.


class _FftnAutograd(_AutogradFunction):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        s: list[int],
        axes: list[int],
        norm: str,
        N: int,
    ) -> Tensor:
        ctx.s = s
        ctx.axes = axes
        ctx.norm = norm
        ctx.N = N
        out = _engine_fftn(x, s, axes)
        scale = _scale_after_fft(N, norm)
        if scale != 1.0:
            out = _scale(out, scale)
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tensor:
        # grad_x = ifft(grad_out, dual(norm)) restricted to the input axis sizes.
        dual = _dual_norm(ctx.norm)
        g = _engine_ifftn(grad_out, ctx.s, ctx.axes)
        scale = _scale_after_ifft(ctx.N, dual)
        if scale != 1.0:
            g = _scale(g, scale)
        return g


class _IfftnAutograd(_AutogradFunction):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        s: list[int],
        axes: list[int],
        norm: str,
        N: int,
    ) -> Tensor:
        ctx.s = s
        ctx.axes = axes
        ctx.norm = norm
        ctx.N = N
        out = _engine_ifftn(x, s, axes)
        scale = _scale_after_ifft(N, norm)
        if scale != 1.0:
            out = _scale(out, scale)
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tensor:
        dual = _dual_norm(ctx.norm)
        g = _engine_fftn(grad_out, ctx.s, ctx.axes)
        scale = _scale_after_fft(ctx.N, dual)
        if scale != 1.0:
            g = _scale(g, scale)
        return g


class _RfftnAutograd(_AutogradFunction):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        s: list[int],
        axes: list[int],
        norm: str,
        N: int,
        in_sizes: list[int],
    ) -> Tensor:
        ctx.in_sizes = in_sizes  # full-length sizes along each transformed axis
        ctx.axes = axes
        ctx.norm = norm
        ctx.N = N
        out = _engine_rfftn(x, s, axes)
        scale = _scale_after_fft(N, norm)
        if scale != 1.0:
            out = _scale(out, scale)
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tensor:
        # rfft backward: the dual transform is irfft restricted to the original
        # full real length along each transformed axis.
        dual = _dual_norm(ctx.norm)
        g = _engine_irfftn(grad_out, ctx.in_sizes, ctx.axes)
        scale = _scale_after_ifft(ctx.N, dual)
        if scale != 1.0:
            g = _scale(g, scale)
        return g


class _IrfftnAutograd(_AutogradFunction):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        s: list[int],
        axes: list[int],
        norm: str,
        N: int,
        out_sizes: list[int],
    ) -> Tensor:
        ctx.axes = axes
        ctx.norm = norm
        ctx.N = N
        ctx.out_sizes = out_sizes  # n[i] for each axis (after expansion)
        out = _engine_irfftn(x, s, axes)
        scale = _scale_after_ifft(N, norm)
        if scale != 1.0:
            out = _scale(out, scale)
        return out

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tensor:
        # irfft backward: rfft of the real grad with the same size along the
        # last transformed axis, dual normalisation.
        dual = _dual_norm(ctx.norm)
        g = _engine_rfftn(grad_out, ctx.out_sizes, ctx.axes)
        scale = _scale_after_fft(ctx.N, dual)
        if scale != 1.0:
            g = _scale(g, scale)
        return g


# ── Public API: complex FFT (fft / fft2 / fftn) ──────────────────────────────


def fftn(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] | None = None,
    norm: str | None = None,
) -> Tensor:
    """N-dimensional discrete Fourier transform."""
    norm_v = _check_norm(norm)
    rank = input.ndim
    axes = _normalise_axes(_as_axis_list(dim, rank, default_all=True), rank)
    s_list = _as_size_list(s)
    _validate_axes_and_s(axes, s_list, "fftn")
    in_sizes = _input_sizes_along_axes(input.shape, axes)
    N = _transform_size(s_list, in_sizes)
    return _FftnAutograd.apply(input, s_list, axes, norm_v, N)


def ifftn(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] | None = None,
    norm: str | None = None,
) -> Tensor:
    """N-dimensional inverse discrete Fourier transform."""
    norm_v = _check_norm(norm)
    rank = input.ndim
    axes = _normalise_axes(_as_axis_list(dim, rank, default_all=True), rank)
    s_list = _as_size_list(s)
    _validate_axes_and_s(axes, s_list, "ifftn")
    in_sizes = _input_sizes_along_axes(input.shape, axes)
    N = _transform_size(s_list, in_sizes)
    return _IfftnAutograd.apply(input, s_list, axes, norm_v, N)


def fft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None) -> Tensor:
    """1-D discrete Fourier transform along ``dim``."""
    return fftn(input, s=None if n is None else [int(n)], dim=[int(dim)], norm=norm)


def ifft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None) -> Tensor:
    """1-D inverse discrete Fourier transform along ``dim``."""
    return ifftn(input, s=None if n is None else [int(n)], dim=[int(dim)], norm=norm)


def fft2(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    """2-D discrete Fourier transform over ``dim``."""
    return fftn(input, s=s, dim=list(dim), norm=norm)


def ifft2(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    """2-D inverse discrete Fourier transform over ``dim``."""
    return ifftn(input, s=s, dim=list(dim), norm=norm)


# ── Public API: real-input FFT (rfft / rfft2 / rfftn) ────────────────────────


def rfftn(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] | None = None,
    norm: str | None = None,
) -> Tensor:
    """N-dimensional FFT of a real input.  Output is C64 with last axis n//2+1."""
    norm_v = _check_norm(norm)
    rank = input.ndim
    axes = _normalise_axes(_as_axis_list(dim, rank, default_all=True), rank)
    s_list = _as_size_list(s)
    _validate_axes_and_s(axes, s_list, "rfftn")
    in_sizes = _input_sizes_along_axes(input.shape, axes)
    full_sizes: list[int] = list(s_list) if s_list else list(in_sizes)
    N = _transform_size(s_list, in_sizes)
    return _RfftnAutograd.apply(input, s_list, axes, norm_v, N, full_sizes)


def irfftn(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] | None = None,
    norm: str | None = None,
) -> Tensor:
    """N-dimensional inverse of :func:`rfftn`.  Output is real."""
    norm_v = _check_norm(norm)
    rank = input.ndim
    axes = _normalise_axes(_as_axis_list(dim, rank, default_all=True), rank)
    s_list = _as_size_list(s)
    _validate_axes_and_s(axes, s_list, "irfftn")
    out_sizes: list[int]
    if s_list:
        out_sizes = list(s_list)
    else:
        # Default: last axis becomes 2*(in_last - 1); others keep size.
        out_sizes = _input_sizes_along_axes(input.shape, axes)
        out_sizes[-1] = (out_sizes[-1] - 1) * 2
    N = 1
    for v in out_sizes:
        N *= int(v)
    return _IrfftnAutograd.apply(input, s_list, axes, norm_v, N, out_sizes)


def rfft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None) -> Tensor:
    return rfftn(input, s=None if n is None else [int(n)], dim=[int(dim)], norm=norm)


def irfft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None) -> Tensor:
    return irfftn(input, s=None if n is None else [int(n)], dim=[int(dim)], norm=norm)


def rfft2(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    return rfftn(input, s=s, dim=list(dim), norm=norm)


def irfft2(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    return irfftn(input, s=s, dim=list(dim), norm=norm)


# ── Public API: Hermitian FFT (hfft / hfft2 / hfftn) ─────────────────────────
#
# Identity used (matches numpy & the standard reference framework):
#     hfft(x, n, norm)  =  irfft(conj(x), n, norm=dual(norm))
#     ihfft(x, n, norm) =  conj(rfft(x, n, norm=dual(norm)))
# where ``dual`` swaps 'backward' ↔ 'forward' and leaves 'ortho' fixed.
# This works because hfft uses the FFT sign convention on a Hermitian
# (conjugate-symmetric) signal — flipping conj on the input flips the
# sign in the exponent, turning the inverse-real transform back into a
# forward-real one.


def hfftn(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] | None = None,
    norm: str | None = None,
) -> Tensor:
    norm_v = _check_norm(norm)
    return irfftn(_conj(input), s=s, dim=dim, norm=_dual_norm(norm_v))


def ihfftn(
    input: Tensor,
    s: int | Sequence[int] | None = None,
    dim: int | Sequence[int] | None = None,
    norm: str | None = None,
) -> Tensor:
    norm_v = _check_norm(norm)
    return _conj(rfftn(input, s=s, dim=dim, norm=_dual_norm(norm_v)))


def hfft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None) -> Tensor:
    return hfftn(input, s=None if n is None else [int(n)], dim=[int(dim)], norm=norm)


def ihfft(input: Tensor, n: int | None = None, dim: int = -1, norm: str | None = None) -> Tensor:
    return ihfftn(input, s=None if n is None else [int(n)], dim=[int(dim)], norm=norm)


def hfft2(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    return hfftn(input, s=s, dim=list(dim), norm=norm)


def ihfft2(
    input: Tensor,
    s: Sequence[int] | None = None,
    dim: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> Tensor:
    return ihfftn(input, s=s, dim=list(dim), norm=norm)


# ── Public API: utility functions ────────────────────────────────────────────


def fftfreq(
    n: int,
    d: float = 1.0,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    """Sample frequencies for ``fft``: ``[0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)``.

    Implemented as a closed-form expression over ``lucid.arange`` — no
    composite numpy needed.  ``n`` may be even or odd; the negative-frequency
    half occupies indices from ``(n+1)//2`` onward.
    """
    n = int(n)
    if n <= 0:
        raise ValueError(f"fftfreq requires n > 0, got {n}")
    half = (n - 1) // 2 + 1  # number of non-negative frequency bins
    pos = lucid.arange(0.0, float(half), 1.0, dtype=dtype, device=device)
    neg = lucid.arange(float(-(n // 2)), 0.0, 1.0, dtype=dtype, device=device)
    full = lucid.cat([pos, neg], 0)
    return full * (1.0 / (float(d) * n))


def rfftfreq(
    n: int,
    d: float = 1.0,
    *,
    dtype: DTypeLike = None,
    device: DeviceLike = None,
) -> Tensor:
    """Sample frequencies for ``rfft``: ``[0, 1, ..., n/2] / (d*n)``."""
    n = int(n)
    if n <= 0:
        raise ValueError(f"rfftfreq requires n > 0, got {n}")
    bins = lucid.arange(0.0, float(n // 2 + 1), 1.0, dtype=dtype, device=device)
    return bins * (1.0 / (float(d) * n))


def fftshift(input: Tensor, dim: int | Sequence[int] | None = None) -> Tensor:
    """Shift the zero-frequency bin to the centre.  Composes ``lucid.roll``."""
    rank = input.ndim
    dims = _normalise_axes(_as_axis_list(dim, rank, default_all=True), rank)
    shifts = [int(input.shape[a]) // 2 for a in dims]
    return lucid.roll(input, shifts=shifts, dims=dims)


def ifftshift(input: Tensor, dim: int | Sequence[int] | None = None) -> Tensor:
    """Inverse of :func:`fftshift`.  Composes ``lucid.roll`` in the other direction."""
    rank = input.ndim
    dims = _normalise_axes(_as_axis_list(dim, rank, default_all=True), rank)
    shifts = [-(int(input.shape[a]) // 2) for a in dims]
    return lucid.roll(input, shifts=shifts, dims=dims)


__all__ = [
    "fft",
    "ifft",
    "fft2",
    "ifft2",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfft2",
    "irfft2",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "hfft2",
    "ihfft2",
    "hfftn",
    "ihfftn",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]
