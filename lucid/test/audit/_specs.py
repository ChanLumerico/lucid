"""How to call each op — the part that decides whether a census is one.

The first hand-rolled sweep of this framework could only build ``f(x)``
and ``f(x, y)`` over a single small tensor, so 335 of 487 names were never
evaluated: every convolution, every pooling layer, every loss,
``batch_norm``, attention.  Coverage was 29.5% of the op surface and the
gap was invisible in the output.  This module is the fix, and the reason
:class:`~lucid.test.audit._result.Status` has a SKIP that gets counted
and listed rather than dropped.

A spec yields **candidate invocations in order**, not one.  Signatures
differ in ways that are tedious to encode exactly — ``dim`` versus
``axis``, ``keepdim`` versus ``keepdims``, a kernel size that may be an
int or a tuple — and a spec that guesses wrong should degrade to trying
the next form rather than reporting a defect that is really a mis-call.
When every candidate fails, the symbol is reported SKIP with the last
error, which is honest and is what tells you a spec needs adding.
"""

import re
from typing import TYPE_CHECKING, Any

import numpy as np

import lucid
import lucid.fft
import lucid.linalg
from lucid.test.audit import _probe

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class Call:
    """One concrete invocation of an op.

    Attributes
    ----------
    args : list
        Positional arguments, already built.
    kwargs : dict
        Keyword arguments.
    primary : int
        Index into ``args`` of the tensor the numeric axes differentiate
        and perturb.  Not always zero — ``embedding`` differentiates its
        weight, not its indices.
    note : str
        How this invocation was chosen, for the SKIP message.
    """

    __slots__ = ("args", "kwargs", "primary", "note")

    def __init__(
        self,
        args: list[Any],
        kwargs: dict[str, Any] | None = None,
        primary: int = 0,
        note: str = "",
    ) -> None:
        self.args = args
        self.kwargs = kwargs or {}
        self.primary = primary
        self.note = note

    def with_primary(self, array: np.ndarray) -> "Call":
        """A copy whose differentiated argument is replaced by ``array``."""
        args = list(self.args)
        args[self.primary] = _probe.as_f64(array)
        return Call(args, dict(self.kwargs), self.primary, self.note)

    @property
    def base(self) -> np.ndarray:
        """The differentiated argument, as an array.

        Raises ``TypeError`` — never ``IndexError`` — when there is no
        such argument.  A keyword-only op like ``affine_matrix(*, cx, cy)``
        has an empty ``args``, and the numeric axes already read a
        TypeError here as "nothing to differentiate, skip"; an IndexError
        would escape as a harness ERROR instead.
        """
        if not 0 <= self.primary < len(self.args):
            raise TypeError("no positional argument to differentiate")
        got = _probe.to_numpy(self.args[self.primary])
        if got is None:
            raise TypeError("primary argument is not a tensor")
        return np.asarray(got, dtype=np.float64)


# ── shape vocabulary ─────────────────────────────────────────────────────────
# Small on purpose: every numeric axis costs one forward per element of
# the differentiated tensor, so the probes stay tiny while still having
# more than one channel, more than one batch element and a spatial extent
# that survives a stride-2 convolution.

_N, _CIN, _COUT = 2, 3, 4
_L, _H, _W = 8, 6, 6
_D = 4


def _f(shape: "tuple[int, ...]", domain: str = "moderate") -> Any:
    return _probe.as_f64(_probe.sample(domain, shape))


def _int(shape: "tuple[int, ...]", high: int) -> Any:
    return _probe.as_int(_probe.rng(_probe.SEED_B).integers(0, high, shape))


def _spatial(rank: int) -> "tuple[int, ...]":
    return {1: (_L,), 2: (_H, _W), 3: (_D, _H, _W)}[rank]


def _rank_of(name: str) -> int:
    match = re.search(r"([123])d$", name)
    return int(match.group(1)) if match else 2


# ── family builders ──────────────────────────────────────────────────────────


def _conv(name: str, domain: str) -> "Iterator[Call]":
    rank = _rank_of(name)
    spatial = _spatial(rank)
    k = (3,) * rank
    transposed = "transpose" in name
    w_shape = (_CIN, _COUT, *k) if transposed else (_COUT, _CIN, *k)
    x = _f((_N, _CIN, *spatial), domain)
    w = _f(w_shape, domain)
    b = _f((_COUT,), domain)
    yield Call([x, w, b], {"stride": 1, "padding": 1}, 0, "conv(x, weight, bias)")
    yield Call([x, w, b], {}, 0, "conv(x, weight, bias) defaults")
    yield Call([x, w], {}, 0, "conv(x, weight)")


def _pool(name: str, domain: str) -> "Iterator[Call]":
    rank = _rank_of(name)
    x = _f((_N, _CIN, *_spatial(rank)), domain)
    yield Call([x, 2], {}, 0, "pool(x, kernel_size=2)")
    yield Call([x], {"kernel_size": 2}, 0, "pool(x, kernel_size=2) kw")
    yield Call([x, (2,) * rank], {}, 0, "pool(x, kernel_size tuple)")


def _adaptive_pool(name: str, domain: str) -> "Iterator[Call]":
    rank = _rank_of(name)
    x = _f((_N, _CIN, *_spatial(rank)), domain)
    out = (2,) * rank if rank > 1 else 2
    yield Call([x, out], {}, 0, "adaptive_pool(x, output_size)")
    yield Call([x], {"output_size": out}, 0, "adaptive_pool kw")


def _fractional_pool(name: str, domain: str) -> "Iterator[Call]":
    rank = _rank_of(name)
    x = _f((_N, _CIN, *_spatial(rank)), domain)
    yield Call([x, 2], {"output_ratio": 0.5}, 0, "fractional_pool(x, k, output_ratio)")
    yield Call(
        [x, 2], {"output_size": (2,) * rank}, 0, "fractional_pool(x, k, output_size)"
    )


def _norm(name: str, domain: str) -> "Iterator[Call]":
    x = _f((_N, _CIN, _H, _W), domain)
    weight, bias = _f((_CIN,), "positive"), _f((_CIN,), domain)
    if "batch" in name:
        mean, var = _f((_CIN,), domain), _f((_CIN,), "positive")
        yield Call(
            [x, mean, var, weight, bias], {"training": True}, 0, "batch_norm full"
        )
        yield Call(
            [x, None, None, weight, bias], {"training": True}, 0, "batch_norm no stats"
        )
        yield Call([x, mean, var], {}, 0, "batch_norm(x, mean, var)")
    elif "layer" in name:
        flat = _f((_N, _CIN * 2), domain)
        yield Call(
            [flat, (_CIN * 2,), _f((_CIN * 2,), "positive"), _f((_CIN * 2,), domain)],
            {},
            0,
            "layer_norm(x, shape, w, b)",
        )
        yield Call([flat, (_CIN * 2,)], {}, 0, "layer_norm(x, shape)")
    elif "group" in name:
        yield Call([x, 1, weight, bias], {}, 0, "group_norm(x, groups, w, b)")
        yield Call([x, 1], {}, 0, "group_norm(x, groups)")
    elif "instance" in name:
        yield Call([x], {}, 0, "instance_norm(x)")
        yield Call([x, None, None, weight, bias], {}, 0, "instance_norm full")
    elif "local_response" in name:
        yield Call([x, 3], {}, 0, "local_response_norm(x, size)")
    else:
        yield Call([x], {}, 0, "norm(x)")


#: Losses grouped by what their target has to look like.  Getting this
#: wrong is not a defect in the loss, so each group is tried in turn.
_CLASS_LOSSES = ("cross_entropy", "nll_loss", "multi_margin", "multilabel_margin")
_PROB_LOSSES = ("binary_cross_entropy", "bce", "kl_div", "poisson_nll")
_PAIR_LOSSES = ("cosine_embedding", "margin_ranking", "hinge_embedding", "triplet")


def _loss(name: str, domain: str) -> "Iterator[Call]":
    classes = 4
    logits = _f((_N, classes), "moderate")
    target_idx = _int((_N,), classes)
    same = _f((_N, classes), domain)
    probs = _f((_N, classes), "small_pos")

    if any(k in name for k in _CLASS_LOSSES):
        yield Call([logits, target_idx], {}, 0, "loss(logits, class index)")
    if any(k in name for k in _PROB_LOSSES):
        yield Call([probs, _f((_N, classes), "small_pos")], {}, 0, "loss(prob, prob)")
    if any(k in name for k in _PAIR_LOSSES):
        sign = _probe.as_f64(np.array([1.0, -1.0]))
        yield Call([logits, same, sign], {}, 0, "loss(a, b, target sign)")
        yield Call([logits, same, logits], {}, 0, "loss(anchor, positive, negative)")
    # The elementwise regressions, and the fallback for anything unmatched.
    yield Call([logits, same], {}, 0, "loss(input, target) same shape")
    yield Call([logits, target_idx], {}, 0, "loss(logits, class index)")


def _embedding(name: str, domain: str) -> "Iterator[Call]":
    vocab, dim = 6, 4
    weight = _f((vocab, dim), domain)
    idx = _int((_N, 3), vocab)
    if "bag" in name:
        offsets = _probe.as_int(np.array([0, 3]))
        yield Call(
            [idx.reshape(-1), weight, offsets],
            {},
            1,
            "embedding_bag(idx, weight, offsets)",
        )
        yield Call([idx, weight], {}, 1, "embedding_bag(idx2d, weight)")
    else:
        yield Call([idx, weight], {}, 1, "embedding(idx, weight)")


def _attention(name: str, domain: str) -> "Iterator[Call]":
    heads, seq, head_dim = 2, 4, 4
    q = _f((_N, heads, seq, head_dim), domain)
    k = _f((_N, heads, seq, head_dim), domain)
    v = _f((_N, heads, seq, head_dim), domain)
    yield Call([q, k, v], {}, 0, "sdpa(q, k, v)")
    yield Call([q, k, v], {"is_causal": True}, 0, "sdpa causal")


def _resample(name: str, domain: str) -> "Iterator[Call]":
    x = _f((_N, _CIN, _H, _W), domain)
    if "grid_sample" in name:
        grid = _f((_N, _H, _W, 2), "unit")
        yield Call([x, grid], {}, 0, "grid_sample(x, grid)")
    elif "affine_grid" in name:
        theta = _f((_N, 2, 3), domain)
        yield Call([theta, (_N, _CIN, _H, _W)], {}, 0, "affine_grid(theta, size)")
    elif "interpolate" in name or "upsample" in name:
        yield Call(
            [x], {"scale_factor": 2.0, "mode": "nearest"}, 0, "interpolate nearest"
        )
        yield Call([x], {"size": (_H * 2, _W * 2)}, 0, "interpolate size")
    elif "pixel_shuffle" in name:
        yield Call([_f((_N, 4, _H, _W), domain), 2], {}, 0, "pixel_shuffle(x, 2)")
    elif "pixel_unshuffle" in name:
        yield Call([x, 2], {}, 0, "pixel_unshuffle(x, 2)")
    elif "channel_shuffle" in name:
        yield Call([x, 3], {}, 0, "channel_shuffle(x, groups)")
    elif name.endswith("unfold"):
        yield Call([x, 3], {"padding": 1}, 0, "unfold(x, kernel)")
    elif name.endswith("fold"):
        cols = _f((_N, _CIN * 9, _H * _W), domain)
        yield Call([cols, (_H, _W), 3], {"padding": 1}, 0, "fold(cols, size, kernel)")
    elif name.endswith("pad"):
        yield Call([x, (1, 1, 1, 1)], {}, 0, "pad(x, padding)")
        yield Call([x, (1, 1)], {}, 0, "pad(x, padding pair)")


def _linalg(name: str, domain: str) -> "Iterator[Call]":
    n = 4
    square = _probe.rng(_probe.SEED_X).standard_normal((n, n)) * 0.4 + np.eye(n) * 2.0
    rect = _probe.sample(domain, (n, 3))
    psd = square @ square.T + np.eye(n) * 0.5
    if any(k in name for k in ("cholesky", "eigh", "inv_psd")):
        yield Call([_probe.as_f64(psd)], {}, 0, "linalg(PSD)")
    if any(k in name for k in ("solve", "lstsq")):
        yield Call(
            [_probe.as_f64(square), _probe.as_f64(_probe.sample(domain, (n, 2)))],
            {},
            0,
            "linalg(A, B)",
        )
    if "matrix_power" in name:
        yield Call([_probe.as_f64(square), 2], {}, 0, "matrix_power(A, 2)")
    if any(k in name for k in ("matmul", "dot", "outer", "kron", "cross")):
        yield Call(
            [_probe.as_f64(rect), _probe.as_f64(rect.T.copy())], {}, 0, "linalg(A, B)"
        )
    yield Call([_probe.as_f64(square)], {}, 0, "linalg(square)")
    yield Call([_probe.as_f64(rect)], {}, 0, "linalg(rectangular)")


def _fft(name: str, domain: str) -> "Iterator[Call]":
    x = _probe.as_f64(_probe.sample(domain, (_N, 8)))
    if name.endswith(("shift",)):
        yield Call([x], {}, 0, "fftshift(x)")
        return
    if "2" in name:
        yield Call([_probe.as_f64(_probe.sample(domain, (4, 8)))], {}, 0, "fft2(x)")
    yield Call([x], {}, 0, "fft(x)")
    yield Call([x], {"n": 8}, 0, "fft(x, n)")


def _einops(name: str, domain: str) -> "Iterator[Call]":
    x = _f((_N, _CIN, _H, _W), domain)
    if "rearrange" in name:
        yield Call([x, "b c h w -> b h w c"], {}, 0, "rearrange")
    elif "reduce" in name:
        yield Call([x, "b c h w -> b c", "mean"], {}, 0, "reduce")
    elif "repeat" in name:
        yield Call(
            [
                x,
                "b c h w -> b c h w r",
            ],
            {"r": 2},
            0,
            "repeat",
        )
    elif "pack" in name or "unpack" in name:
        yield Call([[x, x], "b c h *"], {}, 0, "pack")


def _reduction(name: str, domain: str) -> "Iterator[Call]":
    x = _f(_probe.SHAPE, domain)
    yield Call([x], {}, 0, "reduce(x)")
    yield Call([x], {"dim": -1}, 0, "reduce(x, dim=-1)")
    yield Call([x, -1], {}, 0, "reduce(x, -1)")


def _binary(name: str, domain: str) -> "Iterator[Call]":
    a = _f(_probe.SHAPE, domain)
    b = _f(_probe.SHAPE, "positive")
    yield Call([a, b], {}, 0, "binary(a, b)")


def _unary(name: str, domain: str) -> "Iterator[Call]":
    yield Call([_f(_probe.SHAPE, domain)], {}, 0, "unary(x)")


# ── the groups a first depth probe could not call ────────────────────────────
# 189 of 716 symbols had no invocation the harness could build.  They were
# not scattered: every one belonged to a family with a shape convention,
# and each block below closes a whole family.  The numeric axes ride on
# this, so a symbol that cannot be called here is a symbol whose gradient
# is never checked.


def _ternary(name: str, domain: str) -> "Iterator[Call]":
    """``add*`` fused forms: an accumulator plus two operands."""
    v = _f((_COUT,), domain)
    m = _f((_COUT, _CIN), domain)
    n = _f((_CIN, _COUT), domain)
    batch_a = _f((_N, _COUT, _CIN), domain)
    batch_b = _f((_N, _CIN, _COUT), domain)
    square = _f((_COUT, _COUT), domain)
    same = _f(_probe.SHAPE, domain)
    if name in ("addbmm",):
        yield Call([square, batch_a, batch_b], {}, 0, "addbmm(bias, batch1, batch2)")
    if name in ("baddbmm",):
        yield Call([_f((_N, _COUT, _COUT), domain), batch_a, batch_b], {}, 0, "baddbmm")
    if name in ("addmm",):
        yield Call([square, m, n], {}, 0, "addmm(bias, mat1, mat2)")
    if name in ("addmv",):
        yield Call([v, m, _f((_CIN,), domain)], {}, 0, "addmv(bias, mat, vec)")
    if name in ("addr",):
        yield Call([m, v, _f((_CIN,), domain)], {}, 0, "addr(mat, vec1, vec2)")
    if name in ("addcmul", "addcdiv"):
        yield Call(
            [same, same, _f(_probe.SHAPE, "positive")],
            {"value": 0.5},
            0,
            "addc*(t, a, b)",
        )
        yield Call([same, same, _f(_probe.SHAPE, "positive")], {}, 0, "addc*(t, a, b)")
    if name == "lerp":
        yield Call([same, _f(_probe.SHAPE, domain), 0.3], {}, 0, "lerp(a, b, weight)")


def _bitwise(name: str, domain: str) -> "Iterator[Call]":
    """Integer-only ops.  A float probe is rejected before the op is reached."""
    a = _probe.as_int(_probe.rng(1).integers(1, 30, _probe.SHAPE))
    b = _probe.as_int(_probe.rng(2).integers(1, 6, _probe.SHAPE))
    if "not" in name or "invert" in name:
        yield Call([a], {}, 0, "bitwise_not(int)")
        return
    yield Call([a, b], {}, 0, "bitwise(int, int)")
    yield Call([a, 2], {}, 0, "bitwise(int, scalar)")


def _factory(name: str, domain: str) -> "Iterator[Call]":
    """Constructors take a shape or a range, never an input tensor.

    ``primary`` still points at argument zero, so the numeric axes will
    report SKIP — correctly, since there is no input to differentiate.
    """
    shape = (2, 3)
    if name in ("arange",):
        yield Call([0.0, 6.0, 1.0], {}, 0, "arange(start, stop, step)")
        yield Call([6], {}, 0, "arange(n)")
    elif name in ("linspace", "logspace"):
        yield Call([0.0, 1.0, 5], {}, 0, "linspace(start, stop, num)")
    elif name in ("eye",):
        yield Call([3], {}, 0, "eye(n)")
        yield Call([3, 4], {}, 0, "eye(n, m)")
    elif name in ("full", "new_full"):
        yield Call([shape, 0.5], {}, 0, "full(shape, value)")
    elif name in ("randint",):
        yield Call([0, 6, shape], {}, 0, "randint(low, high, shape)")
    elif name in ("randperm",):
        yield Call([6], {}, 0, "randperm(n)")
    elif name in ("meshgrid",):
        yield Call([_f((3,), domain), _f((4,), domain)], {}, 0, "meshgrid(a, b)")
    elif name in ("bernoulli",):
        probs = _probe.as_f32(_probe.rng(9).uniform(0.2, 0.8, _probe.SHAPE))
        yield Call([probs], {}, 0, "bernoulli(probs)")
        yield Call([_probe.SHAPE, 0.5], {}, 0, "bernoulli(shape, p)")
        yield Call([0.5], {"size": _probe.SHAPE}, 0, "bernoulli(p, size=)")
    elif name in ("normal",):
        yield Call([0.0, 1.0, shape], {}, 0, "normal(mean, std, shape)")
        yield Call(
            [_f(_probe.SHAPE, domain), _f(_probe.SHAPE, "positive")],
            {},
            0,
            "normal(mu, sigma)",
        )
    elif name in ("cartesian_prod",):
        yield Call(
            [_f((3,), domain), _f((2,), domain)], {}, 0, "cartesian_prod(1-D, 1-D)"
        )
    elif name in ("vander",):
        yield Call([_f((4,), "positive")], {}, 0, "vander(1-D)")
    else:
        yield Call([shape], {}, 0, "factory(shape)")
        yield Call([2, 3], {}, 0, "factory(*shape)")


def _indexing(name: str, domain: str) -> "Iterator[Call]":
    """Gather / scatter shapes, which need an index tensor of the right rank."""
    x = _f((4, 5), domain)
    idx_full = _probe.as_int(_probe.rng(3).integers(0, 5, (4, 5)))
    idx_row = _probe.as_int(_probe.rng(3).integers(0, 4, (3,)))
    src = _f((4, 5), domain)
    if name in ("gather", "take_along_dim", "take_along_axis"):
        yield Call([x, 1, idx_full], {}, 0, "gather(x, dim, index)")
        yield Call([x, idx_full], {"dim": 1}, 0, "gather(x, index, dim=)")
    elif name in ("scatter", "scatter_add", "scatter_reduce"):
        yield Call([x, 1, idx_full, src], {}, 0, "scatter(x, dim, index, src)")
    elif name in ("index_select",):
        yield Call([x, 0, idx_row], {}, 0, "index_select(x, dim, index)")
    elif name in ("take",):
        yield Call(
            [x, _probe.as_int(_probe.rng(3).integers(0, 20, (5,)))],
            {},
            0,
            "take(x, flat idx)",
        )
    elif name in ("masked_fill", "masked_fill_"):
        mask = lucid.tensor((_probe.rng(4).random((4, 5)) > 0.5))
        yield Call([x, mask, 0.0], {}, 0, "masked_fill(x, mask, value)")
    elif name in ("searchsorted", "bucketize"):
        sorted_1d = _probe.as_f64(np.linspace(0.0, 1.0, 6))
        values = _probe.as_f64(_probe.rng(5).uniform(0.0, 1.0, (4,)))
        yield Call([sorted_1d, values], {}, 0, "searchsorted(sorted, values)")
        yield Call([values, sorted_1d], {}, 0, "bucketize(values, boundaries)")
    elif name in ("narrow",):
        yield Call([x, 1, 1, 2], {}, 0, "narrow(x, dim, start, length)")
    elif name in ("where",):
        cond = lucid.tensor((_probe.rng(6).random((4, 5)) > 0.5))
        yield Call([cond, x, src], {}, 1, "where(cond, a, b)")


def _shape_with_args(name: str, domain: str) -> "Iterator[Call]":
    """Reshapes and permutations whose target has to be spelled out."""
    x = _f((2, 3, 4), domain)
    flat = _f((2, 3), domain)
    if name in ("broadcast_to",):
        yield Call([_f((1, 3), domain), (2, 3)], {}, 0, "broadcast_to(x, shape)")
    elif name in ("expand",):
        yield Call([_f((1, 3), domain), (2, 3)], {}, 0, "expand(x, shape)")
        yield Call([_f((1, 3), domain), 2, 3], {}, 0, "expand(x, *shape)")
    elif name in ("permute",):
        yield Call([x, (2, 0, 1)], {}, 0, "permute(x, dims)")
        yield Call([x, 2, 0, 1], {}, 0, "permute(x, *dims)")
    elif name in ("moveaxis", "movedim", "swapaxes", "swapdims", "transpose"):
        yield Call([x, 0, 2], {}, 0, "moveaxis(x, src, dst)")
    elif name in ("tile", "repeat"):
        yield Call([flat, (2, 2)], {}, 0, "tile(x, reps)")
        yield Call([flat, 2, 2], {}, 0, "tile(x, *reps)")
    elif name in ("roll",):
        # The binding takes sequences, not bare ints.
        yield Call([flat, (1,), (0,)], {}, 0, "roll(x, shifts, dims)")
        yield Call([flat, (1, 1), (0, 1)], {}, 0, "roll(x, shifts, dims)")
        yield Call([flat, 1], {}, 0, "roll(x, shift)")
    elif name in ("unflatten",):
        yield Call([_f((2, 6), domain), 1, (2, 3)], {}, 0, "unflatten(x, dim, sizes)")
    elif name in ("unfold",):
        yield Call([x, 1, 2, 1], {}, 0, "unfold(x, dim, size, step)")
    elif name in ("dsplit", "hsplit", "vsplit", "split", "chunk", "tensor_split"):
        yield Call([_f((2, 3, 4), domain), 2], {}, 0, "split(x, n)")
    elif name in ("narrow",):
        yield Call([flat, 1, 0, 2], {}, 0, "narrow(x, dim, start, length)")
    elif name in ("view", "reshape"):
        yield Call([flat, (3, 2)], {}, 0, "reshape(x, shape)")
    elif name in ("flatten", "ravel"):
        yield Call([x], {}, 0, "flatten(x)")


def _matmul(name: str, domain: str) -> "Iterator[Call]":
    """Products, each with its own rank convention."""
    vec = _f((4,), domain)
    mat = _f((3, 4), domain)
    other = _f((4, 3), domain)
    batch_a = _f((2, 3, 4), domain)
    batch_b = _f((2, 4, 3), domain)
    if name in ("bmm", "baddbmm"):
        yield Call([batch_a, batch_b], {}, 0, "bmm(batched, batched)")
    elif name in ("mm",):
        yield Call([mat, other], {}, 0, "mm(a, b)")
    elif name in ("dot", "vdot", "inner"):
        yield Call([vec, _f((4,), domain)], {}, 0, "dot(1-D, 1-D)")
    elif name in ("outer", "ger"):
        yield Call([vec, _f((3,), domain)], {}, 0, "outer(1-D, 1-D)")
    elif name in ("multi_dot",):
        yield Call([[mat, other, mat]], {}, 0, "multi_dot([a, b, c])")
    elif name in ("einsum",):
        yield Call(["ij,jk->ik", mat, other], {}, 1, "einsum(subscripts, a, b)")
    elif name in ("float_power",):
        yield Call([_f(_probe.SHAPE, "positive"), 2.0], {}, 0, "float_power(x, p)")
    else:
        yield Call([mat, other], {}, 0, "matmul(a, b)")


def _linalg_extra(name: str, domain: str) -> "Iterator[Call]":
    """Decompositions, each with the matrix property it requires."""
    n = 4
    gen = _probe.rng(_probe.SEED_X)
    square = gen.standard_normal((n, n)) * 0.35 + np.eye(n) * 2.5
    psd = square @ square.T + np.eye(n) * 0.75
    sym = (square + square.T) / 2.0
    rhs = gen.standard_normal((n, 2))
    upper = np.triu(square)
    if name in ("cholesky", "cholesky_ex"):
        yield Call([_probe.as_f64(psd)], {}, 0, "cholesky(PSD)")
    elif name in ("eigh", "eigvalsh"):
        yield Call([_probe.as_f64(sym)], {}, 0, "eigh(symmetric)")
    elif name in ("eig", "eigvals"):
        yield Call([_probe.as_f64(square)], {}, 0, "eig(square)")
    elif name in (
        "inv",
        "det",
        "slogdet",
        "matrix_exp",
        "lu",
        "lu_factor",
        "ldl_factor",
    ):
        yield Call([_probe.as_f64(square)], {}, 0, "linalg(well-conditioned square)")
    elif name in ("matrix_power",):
        yield Call([_probe.as_f64(square), 2], {}, 0, "matrix_power(A, 2)")
    elif name in ("solve", "lstsq"):
        yield Call([_probe.as_f64(square), _probe.as_f64(rhs)], {}, 0, "solve(A, B)")
    elif name in ("solve_triangular",):
        yield Call(
            [_probe.as_f64(upper), _probe.as_f64(rhs)],
            {"upper": True},
            0,
            "solve_triangular",
        )
    elif name in ("lu_solve", "ldl_solve"):
        yield Call(
            [_probe.as_f64(square), _probe.as_f64(rhs)], {}, 0, "lu_solve(LU, B)"
        )
    elif name in ("vander",):
        yield Call([_probe.as_f64(gen.uniform(0.5, 1.5, (4,)))], {}, 0, "vander(1-D)")


def _fft_full(name: str, domain: str) -> "Iterator[Call]":
    """Every transform in the family, at the precision the engine accepts.

    The transforms reject float64 outright (``fftn requires F16/F32/C64``),
    which is why a float64 probe reported all twenty of them as
    uncallable.  The inverse and Hermitian forms want a *spectrum*, so
    they are fed one produced by their own forward partner rather than
    raw samples.
    """
    real_1d = _probe.as_f32(_probe.sample(domain, (8,)))
    real_2d = _probe.as_f32(_probe.sample(domain, (4, 8)))
    if name in ("fftfreq", "rfftfreq"):
        yield Call([8], {}, 0, "fftfreq(n)")
        yield Call([8, 0.5], {}, 0, "fftfreq(n, d)")
        return
    if name in ("fftshift", "ifftshift"):
        yield Call([real_1d], {}, 0, "fftshift(x)")
        return

    rank = 2 if name.endswith("2") else (3 if name.endswith("n") else 1)
    source = real_2d if rank >= 2 else real_1d
    spectrum = None
    forward_name = {1: "fft", 2: "fft2", 3: "fftn"}[rank]
    forward = getattr(lucid.fft, forward_name, None)
    if callable(forward):
        try:
            spectrum = forward(source)
        except Exception:  # noqa: BLE001
            spectrum = None

    # hfft / ihfft and the inverse transforms consume a complex spectrum.
    if spectrum is not None and (name.startswith("i") or "hfft" in name):
        yield Call([spectrum], {}, 0, f"{name}(spectrum)")
    yield Call([source], {}, 0, f"{name}(real)")
    if spectrum is not None:
        yield Call([spectrum], {}, 0, f"{name}(spectrum)")
    if rank == 1:
        yield Call([source], {"n": 8}, 0, f"{name}(x, n=8)")


def _complex(name: str, domain: str) -> "Iterator[Call]":
    """Ops that only mean anything on a complex tensor."""
    real = _probe.as_f64(_probe.sample(domain, (2, 4)))
    pair = _probe.as_f64(_probe.sample(domain, (2, 4)))
    built = None
    maker = getattr(lucid, "complex", None)
    if callable(maker):
        try:
            built = maker(real, pair)
        except Exception:  # noqa: BLE001
            built = None
    if name == "view_as_complex":
        yield Call(
            [_probe.as_f64(_probe.sample(domain, (2, 4, 2)))],
            {},
            0,
            "view_as_complex(...,2)",
        )
        return
    if built is not None:
        yield Call([built], {}, 0, f"{name}(complex)")
    yield Call([real], {}, 0, f"{name}(real)")


def _dtype_util(name: str, domain: str) -> "Iterator[Call]":
    """Type-system helpers: they take dtypes, not tensors."""
    if name in ("can_cast", "promote_types"):
        yield Call([lucid.float32, lucid.float64], {}, 0, f"{name}(dtype, dtype)")
    elif name in ("result_type",):
        yield Call(
            [_f(_probe.SHAPE, domain), _f(_probe.SHAPE, domain)],
            {},
            0,
            "result_type(a, b)",
        )
    elif name in ("type", "astype"):
        # ``Tensor.type`` takes the *name*, not the dtype object.
        yield Call([_f(_probe.SHAPE, domain), "float32"], {}, 0, "type(x, 'float32')")
        yield Call([_f(_probe.SHAPE, domain), lucid.float32], {}, 0, "astype(x, dtype)")


def _nn_leftover(name: str, domain: str) -> "Iterator[Call]":
    """The functional entries with signatures no other family covers."""
    if name in ("lp_pool1d", "lp_pool2d", "lp_pool3d"):
        rank = _rank_of(name)
        yield Call(
            [_f((_N, _CIN, *_spatial(rank)), "positive"), 2.0, 2],
            {},
            0,
            "lp_pool(x, p, k)",
        )
    elif name in ("max_unpool1d", "max_unpool2d", "max_unpool3d"):
        rank = _rank_of(name)
        spatial = _spatial(rank)
        pooled = tuple(v // 2 for v in spatial)
        x = _f((_N, _CIN, *pooled), domain)
        idx = _probe.as_int(
            _probe.rng(7).integers(0, int(np.prod(spatial)), (_N, _CIN, *pooled))
        )
        yield Call(
            [x, idx, 2], {"output_size": spatial}, 0, "max_unpool(x, indices, k)"
        )
        yield Call([x, idx, 2], {}, 0, "max_unpool(x, indices, k)")
    elif name in ("gaussian_nll_loss",):
        yield Call(
            [_f((_N, 4), domain), _f((_N, 4), domain), _f((_N, 4), "positive")],
            {},
            0,
            "gaussian_nll(input, target, var)",
        )
    elif name in ("multilabel_margin_loss",):
        yield Call(
            [
                _f((_N, 4), domain),
                _probe.as_int(np.array([[0, -1, -1, -1], [1, -1, -1, -1]])),
            ],
            {},
            0,
            "multilabel_margin(input, target)",
        )
    elif name in ("ctc_loss",):
        t, n, c = 6, 2, 4
        log_probs = _f((t, n, c), domain)
        targets = _probe.as_int(_probe.rng(8).integers(1, c, (n, 3)))
        yield Call(
            [
                log_probs,
                targets,
                _probe.as_int(np.array([t, t])),
                _probe.as_int(np.array([3, 3])),
            ],
            {},
            0,
            "ctc_loss(log_probs, targets, input_lengths, target_lengths)",
        )
    elif name in ("threshold", "threshold_"):
        yield Call(
            [_f(_probe.SHAPE, domain), 0.0, 0.0],
            {},
            0,
            "threshold(x, threshold, value)",
        )
    elif name in ("fused_linear_relu", "fused_linear_gelu"):
        w, b = _f((_COUT, _CIN), domain), _f((_COUT,), domain)
        yield Call(
            [_f((_N, _CIN), domain), w, b], {}, 0, "fused_linear(x, weight, bias)"
        )
        yield Call(
            [_f((_N, _CIN), domain)],
            {"weight": w, "bias": b},
            0,
            "fused_linear(x, weight=, bias=)",
        )
    elif name in ("sinusoidal_embedding",):
        yield Call(
            [_probe.as_f64(np.arange(4.0)), 8], {}, 0, "sinusoidal_embedding(pos, dim)"
        )
    elif name in ("sinusoidal_embedding_2d",):
        yield Call([8, 4, 4], {}, 0, "sinusoidal_embedding_2d(dim, h, w)")
    elif name in ("multi_head_attention_forward",):
        seq, batch, embed = 4, 2, 8
        q = _f((seq, batch, embed), domain)
        yield Call(
            [
                q,
                q,
                q,
                embed,
                2,
                _f((3 * embed, embed), domain),
                _f((3 * embed,), domain),
            ],
            {},
            0,
            "mha_forward(q, k, v, embed, heads, in_proj_w, in_proj_b)",
        )
    elif name in ("polygamma",):
        yield Call([1, _f(_probe.SHAPE, "positive")], {}, 1, "polygamma(n, x)")
    elif name in ("one_hot",):
        yield Call([_int((_N,), 4), 4], {}, 0, "one_hot(idx, n)")


def _accessor(name: str, domain: str) -> "Iterator[Call]":
    """Shape, dtype and residency predicates — they answer with a scalar."""
    if name in ("grad", "grad_fn"):
        # Both are ``None`` on a fresh tensor, so the probe has to make a
        # gradient exist before asking for one.
        leaf = _f((2, 3), domain)
        leaf.requires_grad_(True)
        (leaf * leaf).sum().backward()
        yield Call([leaf], {}, 0, f"{name} after a backward")
    x = _f((2, 3), domain)
    yield Call([x], {}, 0, "accessor(x)")
    yield Call([x, 0], {}, 0, "accessor(x, dim)")
    yield Call([x, x], {}, 0, "accessor(x, other)")


# ── resolution ───────────────────────────────────────────────────────────────
# Ordered: the first predicate that matches wins.  Patterns are matched
# against the short name, so ``F.conv2d`` and ``lucid.conv2d`` share a
# builder.

_FAMILIES: list[tuple[str, "Callable[[str, str], Iterator[Call]]"]] = [
    (r"^conv(_transpose)?[123]d$", _conv),
    (r"^max_unpool[123]d$|^lp_pool[123]d$", _nn_leftover),
    (
        r"^(ctc_loss|gaussian_nll_loss|multilabel_margin_loss|threshold_?|"
        r"fused_linear_(relu|gelu)|sinusoidal_embedding(_2d)?|"
        r"multi_head_attention_forward|polygamma|one_hot)$",
        _nn_leftover,
    ),
    (r"^(addbmm|baddbmm|addmm|addmv|addr|addcmul|addcdiv|lerp)$", _ternary),
    (r"^bitwise_|^invert$", _bitwise),
    (
        r"^(arange|linspace|logspace|eye|full|new_full|empty|zeros|ones|rand|randn|"
        r"randint|randperm|normal|bernoulli|meshgrid|cartesian_prod|vander|"
        r"empty_like|full_like)$",
        _factory,
    ),
    (
        r"^(gather|scatter|scatter_add|scatter_reduce|index_select|take|"
        r"take_along_dim|take_along_axis|masked_fill_?|searchsorted|bucketize|"
        r"narrow|where)$",
        _indexing,
    ),
    (
        r"^(broadcast_to|expand|permute|moveaxis|movedim|swapaxes|swapdims|tile|"
        r"repeat|roll|unflatten|unfold|dsplit|hsplit|vsplit|tensor_split|view)$",
        _shape_with_args,
    ),
    (
        r"^(bmm|mm|dot|vdot|inner|outer|ger|multi_dot|einsum|float_power|matmul)$",
        _matmul,
    ),
    (
        r"^(cholesky(_ex)?|eig|eigh|eigvals|eigvalsh|inv|det|slogdet|matrix_exp|"
        r"matrix_power|lu|lu_factor|lu_solve|ldl_factor|ldl_solve|solve|"
        r"solve_triangular|lstsq)$",
        _linalg_extra,
    ),
    (
        r"^i?(rfft|hfft|fft)[2n]?$|^ifft[2n]?$|^irfft[2n]?$|^ihfft[2n]?$|"
        r"^r?fftfreq$|^i?fftshift$",
        _fft_full,
    ),
    (r"^(angle|imag|real|view_as_complex|view_as_real|conj|resolve_conj)$", _complex),
    (r"^(can_cast|promote_types|result_type|type|astype)$", _dtype_util),
    (
        r"^(dim|ndim|shape|size|numel|nbytes|element_size|stride|is_[a-z_]+|"
        r"grad|grad_fn|impl|untyped_storage)$",
        _accessor,
    ),
    (r"^(fractional_max_pool)[23]d$", _fractional_pool),
    (r"^adaptive_(avg|max)_pool[123]d$", _adaptive_pool),
    (r"^(avg|max|lp)_pool[123]d$", _pool),
    (
        r"norm$|^batch_norm|^layer_norm|^group_norm|^instance_norm|^local_response",
        _norm,
    ),
    (r"loss$|^cross_entropy$|^kl_div$|^bce", _loss),
    (r"^embedding", _embedding),
    (r"attention|^sdpa$", _attention),
    (
        r"^(interpolate|upsample.*|grid_sample|affine_grid|pixel_(un)?shuffle|"
        r"channel_shuffle|unfold|fold|pad)$",
        _resample,
    ),
    (r"^(rearrange|reduce|repeat|pack|unpack|einsum)$", _einops),
    (
        r"^(sum|mean|prod|max|min|var|std|median|nanmean|nansum|nanmedian|"
        r"amax|amin|logsumexp|cumsum|cumprod|cummax|cummin|norm|argmax|argmin|"
        r"any|all|count_nonzero)$",
        _reduction,
    ),
    (
        r"^(add|sub|mul|div|pow|maximum|minimum|fmod|remainder|floor_divide|"
        r"true_divide|arctan2|atan2|hypot|logaddexp|logaddexp2|copysign|nextafter|"
        r"equal|eq|ne|not_equal|greater|gt|ge|greater_equal|less|lt|le|less_equal|"
        r"logical_and|logical_or|logical_xor|bitwise_.*|matmul|dot|inner|outer|kron)$",
        _binary,
    ),
]

#: Exact overrides, for the handful whose family cannot be inferred.
_EXACT: dict[str, "Callable[[str, str], Iterator[Call]]"] = {
    "one_hot": lambda n, d: iter([Call([_int((_N,), 4), 4], {}, 0, "one_hot(idx, n)")]),
    "linear": lambda n, d: iter(
        [
            Call(
                [_f((_N, _CIN), d), _f((_COUT, _CIN), d), _f((_COUT,), d)],
                {},
                0,
                "linear",
            )
        ]
    ),
    "bilinear": lambda n, d: iter(
        [
            Call(
                [_f((_N, _CIN), d), _f((_N, _CIN), d), _f((_COUT, _CIN, _CIN), d)],
                {},
                0,
                "bilinear",
            )
        ]
    ),
    "softmax": lambda n, d: iter(
        [Call([_f(_probe.SHAPE, d)], {"dim": -1}, 0, "softmax dim=-1")]
    ),
    "log_softmax": lambda n, d: iter(
        [Call([_f(_probe.SHAPE, d)], {"dim": -1}, 0, "log_softmax dim=-1")]
    ),
    "softmin": lambda n, d: iter(
        [Call([_f(_probe.SHAPE, d)], {"dim": -1}, 0, "softmin dim=-1")]
    ),
    "glu": lambda n, d: iter([Call([_f((_N, 4), d)], {"dim": -1}, 0, "glu")]),
    "normalize": lambda n, d: iter(
        [Call([_f(_probe.SHAPE, d)], {"dim": -1}, 0, "normalize")]
    ),
    "repeat_kv": lambda n, d: iter(
        [Call([_f((_N, 2, 4, 4), d), 2], {}, 0, "repeat_kv(x, n_rep)")]
    ),
}


#: Keyed by the *qualified* name, because a short name is not unique:
#: ``einops.repeat`` takes a pattern string while ``lucid.repeat`` takes
#: a repeat count, and ``Tensor.unfold`` slides a window while
#: ``F.unfold`` extracts image patches.  Matching on the short name alone
#: sent four of these to the wrong builder.
_QUALIFIED: dict[str, "Callable[[str], Iterator[Call]]"] = {
    "lucid.einops.repeat": lambda d: iter(
        [
            Call(
                [_f((2, 3), d), "a b -> a b r"],
                {"r": 2},
                0,
                "einops.repeat(x, pattern, **axes)",
            )
        ]
    ),
    "lucid.einops.rearrange": lambda d: iter(
        [Call([_f((2, 3, 4), d), "a b c -> a c b"], {}, 0, "einops.rearrange")]
    ),
    "lucid.einops.reduce": lambda d: iter(
        [Call([_f((2, 3, 4), d), "a b c -> a b", "mean"], {}, 0, "einops.reduce")]
    ),
    "Tensor.where": lambda d: iter(
        [
            Call(
                [
                    _f((3, 4), d),
                    lucid.tensor(_probe.rng(6).random((3, 4)) > 0.5),
                    _f((3, 4), d),
                ],
                {},
                0,
                "x.where(cond, other)",
            )
        ]
    ),
    "Tensor.unfold": lambda d: iter(
        [Call([_f((2, 3, 8), d), 2, 3, 1], {}, 0, "x.unfold(dim, size, step)")]
    ),
    "F.unfold": lambda d: iter(
        [Call([_f((2, 3, 6, 6), d), 3], {"padding": 1}, 0, "F.unfold(x, kernel_size)")]
    ),
    "Tensor.new_full": lambda d: iter(
        [Call([_f((2, 3), d), (2, 2), 0.5], {}, 0, "x.new_full(shape, value)")]
    ),
    "lucid.linalg.vander": lambda d: iter(
        [
            Call(
                [_probe.as_f64(_probe.rng(3).uniform(0.5, 1.5, (4,)))],
                {},
                0,
                "vander(1-D)",
            )
        ]
    ),
    "lucid.normal": lambda d: iter(
        [Call([0.0, 1.0], {"size": (2, 3)}, 0, "normal(mean, std, size=)")]
    ),
    "lucid.initial_seed": lambda d: iter([Call([], {}, 0, "initial_seed()")]),
    "lucid.is_grad_enabled": lambda d: iter([Call([], {}, 0, "is_grad_enabled()")]),
    "lucid.is_nonzero": lambda d: iter(
        [Call([_probe.as_f64(np.array([1.0]))], {}, 0, "is_nonzero(single element)")]
    ),
    "F.multilabel_margin_loss": lambda d: iter(
        [
            Call(
                [
                    _f((2, 4), d),
                    # int32, not the int64 a plain ``lucid.tensor`` of ints
                    # produces — the loss rejects int64, reported separately.
                    lucid.tensor(
                        np.array([[3, 0, -1, -1], [2, -1, -1, -1]]), dtype=lucid.int32
                    ),
                ],
                {},
                0,
                "multilabel_margin_loss(input, target)",
            )
        ]
    ),
    "F.sinusoidal_embedding": lambda d: iter(
        [
            Call(
                [_probe.as_f64(np.arange(4.0)), 8],
                {},
                0,
                "sinusoidal_embedding(t, dim)",
            ),
            Call([4, 8], {}, 0, "sinusoidal_embedding(n, dim)"),
        ]
    ),
}


def invocations(
    name: str,
    domain: str,
    qualname: str | None = None,
    fn: Any = None,
) -> "Iterator[Call]":
    """Every candidate call for ``name``, best guess first.

    Four tiers, narrowing from "someone wrote this down for this exact
    symbol" to "guess":

    ``_QUALIFIED`` / ``_EXACT``
        Hand-written for one symbol, where the name is ambiguous across
        subsystems or the op needs something no rule would infer.
    ``_FAMILIES``
        Hand-written for a group — every convolution, every pooling layer.
    :mod:`~lucid.test.audit._autospec`
        Derived from the signature.  Covers the long tail that used to
        fall straight to the ladder and SKIP there.
    the ladder
        ``f(x)``, ``f(x, y)``, ``f(x, dim)`` — the original sweep's whole
        vocabulary, kept as the floor rather than the ceiling.

    Parameters
    ----------
    name : str
        Short symbol name — ``"conv2d"``, not ``"F.conv2d"``.
    domain : str
        Key into :data:`~lucid.test.audit._probe.DOMAINS`.
    qualname : str, optional
        Full spelling, used to disambiguate names that exist in more than
        one subsystem.
    fn : callable, optional
        The resolved callable.  Without it the signature tier is skipped
        — there is nothing to introspect — and the symbol falls to the
        ladder as it did before.

    Yields
    ------
    Call
        Tried in order until one runs.
    """
    if qualname is not None and qualname in _QUALIFIED:
        yield from _QUALIFIED[qualname](domain)
    if name in _EXACT:
        yield from _EXACT[name](name, domain)
    for pattern, build in _FAMILIES:
        if re.search(pattern, name):
            yield from build(name, domain)
            break
    if fn is not None:
        yield from _autospec.invocations(fn, name, domain)
    yield from _unary(name, domain)
    yield from _binary(name, domain)
    x = _f(_probe.SHAPE, domain)
    yield Call([x, -1], {}, 0, "op(x, dim)")
    yield Call([x, 2], {}, 0, "op(x, int)")
    yield Call([[x, x]], {}, 0, "op([x, x])")


def has_spec(name: str, fn: Any = None) -> bool:
    """Whether ``name`` gets a real invocation rather than the ladder.

    ``fn`` opts in the signature tier: a symbol nobody wrote a spec for is
    still specified if its own signature says enough.
    """
    if name in _EXACT:
        return True
    if any(re.search(p, name) for p, _ in _FAMILIES):
        return True
    if fn is None:
        return False
    return next(_autospec.invocations(fn, name, "moderate"), None) is not None


def spec_families() -> list[str]:
    """The family patterns, for ``--list-specs``."""
    return [p for p, _ in _FAMILIES]


__all__ = ["Call", "has_spec", "invocations", "spec_families"]

# Imported last, not at the top: ``_autospec`` needs :class:`Call` from
# here, so the two import each other.  By this line ``Call`` is bound, and
# the partially-initialised module it sees is complete enough.  Same
# arrangement as ``_axes`` and its sub-axis modules.
from lucid.test.audit import _autospec  # noqa: E402
