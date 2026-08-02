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
        """The differentiated argument, as an array."""
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


# ── resolution ───────────────────────────────────────────────────────────────
# Ordered: the first predicate that matches wins.  Patterns are matched
# against the short name, so ``F.conv2d`` and ``lucid.conv2d`` share a
# builder.

_FAMILIES: list[tuple[str, "Callable[[str, str], Iterator[Call]]"]] = [
    (r"^conv(_transpose)?[123]d$", _conv),
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
    (r"^(fft|ifft|rfft|irfft|hfft|ihfft).*|.*fftshift$|^fftfreq$", _fft),
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


def invocations(name: str, domain: str) -> "Iterator[Call]":
    """Every candidate call for ``name``, best guess first.

    Parameters
    ----------
    name : str
        Short symbol name — ``"conv2d"``, not ``"F.conv2d"``.
    domain : str
        Key into :data:`~lucid.test.audit._probe.DOMAINS`.

    Yields
    ------
    Call
        Tried in order until one runs.  The generic unary and binary
        forms come last so a symbol with no spec still gets a chance.
    """
    if name in _EXACT:
        yield from _EXACT[name](name, domain)
    for pattern, build in _FAMILIES:
        if re.search(pattern, name):
            yield from build(name, domain)
            break
    # Generic ladder — the original sweep's entire vocabulary, kept as the
    # floor rather than the ceiling.
    yield from _unary(name, domain)
    yield from _binary(name, domain)
    x = _f(_probe.SHAPE, domain)
    yield Call([x, -1], {}, 0, "op(x, dim)")
    yield Call([x, 2], {}, 0, "op(x, int)")
    yield Call([[x, x]], {}, 0, "op([x, x])")


def has_spec(name: str) -> bool:
    """Whether ``name`` is covered by a real spec rather than the ladder."""
    if name in _EXACT:
        return True
    return any(re.search(p, name) for p, _ in _FAMILIES)


def spec_families() -> list[str]:
    """The family patterns, for ``--list-specs``."""
    return [p for p, _ in _FAMILIES]


__all__ = ["Call", "has_spec", "invocations", "spec_families"]
