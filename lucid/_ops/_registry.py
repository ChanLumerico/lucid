"""
Ops registry: maps op names to engine functions and Tensor method names.

Used by ``_methods.py`` to auto-inject Tensor methods and by
``_ops/__init__.py`` to expose free functions.

Adapter functions (signature normalisers, e.g. ``A._sum_adapter``) live in
``_adapters.py`` — this file references them so the registry stays a
declarative table of ``OpEntry`` records, not a mix of declarations and
plumbing.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from lucid._C import engine as _C_engine
from lucid._ops import _adapters as A  # accessor for every signature-normaliser


@dataclass
class OpEntry:
    """Descriptor for a single operation."""

    name: str
    engine_fn: Callable[..., Any]
    n_tensor_args: int  # positional Tensor args to unwrap; -1 = first arg is list
    returns_tensor: bool = True
    inplace: bool = False
    method_name: str | None = None  # Tensor method name (None = no method)
    free_fn_name: str | None = None  # lucid.xxx name (None = same as name)
    extra_kwargs: list[str] = field(default_factory=list)


_R = _C_engine  # shorthand

# fmt: off
_REGISTRY: list[OpEntry] = [
    # ── unary ──────────────────────────────────────────────────────────────
    OpEntry("neg",       _R.neg,       1, method_name="neg",       free_fn_name="neg"),
    OpEntry("abs",       _R.abs,       1, method_name="abs",       free_fn_name="abs"),
    OpEntry("sign",      _R.sign,      1, method_name="sign",      free_fn_name="sign"),
    OpEntry("exp",       _R.exp,       1, method_name="exp",       free_fn_name="exp"),
    OpEntry("log",       _R.log,       1, method_name="log",       free_fn_name="log"),
    OpEntry("log2",      _R.log2,      1, method_name="log2",      free_fn_name="log2"),
    OpEntry("sqrt",      _R.sqrt,      1, method_name="sqrt",      free_fn_name="sqrt"),
    # rsqrt: not in engine — implemented in Python as reciprocal(sqrt(x))
    OpEntry("square",    _R.square,    1, method_name="square",    free_fn_name="square"),
    OpEntry("reciprocal",_R.reciprocal,1, method_name="reciprocal",free_fn_name="reciprocal"),
    OpEntry("floor",     _R.floor,     1, method_name="floor",     free_fn_name="floor"),
    OpEntry("ceil",      _R.ceil,      1, method_name="ceil",      free_fn_name="ceil"),
    OpEntry("round",     _R.round,     1, method_name="round",     free_fn_name="round"),
    OpEntry("sin",       _R.sin,       1, method_name="sin",       free_fn_name="sin"),
    OpEntry("cos",       _R.cos,       1, method_name="cos",       free_fn_name="cos"),
    OpEntry("tan",       _R.tan,       1, method_name="tan",       free_fn_name="tan"),
    OpEntry("arcsin",    _R.arcsin,    1, method_name="arcsin",    free_fn_name="arcsin"),
    OpEntry("arccos",    _R.arccos,    1, method_name="arccos",    free_fn_name="arccos"),
    OpEntry("arctan",    _R.arctan,    1, method_name="arctan",    free_fn_name="arctan"),
    OpEntry("sinh",      _R.sinh,      1, method_name="sinh",      free_fn_name="sinh"),
    OpEntry("cosh",      _R.cosh,      1, method_name="cosh",      free_fn_name="cosh"),
    OpEntry("tanh",      _R.tanh,      1, method_name="tanh",      free_fn_name="tanh"),
    OpEntry("relu",      _R.relu,      1, method_name="relu",      free_fn_name="relu"),
    OpEntry("sigmoid",   _R.sigmoid,   1, method_name="sigmoid",   free_fn_name="sigmoid"),
    OpEntry("silu",      _R.silu,      1, method_name="silu",      free_fn_name="silu"),
    OpEntry("gelu",      _R.gelu,      1, method_name="gelu",      free_fn_name="gelu"),
    OpEntry("mish",      _R.mish,      1, method_name="mish",      free_fn_name="mish"),
    OpEntry("selu",      _R.selu,      1, method_name="selu",      free_fn_name="selu"),
    OpEntry("softplus",  _R.softplus,  1, method_name="softplus",  free_fn_name="softplus"),
    OpEntry("relu6",     _R.relu6,     1, method_name="relu6",     free_fn_name="relu6"),
    OpEntry("hard_sigmoid", _R.hard_sigmoid, 1, method_name="hard_sigmoid", free_fn_name="hard_sigmoid"),
    OpEntry("hard_swish",   _R.hard_swish,   1, method_name="hard_swish",   free_fn_name="hard_swish"),
    OpEntry("ravel",     _R.ravel,     1, method_name="ravel",     free_fn_name="ravel"),
    OpEntry("contiguous",_R.contiguous,1, method_name="contiguous",free_fn_name="contiguous"),

    # ── in-place unary ─────────────────────────────────────────────────────
    OpEntry("neg_",        _R.neg_,        1, inplace=True, method_name="neg_"),
    OpEntry("abs_",        _R.abs_,        1, inplace=True, method_name="abs_"),
    OpEntry("sign_",       _R.sign_,       1, inplace=True, method_name="sign_"),
    OpEntry("reciprocal_", _R.reciprocal_, 1, inplace=True, method_name="reciprocal_"),
    OpEntry("square_",     _R.square_,     1, inplace=True, method_name="square_"),
    OpEntry("cube_",       _R.cube_,       1, inplace=True, method_name="cube_"),
    OpEntry("exp_",        _R.exp_,        1, inplace=True, method_name="exp_"),
    OpEntry("log_",        _R.log_,        1, inplace=True, method_name="log_"),
    OpEntry("log2_",       _R.log2_,       1, inplace=True, method_name="log2_"),
    OpEntry("sqrt_",       _R.sqrt_,       1, inplace=True, method_name="sqrt_"),
    OpEntry("sin_",        _R.sin_,        1, inplace=True, method_name="sin_"),
    OpEntry("cos_",        _R.cos_,        1, inplace=True, method_name="cos_"),
    OpEntry("tan_",        _R.tan_,        1, inplace=True, method_name="tan_"),
    OpEntry("arcsin_",     _R.arcsin_,     1, inplace=True, method_name="arcsin_"),
    OpEntry("arccos_",     _R.arccos_,     1, inplace=True, method_name="arccos_"),
    OpEntry("arctan_",     _R.arctan_,     1, inplace=True, method_name="arctan_"),
    OpEntry("sinh_",       _R.sinh_,       1, inplace=True, method_name="sinh_"),
    OpEntry("cosh_",       _R.cosh_,       1, inplace=True, method_name="cosh_"),
    OpEntry("tanh_",       _R.tanh_,       1, inplace=True, method_name="tanh_"),
    OpEntry("sigmoid_",    _R.sigmoid_,    1, inplace=True, method_name="sigmoid_"),
    OpEntry("relu_",       _R.relu_,       1, inplace=True, method_name="relu_"),
    OpEntry("floor_",      _R.floor_,      1, inplace=True, method_name="floor_"),
    OpEntry("ceil_",       _R.ceil_,       1, inplace=True, method_name="ceil_"),
    OpEntry("round_",      _R.round_,      1, inplace=True, method_name="round_"),

    # ── binary ─────────────────────────────────────────────────────────────
    OpEntry("add",      _R.add,      2, method_name="add",      free_fn_name="add"),
    OpEntry("sub",      _R.sub,      2, method_name="sub",      free_fn_name="sub"),
    OpEntry("mul",      _R.mul,      2, method_name="mul",      free_fn_name="mul"),
    OpEntry("div",      _R.div,      2, method_name="div",      free_fn_name="div"),
    OpEntry("pow",      _R.pow,      2, method_name="pow",      free_fn_name="pow"),
    OpEntry("maximum",  _R.maximum,  2, method_name="maximum",  free_fn_name="maximum"),
    OpEntry("minimum",  _R.minimum,  2, method_name="minimum",  free_fn_name="minimum"),
    OpEntry("matmul",   _R.matmul,   2, method_name="matmul",   free_fn_name="matmul"),
    OpEntry("dot",      _R.dot,      2, method_name="dot",      free_fn_name="dot"),
    OpEntry("inner",    _R.inner,    2, method_name="inner",    free_fn_name="inner"),
    OpEntry("outer",    _R.outer,    2, method_name="outer",    free_fn_name="outer"),

    # ── in-place binary ────────────────────────────────────────────────────
    OpEntry("add_",     _R.add_,     2, inplace=True, method_name="add_"),
    OpEntry("sub_",     _R.sub_,     2, inplace=True, method_name="sub_"),
    OpEntry("mul_",     _R.mul_,     2, inplace=True, method_name="mul_"),
    OpEntry("div_",     _R.div_,     2, inplace=True, method_name="div_"),
    OpEntry("pow_",     _R.pow_,     2, inplace=True, method_name="pow_"),
    OpEntry("maximum_", _R.maximum_, 2, inplace=True, method_name="maximum_"),
    OpEntry("minimum_", _R.minimum_, 2, inplace=True, method_name="minimum_"),

    # ── reduction (with API-compat adapters) ───────────────────────────────
    OpEntry("sum",    A._sum_adapter,    1, method_name="sum",    free_fn_name="sum",
            extra_kwargs=["dim", "keepdim", "axis", "axes", "keepdims"]),
    OpEntry("mean",   A._mean_adapter,   1, method_name="mean",   free_fn_name="mean",
            extra_kwargs=["dim", "keepdim", "axis", "axes", "keepdims"]),
    OpEntry("prod",   A._prod_adapter,   1, method_name="prod",   free_fn_name="prod",
            extra_kwargs=["dim", "keepdim", "axis", "axes", "keepdims"]),
    OpEntry("max",    A._max_adapter,    1, method_name="max",    free_fn_name="max",
            extra_kwargs=["dim", "keepdim", "axis", "axes", "keepdims"]),
    OpEntry("min",    A._min_adapter,    1, method_name="min",    free_fn_name="min",
            extra_kwargs=["dim", "keepdim", "axis", "axes", "keepdims"]),
    OpEntry("var",    A._var_adapter,    1, method_name="var",    free_fn_name="var",
            extra_kwargs=["dim", "keepdim", "correction", "unbiased",
                          "axis", "axes", "keepdims"]),
    OpEntry("argmax", A._argmax_adapter, 1, method_name="argmax", free_fn_name="argmax",
            extra_kwargs=["dim", "keepdim", "axis", "keepdims"]),
    OpEntry("argmin", A._argmin_adapter, 1, method_name="argmin", free_fn_name="argmin",
            extra_kwargs=["dim", "keepdim", "axis", "keepdims"]),
    OpEntry("cumsum", _R.cumsum, 1, method_name="cumsum", free_fn_name="cumsum",
            extra_kwargs=["axis"]),
    OpEntry("cumprod",_R.cumprod,1, method_name="cumprod",free_fn_name="cumprod",
            extra_kwargs=["axis"]),
    OpEntry("trace",  _R.trace,  1, method_name="trace",  free_fn_name="trace"),

    # ── shape / layout ─────────────────────────────────────────────────────
    OpEntry("reshape",    A._reshape_adapter, 1, method_name="reshape",    free_fn_name="reshape"),
    OpEntry("squeeze",    A._squeeze_adapter, 1, method_name="squeeze",    free_fn_name="squeeze",
            extra_kwargs=["dim"]),
    OpEntry("squeeze_all",_R.squeeze_all,1, method_name="squeeze_all"),
    OpEntry("unsqueeze",  _R.unsqueeze,  1, method_name="unsqueeze",  free_fn_name="unsqueeze",
            extra_kwargs=["dim"]),
    OpEntry("flatten",    _R.flatten,    1, method_name="flatten",    free_fn_name="flatten",
            extra_kwargs=["start", "end"]),
    OpEntry("permute",    A._permute_adapter, 1, method_name="permute",    free_fn_name="permute"),
    OpEntry("transpose",  _R.transpose,  1, method_name="transpose",  free_fn_name="transpose"),
    OpEntry("swapaxes",   _R.swapaxes,   1, method_name="swapaxes",
            extra_kwargs=["d0", "d1"]),  # positional: swapaxes(d0, d1)
    OpEntry("broadcast_to",_R.broadcast_to,1,method_name="broadcast_to",free_fn_name="broadcast_to",
            extra_kwargs=["shape"]),
    OpEntry("expand",     A._expand_adapter, 1, method_name="expand",     free_fn_name="expand"),
    OpEntry("expand_dims",_R.expand_dims,1, method_name="expand_dims",
            extra_kwargs=["axis"]),
    # ``lucid.repeat(x, repeats, dim=None)`` — interleave semantics.  No
    # ``method_name`` because Tensor.repeat (below) follows the reference
    # framework's ``Tensor.repeat`` instead, which tiles copies.
    OpEntry("repeat",     A._repeat_adapter, 1, method_name=None,         free_fn_name="repeat",
            extra_kwargs=["dim"]),
    # ``Tensor.repeat(*sizes)`` — tile copies (separate semantics from the
    # free function above; see ``A._repeat_method_adapter``).
    OpEntry("repeat_method", A._repeat_method_adapter, 1,
            method_name="repeat", free_fn_name=None),
    OpEntry("tile",       _R.tile,       1, method_name="tile",       free_fn_name="tile",
            extra_kwargs=["reps"]),
    OpEntry("roll",       _R.roll,       1, method_name="roll",       free_fn_name="roll",
            extra_kwargs=["shifts", "dims"]),
    OpEntry("tril",       _R.tril,       1, method_name="tril",       free_fn_name="tril",
            extra_kwargs=["k"]),
    OpEntry("triu",       _R.triu,       1, method_name="triu",       free_fn_name="triu",
            extra_kwargs=["k"]),
    OpEntry("pad",        A._pad_adapter, 1, method_name="pad",        free_fn_name="pad",
            extra_kwargs=["padding", "mode", "value"]),

    # ── index / gather ─────────────────────────────────────────────────────
    OpEntry("gather",    _R.gather,    2, method_name="gather",    free_fn_name="gather",
            extra_kwargs=["dim"]),
    OpEntry("sort",      _R.sort,      1, method_name="sort",      free_fn_name="sort",
            extra_kwargs=["dim", "descending"]),
    OpEntry("argsort",   _R.argsort,   1, method_name="argsort",   free_fn_name="argsort",
            extra_kwargs=["dim", "descending"]),
    OpEntry("nonzero",   _R.nonzero,   1, method_name="nonzero",   free_fn_name="nonzero"),
    OpEntry("unique",    _R.unique,    1, method_name=None,        free_fn_name="unique"),
    OpEntry("topk",      _R.topk,      1, method_name="topk",      free_fn_name="topk",
            extra_kwargs=["k", "dim", "largest"]),
    OpEntry("diagonal",  _R.diagonal,  1, method_name="diagonal",  free_fn_name="diagonal",
            extra_kwargs=["offset", "dim1", "dim2"]),

    # ── comparison ────────────────────────────────────────────────────────
    OpEntry("equal",         _R.equal,         2, method_name=None, free_fn_name="equal"),
    OpEntry("not_equal",     _R.not_equal,     2, method_name=None, free_fn_name="not_equal"),
    OpEntry("greater",       _R.greater,       2, method_name=None, free_fn_name="greater"),
    OpEntry("greater_equal", _R.greater_equal, 2, method_name=None, free_fn_name="greater_equal"),
    OpEntry("less",          _R.less,          2, method_name=None, free_fn_name="less"),
    OpEntry("less_equal",    _R.less_equal,    2, method_name=None, free_fn_name="less_equal"),

    # ── masking ────────────────────────────────────────────────────────────
    # ``where`` and ``masked_fill`` auto-cast their condition/mask to bool to
    # match reference behaviour, so they need adapters rather than direct
    # engine bindings.
    OpEntry("where",       A._where_adapter, 0, free_fn_name="where"),
    OpEntry("masked_fill", A._masked_fill_adapter, 1,
            method_name="masked_fill", free_fn_name="masked_fill",
            extra_kwargs=["value"]),

    # ── joining ────────────────────────────────────────────────────────────
    OpEntry("concatenate", _R.concatenate, -1, free_fn_name="cat",
            extra_kwargs=["axis"]),
    OpEntry("stack",       _R.stack,       -1, free_fn_name="stack",
            extra_kwargs=["axis"]),
    OpEntry("hstack",      _R.hstack,      -1, free_fn_name="hstack"),
    OpEntry("vstack",      _R.vstack,      -1, free_fn_name="vstack"),
    OpEntry("split",       A._split_adapter, 1,  method_name="split",      free_fn_name="split",
            extra_kwargs=["dim"]),
    OpEntry("chunk",       _R.chunk,       1,  method_name="chunk",      free_fn_name="chunk",
            extra_kwargs=["n", "axis"]),
    OpEntry("unbind",      _R.unbind,      1,  method_name="unbind",     free_fn_name="unbind",
            extra_kwargs=["axis"]),
    OpEntry("meshgrid",    A._meshgrid_adapter, 0, free_fn_name="meshgrid",
            extra_kwargs=["indexing"]),

    # ── softmax / log_softmax (have axis kwarg) ────────────────────────────
    OpEntry("softmax",     _R.softmax,     1, method_name="softmax",     free_fn_name="softmax",
            extra_kwargs=["axis"]),
    OpEntry("log_softmax", _R.log_softmax, 1, method_name="log_softmax", free_fn_name="log_softmax",
            extra_kwargs=["axis"]),

    # ── rsqrt / std (engine-native) ────────────────────────────────────────
    OpEntry("rsqrt", _R.rsqrt, 1, method_name="rsqrt", free_fn_name="rsqrt"),
    OpEntry("std",   A._std_adapter, 1, method_name="std",   free_fn_name="std",
            extra_kwargs=["dim", "keepdim", "correction", "unbiased",
                          "axis", "axes", "keepdims"]),

    # ── boolean reductions ─────────────────────────────────────────────────
    OpEntry("any", _R.any, 1, method_name="any", free_fn_name="any"),
    OpEntry("all", _R.all, 1, method_name="all", free_fn_name="all"),

    # ── linear algebra ─────────────────────────────────────────────────────
    OpEntry("tensordot",  A._tensordot_adapter, 2, free_fn_name="tensordot",
            extra_kwargs=["dims"]),
    OpEntry("clip",       _R.clip,      1, method_name="clip",       free_fn_name="clip",
            extra_kwargs=["min", "max"]),

    # ── floating-point predicates (output is always bool) ──────────────────
    OpEntry("isinf",      _R.isinf,     1, method_name="isinf",     free_fn_name="isinf"),
    OpEntry("isnan",      _R.isnan,     1, method_name="isnan",     free_fn_name="isnan"),
    OpEntry("isfinite",   _R.isfinite,  1, method_name="isfinite",  free_fn_name="isfinite"),
    OpEntry("nan_to_num", _R.nan_to_num,1, method_name="nan_to_num",free_fn_name="nan_to_num",
            extra_kwargs=["nan", "posinf", "neginf"]),

    # ── tensor lifecycle ────────────────────────────────────────────────────
    # detach: deep-copy without gradient tracking (uses contiguous + clone_with_grad).
    OpEntry("detach",      A._detach_adapter, 1, method_name="detach", free_fn_name="detach"),
    # clone: deep-copy preserving autograd history (contiguous = storage copy).
    OpEntry("clone",       _R.contiguous,   1, method_name="clone",  free_fn_name="clone"),
    # clamp is an alias for clip (same signature, same engine op).
    OpEntry("clamp",       _R.clip,         1, method_name="clamp",  free_fn_name="clamp",
            extra_kwargs=["min", "max"]),
    # scatter_add: arg order differs — adapter reorders before calling engine.
    # Python:  scatter_add(x, dim, index, src)
    # Engine:  scatter_add(base, indices, src, dim)
    # n_tensor_args=1 auto-unwraps x; the adapter manually unwraps index/src.
    OpEntry("scatter_add", A._scatter_add_adapter, 1,
            method_name="scatter_add", free_fn_name="scatter_add"),

    # ══ composite ops (impl in _C/ops/composite/) ═══════════════════════════
    # Composition wrappers built atop primitives.  Autograd flows through the
    # underlying primitive backward nodes; no new schemas are registered.

    # ── elementwise math compositions (unary) ───────────────────────────────
    OpEntry("log10", _R.log10, 1, method_name="log10", free_fn_name="log10"),
    OpEntry("log1p", _R.log1p, 1, method_name="log1p", free_fn_name="log1p"),
    OpEntry("exp2",  _R.exp2,  1, method_name="exp2",  free_fn_name="exp2"),
    OpEntry("trunc", _R.trunc, 1, method_name="trunc", free_fn_name="trunc"),
    OpEntry("frac",  _R.frac,  1, method_name="frac",  free_fn_name="frac"),

    # ── elementwise math compositions (binary) ──────────────────────────────
    OpEntry("atan2",     _R.atan2,     2, method_name="atan2",     free_fn_name="atan2"),
    OpEntry("fmod",      _R.fmod,      2, method_name="fmod",      free_fn_name="fmod"),
    OpEntry("remainder", _R.remainder, 2, method_name="remainder", free_fn_name="remainder"),
    OpEntry("hypot",     _R.hypot,     2, method_name="hypot",     free_fn_name="hypot"),
    OpEntry("logaddexp", _R.logaddexp, 2, method_name="logaddexp", free_fn_name="logaddexp"),

    # ── reduction compositions ──────────────────────────────────────────────
    OpEntry("logsumexp", A._logsumexp_adapter, 1,
            method_name="logsumexp", free_fn_name="logsumexp"),

    # ── linear-algebra compositions ─────────────────────────────────────────
    OpEntry("mm",   _R.mm,   2, method_name="mm",   free_fn_name="mm"),
    OpEntry("bmm",  _R.bmm,  2, method_name="bmm",  free_fn_name="bmm"),
    OpEntry("kron", _R.kron, 2, method_name="kron", free_fn_name="kron"),

    # ── logical compositions ────────────────────────────────────────────────
    OpEntry("logical_and", _R.logical_and, 2,
            method_name="logical_and", free_fn_name="logical_and"),
    OpEntry("logical_or",  _R.logical_or, 2,
            method_name="logical_or",  free_fn_name="logical_or"),
    OpEntry("logical_xor", _R.logical_xor, 2,
            method_name="logical_xor", free_fn_name="logical_xor"),
    OpEntry("logical_not", _R.logical_not, 1,
            method_name="logical_not", free_fn_name="logical_not"),

    # ── indexing compositions ───────────────────────────────────────────────
    OpEntry("take",         A._take_adapter, 1,
            method_name="take", free_fn_name="take"),
    OpEntry("index_select", A._index_select_adapter, 1,
            method_name="index_select", free_fn_name="index_select"),
    OpEntry("narrow",       A._narrow_adapter, 1,
            method_name="narrow", free_fn_name="narrow"),
    OpEntry("scatter",      A._scatter_adapter, 1,
            method_name="scatter", free_fn_name="scatter",
            extra_kwargs=["reduce"]),
    OpEntry("kthvalue",     A._kthvalue_adapter, 1,
            method_name="kthvalue", free_fn_name="kthvalue"),

    # ── layout compositions ─────────────────────────────────────────────────
    OpEntry("movedim",   A._movedim_adapter, 1,
            method_name="movedim", free_fn_name="movedim"),
    OpEntry("unflatten", A._unflatten_adapter, 1,
            method_name="unflatten", free_fn_name="unflatten"),

    # ── stats / search compositions ─────────────────────────────────────────
    OpEntry("histc",         A._histc_adapter, 1,
            method_name="histc", free_fn_name="histc"),
    OpEntry("cartesian_prod", A._cartesian_prod_adapter, 0,
            method_name=None, free_fn_name="cartesian_prod"),
    OpEntry("searchsorted",  A._searchsorted_adapter, 2,
            method_name=None, free_fn_name="searchsorted",
            extra_kwargs=["right"]),
    OpEntry("bucketize",     A._bucketize_adapter, 2,
            method_name=None, free_fn_name="bucketize",
            extra_kwargs=["right"]),

    # ══ aliases of existing primitives (no new C++ — share engine kernels) ══

    # Comparison short names — same kernels as equal/not_equal/less/...
    OpEntry("eq", _R.eq, 2, method_name="eq", free_fn_name="eq"),
    OpEntry("ne", _R.ne, 2, method_name="ne", free_fn_name="ne"),
    OpEntry("lt", _R.lt, 2, method_name="lt", free_fn_name="lt"),
    OpEntry("le", _R.le, 2, method_name="le", free_fn_name="le"),
    OpEntry("gt", _R.gt, 2, method_name="gt", free_fn_name="gt"),
    OpEntry("ge", _R.ge, 2, method_name="ge", free_fn_name="ge"),

    # Trig short names — share kernels with arcsin/arccos/arctan.
    OpEntry("asin", _R.asin, 1, method_name="asin", free_fn_name="asin"),
    OpEntry("acos", _R.acos, 1, method_name="acos", free_fn_name="acos"),
    OpEntry("atan", _R.atan, 1, method_name="atan", free_fn_name="atan"),

    # ``bitwise_not`` is the reference framework's name for ``invert``.
    OpEntry("bitwise_not", _R.bitwise_not, 1,
            method_name="bitwise_not", free_fn_name="bitwise_not"),
    OpEntry("bitwise_and", _R.bitwise_and, 2,
            method_name="bitwise_and", free_fn_name="bitwise_and"),
    OpEntry("bitwise_or",  _R.bitwise_or,  2,
            method_name="bitwise_or",  free_fn_name="bitwise_or"),
    OpEntry("bitwise_xor", _R.bitwise_xor, 2,
            method_name="bitwise_xor", free_fn_name="bitwise_xor"),

    # masked_select: engine kernel returns a flat 1-D tensor of selected
    # elements.  Both inputs are tensors so n_tensor_args=2.
    OpEntry("masked_select", _R.masked_select, 2,
            method_name="masked_select", free_fn_name="masked_select"),

    # isclose: composite op with rtol/atol/equal_nan keyword arguments.
    # n_tensor_args=2 so both operands get unwrapped before the adapter runs.
    OpEntry("isclose", A._isclose_adapter, 2,
            method_name="isclose", free_fn_name="isclose",
            extra_kwargs=["rtol", "atol", "equal_nan"]),

    # repeat_interleave: thin wrapper around engine.repeat that supports
    # ``dim=None`` (flatten first) like the reference framework's API.
    OpEntry("repeat_interleave", A._repeat_interleave_adapter, 1,
            method_name="repeat_interleave", free_fn_name="repeat_interleave"),

    # Shape aliases.
    OpEntry("view",   A._view_adapter,   1,
            method_name="view", free_fn_name="view"),
    OpEntry("concat", A._concat_adapter, -1,
            method_name=None, free_fn_name="concat"),

    # ── top-level forwarders into the linalg sub-module ────────────────────
    OpEntry("cross", A._cross_adapter, 2,
            method_name="cross", free_fn_name="cross"),
    OpEntry("norm",  A._norm_adapter, 1,
            method_name="norm", free_fn_name="norm"),

    # ── transpose shorthand ─────────────────────────────────────────────────
    # ``Tensor.t()`` is the standard 2-D transpose — same engine kernel as
    # the ``T`` property, exposed here as a method so registry-driven method
    # injection picks it up alongside everything else.
    OpEntry("t", _R.T, 1, method_name="t", free_fn_name=None),

    # ── top-level forwarder into the einops sub-module ─────────────────────
    # The user explicitly wanted ``lucid.einops.einsum`` to remain the
    # primary entry point; this top-level alias just gives Python users the
    # familiar ``lucid.einsum(...)`` shorthand without going through that
    # sub-module.  Both expose the same engine kernel.
    OpEntry("einsum", A._einsum_adapter, 0,
            method_name=None, free_fn_name="einsum"),
]
# fmt: on
