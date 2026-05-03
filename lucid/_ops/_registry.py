"""
Ops registry: maps op names to engine functions and Tensor method names.

Used by _methods.py to auto-inject Tensor methods and by _ops/__init__.py
to expose free functions.
"""

from dataclasses import dataclass, field
from typing import Callable, Any
from lucid._C import engine as _C_engine


@dataclass
class OpEntry:
    """Descriptor for a single operation."""

    name: str
    engine_fn: Callable[..., Any]
    n_tensor_args: int          # positional Tensor args to unwrap; -1 = first arg is list
    returns_tensor: bool = True
    inplace: bool = False
    method_name: str | None = None     # Tensor method name (None = no method)
    free_fn_name: str | None = None    # lucid.xxx name (None = same as name)
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
    OpEntry("neg_",      _R.neg_,      1, inplace=True, method_name="neg_"),
    OpEntry("abs_",      _R.abs_,      1, inplace=True, method_name="abs_"),
    OpEntry("exp_",      _R.exp_,      1, inplace=True, method_name="exp_"),
    OpEntry("log_",      _R.log_,      1, inplace=True, method_name="log_"),
    OpEntry("sqrt_",     _R.sqrt_,     1, inplace=True, method_name="sqrt_"),
    OpEntry("floor_",    _R.floor_,    1, inplace=True, method_name="floor_"),
    OpEntry("ceil_",     _R.ceil_,     1, inplace=True, method_name="ceil_"),
    OpEntry("round_",    _R.round_,    1, inplace=True, method_name="round_"),

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

    # ── reduction ──────────────────────────────────────────────────────────
    OpEntry("sum",    _R.sum,    1, method_name="sum",    free_fn_name="sum",
            extra_kwargs=["axes", "keepdims"]),
    OpEntry("mean",   _R.mean,   1, method_name="mean",   free_fn_name="mean",
            extra_kwargs=["axes", "keepdims"]),
    OpEntry("prod",   _R.prod,   1, method_name="prod",   free_fn_name="prod",
            extra_kwargs=["axes", "keepdims"]),
    OpEntry("max",    _R.max,    1, method_name="max",    free_fn_name="max",
            extra_kwargs=["axes", "keepdims"]),
    OpEntry("min",    _R.min,    1, method_name="min",    free_fn_name="min",
            extra_kwargs=["axes", "keepdims"]),
    OpEntry("var",    _R.var,    1, method_name="var",    free_fn_name="var",
            extra_kwargs=["axes", "keepdims"]),
    OpEntry("argmax", _R.argmax, 1, method_name="argmax", free_fn_name="argmax",
            extra_kwargs=["axis", "keepdims"]),
    OpEntry("argmin", _R.argmin, 1, method_name="argmin", free_fn_name="argmin",
            extra_kwargs=["axis", "keepdims"]),
    OpEntry("cumsum", _R.cumsum, 1, method_name="cumsum", free_fn_name="cumsum",
            extra_kwargs=["axis"]),
    OpEntry("cumprod",_R.cumprod,1, method_name="cumprod",free_fn_name="cumprod",
            extra_kwargs=["axis"]),
    OpEntry("trace",  _R.trace,  1, method_name="trace",  free_fn_name="trace"),

    # ── shape / layout ─────────────────────────────────────────────────────
    OpEntry("reshape",    _R.reshape,    1, method_name="reshape",    free_fn_name="reshape",
            extra_kwargs=["shape"]),
    OpEntry("squeeze",    _R.squeeze,    1, method_name="squeeze",    free_fn_name="squeeze",
            extra_kwargs=["dim"]),
    OpEntry("squeeze_all",_R.squeeze_all,1, method_name="squeeze_all"),
    OpEntry("unsqueeze",  _R.unsqueeze,  1, method_name="unsqueeze",  free_fn_name="unsqueeze",
            extra_kwargs=["dim"]),
    OpEntry("flatten",    _R.flatten,    1, method_name="flatten",    free_fn_name="flatten",
            extra_kwargs=["start", "end"]),
    OpEntry("permute",    _R.permute,    1, method_name="permute",    free_fn_name="permute",
            extra_kwargs=["dims"]),
    OpEntry("transpose",  _R.transpose,  1, method_name="transpose",  free_fn_name="transpose"),
    OpEntry("swapaxes",   _R.swapaxes,   1, method_name="swapaxes",
            extra_kwargs=["d0", "d1"]),  # positional: swapaxes(d0, d1)
    OpEntry("broadcast_to",_R.broadcast_to,1,method_name="broadcast_to",free_fn_name="broadcast_to",
            extra_kwargs=["shape"]),
    OpEntry("expand",     _R.expand,     1, method_name="expand",     free_fn_name="expand",
            extra_kwargs=["shape"]),
    OpEntry("expand_dims",_R.expand_dims,1, method_name="expand_dims",
            extra_kwargs=["axis"]),
    OpEntry("repeat",     _R.repeat,     1, method_name="repeat",     free_fn_name="repeat",
            extra_kwargs=["repeats"]),
    OpEntry("tile",       _R.tile,       1, method_name="tile",       free_fn_name="tile",
            extra_kwargs=["reps"]),
    OpEntry("roll",       _R.roll,       1, method_name="roll",       free_fn_name="roll",
            extra_kwargs=["shifts", "dims"]),
    OpEntry("tril",       _R.tril,       1, method_name="tril",       free_fn_name="tril",
            extra_kwargs=["k"]),
    OpEntry("triu",       _R.triu,       1, method_name="triu",       free_fn_name="triu",
            extra_kwargs=["k"]),
    OpEntry("pad",        _R.pad,        1, method_name="pad",        free_fn_name="pad",
            extra_kwargs=["paddings", "mode", "constant"]),

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
    OpEntry("where",       _R.where,       3, free_fn_name="where"),
    OpEntry("masked_fill", _R.masked_fill, 2, method_name="masked_fill", free_fn_name="masked_fill",
            extra_kwargs=["value"]),

    # ── joining ────────────────────────────────────────────────────────────
    OpEntry("concatenate", _R.concatenate, -1, free_fn_name="cat",
            extra_kwargs=["axis"]),
    OpEntry("stack",       _R.stack,       -1, free_fn_name="stack",
            extra_kwargs=["axis"]),
    OpEntry("hstack",      _R.hstack,      -1, free_fn_name="hstack"),
    OpEntry("vstack",      _R.vstack,      -1, free_fn_name="vstack"),
    OpEntry("split",       _R.split,       1,  method_name="split",      free_fn_name="split",
            extra_kwargs=["sections", "axis"]),
    OpEntry("chunk",       _R.chunk,       1,  method_name="chunk",      free_fn_name="chunk",
            extra_kwargs=["n", "axis"]),
    OpEntry("unbind",      _R.unbind,      1,  method_name="unbind",     free_fn_name="unbind",
            extra_kwargs=["axis"]),
    OpEntry("meshgrid",    _R.meshgrid,   -1,  free_fn_name="meshgrid",
            extra_kwargs=["indexing"]),

    # ── softmax (has axis kwarg) ────────────────────────────────────────────
    OpEntry("softmax",    _R.softmax,   1, method_name="softmax",    free_fn_name="softmax",
            extra_kwargs=["axis"]),

    # ── linear algebra ─────────────────────────────────────────────────────
    OpEntry("tensordot",  _R.tensordot, 2, free_fn_name="tensordot",
            extra_kwargs=["axes_a", "axes_b"]),
    OpEntry("clip",       _R.clip,      1, method_name="clip",       free_fn_name="clip",
            extra_kwargs=["min", "max"]),
]
# fmt: on
