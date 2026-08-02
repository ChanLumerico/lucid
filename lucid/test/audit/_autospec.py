"""Deriving the invocation from the signature instead of writing it out.

:mod:`~lucid.test.audit._specs` is a hand-written ladder: a family
pattern per group of ops, and a generic ``f(x)`` / ``f(x, y)`` floor for
everything else.  446 of 827 callable symbols reached only that floor,
which is fine for a plain unary op and useless for anything taking a
second required argument of a shape the floor cannot guess.  Those became
SKIP — counted honestly, and still not verified.

That is the same failure mode as the module list which held the audit's
reach at 73.8%: a hand-maintained enumeration of something the
interpreter already knows.  A signature is machine-readable, so read it.
With this module the figure is 825 of 827; the two left take no arguments
at all, and calling them with none is what the smoke axis already does.

    def conv2d(input, weight, bias=None, stride=1, ...)
              ^tensor ^tensor  ^skipped, it has a default

Three rules, in order of how much they know:

1.  **Relations** — a parameter whose value depends on one already
    chosen.  ``weight`` has to agree with ``input`` on channels, a
    class-index ``target`` has to be bounded by the logits' width.  This
    is the only part that encodes semantics, and it is a dozen entries
    rather than one per op.
2.  **Name** — ``dim``, ``eps``, ``groups``, ``kernel_size``.  A name
    says more than ``int`` does.
3.  **Annotation text** — the fallback: ``Tensor``, ``bool``,
    ``tuple[int, int]``, ``Literal["mean", "sum"]``.

Only parameters *without* a default are filled.  A default is the
author's own statement of a value that works, and overriding it would
test this module's opinion rather than the op.

Annotations are read with ``Format.STRING``.  Lucid puts tensor types in
a ``TYPE_CHECKING`` block and leaves the annotation bare (H1 / H7 plus
PEP 649), so resolving them raises ``NameError`` for a large part of the
framework — the unevaluated text is both sufficient and always available.
"""

import annotationlib
import inspect
import re
from typing import TYPE_CHECKING, Any

import numpy as np

import lucid
import lucid.nn.utils.rnn
from lucid.test.audit import _probe
from lucid.test.audit._specs import Call

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Probe extents, matching :mod:`_specs` so a hand spec and a derived one
#: produce comparably sized work.
_N, _CIN, _COUT = 2, 3, 4
_L, _H, _W = 8, 6, 6

#: How many derived candidates a single symbol may yield.  The variants
#: exist to survive a wrong guess about rank or ``dim``, not to search;
#: an unbounded product would multiply the sweep by the arity.
_MAX_CANDIDATES = 6


def _tensor(shape: "tuple[int, ...]", domain: str) -> Any:
    return _probe.as_f64(_probe.sample(domain, shape))


def _indices(shape: "tuple[int, ...]", high: int) -> Any:
    return _probe.as_int(_probe.rng(_probe.SEED_B).integers(0, high, shape))


def _mask(shape: "tuple[int, ...]") -> Any:
    """An all-true boolean mask — masking everything out hides the op."""
    return lucid.tensor(np.ones(shape, dtype=bool), dtype=lucid.bool)


def _clean(annotation: Any) -> str:
    """The annotation as comparable text, whatever format it arrived in."""
    if annotation is inspect.Parameter.empty:
        return ""
    return str(annotation).replace("'", "").replace('"', "")


def _is_tensor_annotation(text: str) -> bool:
    if not re.search(r"\bTensor\b|\bTensorOrScalar\b|\b_ArrayOrScalar\b", text):
        return False
    return not re.search(r"\b(list|Sequence|tuple|Iterable)\b", text)


def _is_tensor_sequence(text: str) -> bool:
    return bool(re.search(r"\bTensor\b", text)) and bool(
        re.search(r"\b(list|Sequence|tuple|Iterable)\b", text)
    )


#: Parameters that want a live module.  ``fuse_conv_bn_eval`` needs a
#: convolution and a norm layer specifically, so the name picks the class
#: rather than everything getting a ``Linear``.
_MODULE_BY_NAME: "dict[str, Any]" = {
    "conv": lambda: lucid.nn.Conv2d(_CIN, _COUT, 3, padding=1),
    "bn": lambda: lucid.nn.BatchNorm2d(_COUT),
    "norm": lambda: lucid.nn.BatchNorm2d(_COUT),
    "linear": lambda: lucid.nn.Linear(_COUT, _COUT),
}


def _packed_sequence(domain: str) -> Any:
    """A real ``PackedSequence`` — the only thing ``pad_packed_sequence`` takes."""
    try:
        return lucid.nn.utils.rnn.pack_sequence(
            [_tensor((3, _COUT), domain), _tensor((2, _COUT), domain)],
            enforce_sorted=True,
        )
    except Exception:  # noqa: BLE001 - surveying, not asserting
        return None


def _module_for(name: str) -> Any:
    """A small module for a parameter that takes one."""
    build = _MODULE_BY_NAME.get(name)
    if build is not None:
        return build()
    return lucid.nn.Linear(_COUT, _COUT)


#: Parameters that take a module, by name.  ``prune`` and the fusion
#: helpers are the users; without these they were the only group of
#: ``nn.utils`` left on the ladder.
_MODULE_PARAMS = frozenset(
    {"module", "mod", "model", "source", "dest", "conv", "bn", "norm", "linear"}
)

#: Parameters that take a sequence of tensors.  ``cat`` and ``stack``
#: annotate theirs as bare ``tensors`` with no annotation at all, so the
#: name is the only signal there is.
_TENSOR_SEQUENCE_PARAMS = frozenset(
    {"tensors", "sequences", "arrays", "inputs", "seq", "primals", "tangents"}
)


def _literal_choice(text: str) -> "str | None":
    """The first option of a ``Literal[...]`` annotation, if it is one."""
    match = re.search(r"Literal\[([^\]]+)\]", text)
    if match is None:
        return None
    first = match.group(1).split(",")[0].strip()
    return first or None


#: Names that mean "the tensor being operated on", beyond the ones the
#: annotation already identifies.  ``self`` is here because a Tensor
#: method arrives unbound and its receiver is an ordinary parameter.
_TENSOR_PARAMS = frozenset(
    {"input", "x", "tensor", "a", "b", "self", "real", "imag", "abs", "angle"}
)

#: Values keyed on the parameter *name*, tried before the annotation.
#: ``int`` describes ``dim``, ``groups``, ``num_classes`` and
#: ``kernel_size`` equally badly.
_BY_NAME: "dict[str, Any]" = {
    "dim": -1,
    "M": 8,
    "nonlinearity": "relu",
    "clip_value": 1.0,
    "max_norm": 1.0,
    "cx": 0.0,
    "cy": 0.0,
    "axis": -1,
    "dim0": 0,
    "dim1": 1,
    "axis0": 0,
    "axis1": 1,
    "dims": (0, 1),
    "axes": (0, 1),
    "keepdim": False,
    "keepdims": False,
    "eps": 1e-5,
    "epsilon": 1e-5,
    "groups": 1,
    "num_groups": 1,
    "num_classes": _COUT,
    "num_embeddings": 8,
    "embedding_dim": _COUT,
    "num_heads": 2,
    "kernel_size": 3,
    "stride": 1,
    "padding": 0,
    "dilation": 1,
    "output_size": 2,
    "num_samples": 2,
    "p": 0.5,
    "q": 2,
    "n": 2,
    "k": 1,
    "diagonal": 0,
    "repeats": 2,
    "chunks": 2,
    "sections": 2,
    "split_size": 2,
    "start": 0,
    "end": 4,
    "step": 1,
    "min": 0.0,
    "max": 1.0,
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 1.0,
    "lambd": 0.5,
    "margin": 1.0,
    "reduction": "mean",
    "mode": "constant",
    "value": 0.0,
    "shape": (_N, _COUT),
    "size": (_N, _COUT),
    "shift": 1,
    "shifts": 1,
    "decimals": 2,
    "rounding_mode": None,
    "generator": None,
    "out": None,
}

#: Names whose value must agree with a tensor already chosen.  The only
#: semantic knowledge here, and the reason a derived spec can reach
#: ``conv2d`` and ``cross_entropy`` at all.
_RELATED = (
    "weight",
    "bias",
    "target",
    "labels",
    "other",
    "mat2",
    "tensor2",
    "key",
    "value",
    "mask",
    "attn_mask",
    "index",
    "indices",
    "running_mean",
    "running_var",
)


class _Plan:
    """The arguments chosen for one derived invocation."""

    __slots__ = ("args", "kwargs", "primary", "unknown", "note")

    def __init__(self) -> None:
        self.args: "list[Any]" = []
        self.kwargs: "dict[str, Any]" = {}
        self.primary: int = 0
        self.unknown: "list[str]" = []
        self.note: str = ""


def _related_value(
    name: str, text: str, reference: Any, op_name: str, domain: str
) -> "tuple[bool, Any]":
    """A value for ``name`` that agrees with the tensor already chosen."""
    if reference is None:
        return False, None
    shape = tuple(np.shape(reference))
    if not shape:
        return False, None

    if name in ("other", "mat2", "tensor2"):
        return True, _tensor(shape, domain)
    if name in ("key", "value"):
        return True, _tensor(shape, domain)
    if name in ("mask", "attn_mask"):
        return True, _mask(shape)
    if name in ("index", "indices"):
        return True, _indices(shape, max(shape[-1], 1))
    if name in ("running_mean", "running_var"):
        channels = shape[1] if len(shape) > 1 else shape[0]
        fill = 0.0 if name == "running_mean" else 1.0
        return True, _probe.as_f64(np.full((channels,), fill))

    if name in ("target", "labels"):
        # A class-index target is bounded by the width of the logits; a
        # regression target has the logits' shape.  Getting this wrong is
        # not a soft failure — an out-of-range index reads past the table.
        if re.search(r"cross_entropy|nll|multi_margin|_class", op_name):
            return True, _indices(shape[:1] or (1,), max(shape[-1], 1))
        return True, _tensor(shape, domain)

    if name == "weight":
        if re.search(r"conv", op_name):
            rank = _rank_from(op_name, len(shape))
            spatial = (3,) * rank
            return True, _tensor(
                (_COUT, shape[1] if len(shape) > 1 else 1, *spatial), domain
            )
        if re.search(r"linear|bilinear", op_name):
            return True, _tensor((_COUT, shape[-1]), domain)
        if re.search(r"embedding", op_name):
            return True, _tensor((8, _COUT), domain)
        # A per-element or per-channel weight, as norms and losses use.
        return True, _tensor((shape[-1],), domain)
    if name == "bias":
        return True, _tensor((_COUT,), domain)
    return False, None


def _rank_from(op_name: str, tensor_rank: int) -> int:
    match = re.search(r"([123])d\b", op_name)
    if match is not None:
        return int(match.group(1))
    return max(tensor_rank - 2, 1)


def _tensor_shapes(op_name: str) -> "tuple[tuple[int, ...], ...]":
    """Candidate shapes for the first tensor, most likely first."""
    match = re.search(r"([123])d\b", op_name)
    if match is not None:
        rank = int(match.group(1))
        spatial = {1: (_L,), 2: (_H, _W), 3: (4, _H, _W)}[rank]
        return ((_N, _CIN, *spatial),)
    if re.search(
        r"matmul|bmm|mm\b|matrix|linalg|inv|det|solve|cholesky|qr|svd", op_name
    ):
        return ((_N, _COUT, _COUT), (_COUT, _COUT))
    return ((_N, _COUT), (_N, _CIN, _H, _W), (_COUT,))


def _value_for(
    param: "inspect.Parameter",
    op_name: str,
    domain: str,
    reference: Any,
    shape: "tuple[int, ...]",
) -> "tuple[bool, Any, bool]":
    """``(found, value, is_tensor)`` for one required parameter."""
    name = param.name
    text = _clean(param.annotation)

    if name in _RELATED:
        found, value = _related_value(name, text, reference, op_name, domain)
        if found:
            return True, value, _probe.to_numpy(value) is not None

    if name in _MODULE_PARAMS and not _is_tensor_annotation(text):
        return True, _module_for(name), False
    if name in _TENSOR_SEQUENCE_PARAMS or _is_tensor_sequence(text):
        # Marked differentiable: for ``cat`` and ``stack`` the sequence is
        # the operand, and the generic ladder's ``op([x, x])`` form treats
        # it the same way.  Left un-marked, ``_build`` finds no tensor to
        # differentiate and discards a perfectly good invocation.
        return True, [_tensor(shape, domain), _tensor(shape, domain)], True
    if name == "parameters" or re.search(r"Iterable\[Parameter\]", text):
        return True, list(lucid.nn.Linear(_COUT, _COUT).parameters()), False
    if re.search(r"\bCallable\b", text):
        # ``gradcheck``, ``jacrev``, ``vmap`` — the function under test is
        # the argument.  Squaring is differentiable twice and has a second
        # derivative that is not zero, so it exercises what they compute.
        return True, (lambda *operands: operands[0] * operands[0]), False
    if _is_tensor_annotation(text) or name in _TENSOR_PARAMS:
        return True, _tensor(shape, domain), True
    if name == "module_cls" or re.search(r"^type$", text):
        return True, lucid.nn.Linear, False
    if re.search(r"PackedSequence", text):
        packed = _packed_sequence(domain)
        if packed is not None:
            return True, packed, False
    if name in ("data", "arr", "ext_tensor", "array"):
        # The bridge functions — ``tensor``, ``as_tensor``, ``from_numpy``.
        # Annotated ``object`` or ``np.ndarray``, so only the name says
        # what they want, and what they want is not a Lucid tensor.
        return True, np.asarray(_probe.sample(domain, shape)), True

    if name in _BY_NAME:
        return True, _BY_NAME[name], False

    choice = _literal_choice(text)
    if choice is not None:
        return True, choice, False

    if re.search(r"\bbool\b", text):
        return True, False, False
    if re.search(r"tuple\[int|Sequence\[int\]|list\[int\]|_IntTuple|_Shape", text):
        return True, (2, 2), False
    if re.search(r"\bint\b", text):
        return True, 2, False
    if re.search(r"\bfloat\b", text):
        return True, 0.5, False
    if re.search(r"\bstr\b", text):
        return True, "mean", False
    if re.search(r"\bdtype\b|Dtype", text):
        return True, lucid.float32, False
    if re.search(r"\bdevice\b|Device", text):
        return True, "cpu", False
    if "None" in text:
        # ``T | None`` with no default still accepts None, and a None is a
        # truthful "not supplied" rather than a guess.
        return True, None, False
    return False, None, False


def _build(
    fn: Any, op_name: str, domain: str, shape: "tuple[int, ...]"
) -> "_Plan | None":
    try:
        signature = inspect.signature(fn, annotation_format=annotationlib.Format.STRING)
    except TypeError, ValueError, NameError:
        return None

    plan = _Plan()
    first_tensor: "int | None" = None
    reference: Any = None
    for name, param in signature.parameters.items():
        # ``self`` is *not* skipped.  A Tensor method is enumerated with
        # ``getattr_static`` and so arrives unbound — the receiver is a
        # real positional parameter and has to be supplied, or 68 of the
        # 253 methods look like they take an argument nothing can fill.
        if name == "cls":
            continue
        if param.kind is param.VAR_KEYWORD:
            continue
        if param.kind is param.VAR_POSITIONAL:
            # ``f(*tensors)`` — two of the primary shape is the reading
            # that makes ``stack`` and ``cat`` mean something.
            plan.args.append(_tensor(shape, domain))
            plan.args.append(_tensor(shape, domain))
            if first_tensor is None:
                first_tensor, reference = len(plan.args) - 2, plan.args[-2]
            continue
        if param.default is not param.empty:
            continue

        found, value, is_tensor = _value_for(param, op_name, domain, reference, shape)
        if not found:
            plan.unknown.append(
                f"{name}: {_clean(param.annotation) or '<no annotation>'}"
            )
            return None
        if param.kind is param.KEYWORD_ONLY:
            plan.kwargs[name] = value
        else:
            plan.args.append(value)
            if is_tensor and first_tensor is None:
                first_tensor, reference = len(plan.args) - 1, value

    if first_tensor is None:
        # No tensor *input* is not the same as nothing to check.
        # ``windows.bartlett(8)`` builds a tensor from an int,
        # ``calculate_gain("relu")`` returns a float, ``fuse_conv_bn_eval``
        # takes two modules — all callable, all worth running, none of
        # them differentiable with respect to an argument.  Discarding
        # these left 35 symbols on the ladder for want of an operand they
        # do not have.  The gradient axes ask ``Call.base`` for a tensor,
        # get a TypeError and skip, which is the truthful outcome; the
        # smoke, edge and dtype axes still get their answer.
        if not plan.args and not plan.kwargs:
            return None
        plan.primary = 0 if plan.args else -1
        plan.note = f"derived from signature{signature}, no tensor operand"
        return plan
    plan.primary = first_tensor
    plan.note = f"derived from signature{signature} at {shape}"
    return plan


def invocations(fn: Any, op_name: str, domain: str) -> "Iterator[Call]":
    """Candidate calls for ``fn``, derived from its signature.

    Yields nothing when the signature cannot be read or leaves a required
    parameter this module has no value for — that is the honest answer,
    and :mod:`_specs` falls through to its generic ladder behind it.

    Parameters
    ----------
    fn : callable
        The resolved callable, not the :class:`~lucid.test.audit.
        _surface.Symbol`.
    op_name : str
        Short name, used for the handful of shape decisions a signature
        cannot express — ``conv2d`` wants a 4-D input, ``cross_entropy``
        wants an integer target.
    domain : str
        Key into :data:`~lucid.test.audit._probe.DOMAINS`.
    """
    emitted = 0
    for shape in _tensor_shapes(op_name):
        plan = _build(fn, op_name, domain, shape)
        if plan is None:
            continue
        yield Call(plan.args, plan.kwargs, plan.primary, plan.note)
        emitted += 1
        if emitted >= _MAX_CANDIDATES:
            return


def explain(fn: Any, op_name: str) -> str:
    """Why a symbol gets no derived invocation, for ``--list-uncovered``."""
    try:
        signature = inspect.signature(fn, annotation_format=annotationlib.Format.STRING)
    except (TypeError, ValueError, NameError) as exc:
        return f"unreadable signature: {type(exc).__name__}"
    for shape in _tensor_shapes(op_name):
        plan = _Plan()
        probe = _build(fn, op_name, "moderate", shape)
        if probe is not None:
            return "derived"
        del plan
    required = [
        f"{p.name}: {_clean(p.annotation) or '<no annotation>'}"
        for p in signature.parameters.values()
        if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    return f"no value for {required}" if required else "no tensor parameter"


__all__ = ["explain", "invocations"]
