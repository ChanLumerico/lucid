"""Every in-place entry point writes into ``self``, as the reference's does (LCD-519).

An in-place op has more than one spelling — the augmented assignment
``a //= b``, the method ``a.floor_divide_(b)`` — and each must keep the one
contract: the values land in ``self``'s storage, so ``self`` stays the same
object and its views and aliases see them; the version counter moves; and a
result dtype ``self`` cannot hold is refused before anything is written, with
the error ``add_`` gives.  Two families broke it:

* ``//= %= |= &= ^= <<= >>=`` had no ``__i<op>__`` at all, so Python fell
  back to the out-of-place operator and rebound the name: ``v = a[:2];
  a //= 2.0`` left ``v`` at the old values and made ``a`` a new tensor.  Their
  methods (``floor_divide_``, ``remainder_``, ``bitwise_or_`` ...) were
  missing too.
* ``clamp_`` / ``clamp_min_`` / ``clamp_max_`` called the engine kernel with
  the bounds as given, skipping the promotion ``clamp`` does and the in-place
  cast check: ``int64.clamp_(0.5, 1.5)`` wrote ``[0, 1, 1, 1]``.

The tables below are the whole family: the augmented assignments the
reference writes in place, and the ``*_`` methods with an operand.
"""

import operator
import re
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

# The reference's in-place refusal, whichever op raises it.
_CANNOT_CAST = re.compile(r"result type .+ can't be cast to the desired output type .+")

# ── augmented assignment ─────────────────────────────────────────────────────

#: Each augmented assignment: its symbol, a ``self`` of the kind it computes
#: on, and an operand.  ``self`` is ``[5, 6, 7, 8]``.
AUGMENTED: dict[str, tuple[str, str, object]] = {
    "__iadd__": ("+=", "float32", 2.0),
    "__isub__": ("-=", "float32", 2.0),
    "__imul__": ("*=", "float32", 2.0),
    "__itruediv__": ("/=", "float32", 2.0),
    "__ifloordiv__": ("//=", "float32", 2.0),
    "__imod__": ("%=", "float32", 3.0),
    "__ipow__": ("**=", "float32", 2.0),
    "__iand__": ("&=", "int64", 3),
    "__ior__": ("|=", "int64", 3),
    "__ixor__": ("^=", "int64", 3),
    "__ilshift__": ("<<=", "int64", 1),
    "__irshift__": (">>=", "int64", 1),
}


def _base(lib: ModuleType, dtype: str) -> object:
    return lib.tensor([5, 6, 7, 8], dtype=getattr(lib, dtype))


def _augment(target: object, symbol: str, operand: object) -> object:
    """``target <symbol> operand`` as a statement; the name's value afterwards."""
    scope = {"target": target, "operand": operand}
    exec(f"target {symbol} operand", scope)
    return scope["target"]


def test_the_table_is_every_augmented_assignment_the_reference_writes_in_place(
    ref: ModuleType,
) -> None:
    # The Python operators that have an in-place form, as ``operator`` lists
    # them, which the reference's tensor defines beside the operator itself.
    in_place = {
        name
        for name in dir(ref.Tensor)
        if name.startswith("__i")
        and hasattr(operator, name)
        and hasattr(ref.Tensor, "__" + name[3:])
    }
    assert in_place == set(AUGMENTED)


@pytest.mark.parametrize("dunder", list(AUGMENTED))
def test_lucid_defines_the_in_place_operator(dunder: str) -> None:
    # Without it Python falls back to the out-of-place one and rebinds.
    assert dunder in vars(lucid.Tensor)


@pytest.mark.parametrize("dunder", list(AUGMENTED))
def test_the_operator_writes_into_self(device: str, dunder: str) -> None:
    symbol, dtype, operand = AUGMENTED[dunder]
    a = _base(lucid, dtype).to(device)
    alias = a
    # ``_impl.version``: Lucid has no ``Tensor._version`` yet (LCD-488).
    impl, version = a._impl, a._impl.version
    expected = getattr(_base(lucid, dtype), "__" + dunder[3:])(operand).tolist()

    out = _augment(a, symbol, operand)

    assert out is alias
    assert a._impl is impl
    assert a._impl.version == version + 1
    assert a.dtype == getattr(lucid, dtype)
    assert alias.tolist() == expected


@pytest.mark.parametrize(
    "dev",
    [
        "cpu",
        pytest.param(
            "metal",
            marks=[
                pytest.mark.skipif(not metal_available(), reason="metal unavailable"),
                pytest.mark.xfail(
                    strict=True,
                    reason="LCD-18: a Metal view taken before its base is written "
                    "does not see the write (lucid/_tensor/_metal_views.py)",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize("dunder", list(AUGMENTED))
def test_a_view_of_self_sees_the_write(dev: str, dunder: str) -> None:
    symbol, dtype, operand = AUGMENTED[dunder]
    a = _base(lucid, dtype).to(dev)
    view = a[:2]
    expected = getattr(_base(lucid, dtype), "__" + dunder[3:])(operand).tolist()
    _augment(a, symbol, operand)
    assert view.tolist() == expected[:2]


@pytest.mark.parametrize("dunder", list(AUGMENTED))
def test_the_operator_on_a_view_writes_through_to_its_base(
    device: str, dunder: str
) -> None:
    symbol, dtype, operand = AUGMENTED[dunder]
    a = _base(lucid, dtype).to(device)
    view = a[1:3]
    expected = getattr(_base(lucid, dtype), "__" + dunder[3:])(operand).tolist()
    _augment(view, symbol, operand)
    assert a.tolist() == [5, *expected[1:3], 8]


def _refusal(fn: Callable[[], object]) -> BaseException:
    """What ``fn`` raised — every refusal here is a RuntimeError or a TypeError."""
    with pytest.raises((RuntimeError, TypeError)) as info:
        fn()
    return info.value


@pytest.mark.parametrize("operand", [2.5, "tensor"], ids=["number", "tensor"])
@pytest.mark.parametrize("dunder", list(AUGMENTED))
def test_an_integer_self_refuses_a_float_result_as_the_reference_does(
    ref: ModuleType, device: str, dunder: str, operand: object
) -> None:
    symbol = AUGMENTED[dunder][0]

    def write(lib: ModuleType, to: str | None) -> Callable[[], object]:
        a = _base(lib, "int64")
        other = lib.tensor([1.5, 2.5, 3.5, 4.5]) if operand == "tensor" else operand
        if to is not None:
            a = a.to(to)
            other = other.to(to) if operand == "tensor" else other
        return lambda: _augment(a, symbol, other)

    theirs = _refusal(write(ref, None))
    a = _base(lucid, "int64").to(device)
    other = (
        lucid.tensor([1.5, 2.5, 3.5, 4.5]).to(device)
        if operand == "tensor"
        else operand
    )
    ours = _refusal(lambda: _augment(a, symbol, other))
    assert isinstance(ours, type(theirs))
    assert _CANNOT_CAST.search(str(theirs))
    assert _CANNOT_CAST.search(str(ours))
    assert a.tolist() == [5, 6, 7, 8]


# ── methods ─────────────────────────────────────────────────────────────────


def _floats(lib: ModuleType, values: list[float]) -> object:
    return lib.tensor(values, dtype=lib.float32)


#: Each in-place method with its ``self`` and its arguments, built for
#: either library.  ``self`` holds four elements.
METHODS: dict[
    str, tuple[Callable[[ModuleType], object], Callable[[ModuleType], tuple]]
] = {
    "add_": (lambda lib: _floats(lib, [1, 2, 3, 4]), lambda lib: (2.5,)),
    "sub_": (lambda lib: _floats(lib, [1, 2, 3, 4]), lambda lib: (2.5,)),
    "mul_": (lambda lib: _floats(lib, [1, 2, 3, 4]), lambda lib: (2.5,)),
    "div_": (lambda lib: _floats(lib, [1, 2, 3, 4]), lambda lib: (2.5,)),
    "pow_": (lambda lib: _floats(lib, [1, 2, 3, 4]), lambda lib: (2.0,)),
    "remainder_": (lambda lib: _floats(lib, [-7, 7, 5.5, -2]), lambda lib: (3.0,)),
    "fmod_": (lambda lib: _floats(lib, [-7, 7, 5.5, -2]), lambda lib: (3.0,)),
    "floor_divide_": (lambda lib: _floats(lib, [-7, 7, 5.5, -2]), lambda lib: (2.0,)),
    "bitwise_and_": (lambda lib: lib.tensor([12, 10, 7, 0]), lambda lib: (6,)),
    "bitwise_or_": (lambda lib: lib.tensor([12, 10, 7, 0]), lambda lib: (6,)),
    "bitwise_xor_": (lambda lib: lib.tensor([12, 10, 7, 0]), lambda lib: (6,)),
    "bitwise_left_shift_": (lambda lib: lib.tensor([1, 2, 3, 4]), lambda lib: (2,)),
    "bitwise_right_shift_": (lambda lib: lib.tensor([8, 9, 16, 4]), lambda lib: (2,)),
    "bitwise_not_": (lambda lib: lib.tensor([0, 5, -1, 3]), lambda lib: ()),
    "logical_and_": (
        lambda lib: _floats(lib, [0, 1, 2, 0]),
        lambda lib: (_floats(lib, [1, 0, 3, 0]),),
    ),
    "logical_or_": (
        lambda lib: _floats(lib, [0, 1, 2, 0]),
        lambda lib: (_floats(lib, [1, 0, 3, 0]),),
    ),
    "logical_xor_": (
        lambda lib: _floats(lib, [0, 1, 2, 0]),
        lambda lib: (_floats(lib, [1, 0, 3, 0]),),
    ),
    "logical_not_": (lambda lib: lib.tensor([0, 1, 2, 0]), lambda lib: ()),
    "lerp_": (
        lambda lib: _floats(lib, [0, 1, 2, 3]),
        lambda lib: (_floats(lib, [4, 4, 4, 4]), 0.25),
    ),
    "addcmul_": (
        lambda lib: _floats(lib, [1, 2, 3, 4]),
        lambda lib: (_floats(lib, [1, 2, 3, 4]), _floats(lib, [0.5, 0.5, 2, 2])),
    ),
    "addcdiv_": (
        lambda lib: _floats(lib, [1, 2, 3, 4]),
        lambda lib: (_floats(lib, [1, 2, 3, 4]), _floats(lib, [0.5, 0.5, 2, 2])),
    ),
    "clamp_": (lambda lib: _floats(lib, [-1, 0.5, 2, 4]), lambda lib: (0.0, 2.5)),
    "clip_": (lambda lib: _floats(lib, [-1, 0.5, 2, 4]), lambda lib: (0.0, 2.5)),
    "clamp_min_": (lambda lib: _floats(lib, [-1, 0.5, 2, 4]), lambda lib: (0.5,)),
    "clamp_max_": (lambda lib: _floats(lib, [-1, 0.5, 2, 4]), lambda lib: (2.0,)),
}

#: The methods whose result dtype follows their operands', each with
#: arguments that make it float for an ``int64`` self.  A logical op answers
#: in bool and ``bitwise_not_`` takes no operand, so neither can be refused.
KIND_DOWN: dict[str, Callable[[ModuleType], tuple]] = {
    "add_": lambda lib: (2.5,),
    "sub_": lambda lib: (2.5,),
    "mul_": lambda lib: (2.5,),
    "div_": lambda lib: (2,),
    "pow_": lambda lib: (0.5,),
    "remainder_": lambda lib: (2.5,),
    "fmod_": lambda lib: (2.5,),
    "floor_divide_": lambda lib: (2.5,),
    "bitwise_and_": lambda lib: (2.5,),
    "bitwise_or_": lambda lib: (2.5,),
    "bitwise_xor_": lambda lib: (2.5,),
    "bitwise_left_shift_": lambda lib: (_floats(lib, [1, 1, 1, 1]),),
    "bitwise_right_shift_": lambda lib: (_floats(lib, [1, 1, 1, 1]),),
    "addcmul_": lambda lib: (_floats(lib, [1, 2, 3, 4]), _floats(lib, [1, 2, 3, 4])),
    "addcdiv_": lambda lib: (_floats(lib, [1, 2, 3, 4]), _floats(lib, [1, 2, 3, 4])),
    "clamp_": lambda lib: (0.5, 1.5),
    "clip_": lambda lib: (0.5, 1.5),
    "clamp_min_": lambda lib: (0.5,),
    "clamp_max_": lambda lib: (1.5,),
    "clamp_ tensor bounds": lambda lib: (
        _floats(lib, [0, 0, 0, 0]),
        _floats(lib, [1, 1, 1, 1]),
    ),
}


def _on(device: str, args: tuple) -> tuple:
    return tuple(a.to(device) if isinstance(a, lucid.Tensor) else a for a in args)


@pytest.mark.parametrize("name", list(METHODS))
def test_lucid_has_every_in_place_method_the_table_names(
    ref: ModuleType, name: str
) -> None:
    assert hasattr(ref.Tensor, name)
    assert hasattr(lucid.Tensor, name)


@pytest.mark.parametrize("name", list(METHODS))
def test_the_method_writes_into_self_as_the_reference_does(
    ref: ModuleType, device: str, name: str
) -> None:
    make, args = METHODS[name]
    expected = getattr(make(ref), name)(*args(ref))
    a = make(lucid).to(device)
    impl, version = a._impl, a._impl.version

    out = getattr(a, name)(*_on(device, args(lucid)))

    assert out is a
    assert a._impl is impl
    assert a._impl.version == version + 1
    assert str(a.dtype).split(".")[-1] == str(expected.dtype).split(".")[-1]
    assert a.tolist() == pytest.approx(expected.tolist())


@pytest.mark.parametrize("name", list(METHODS))
def test_the_method_on_a_view_writes_through_to_its_base(
    device: str, name: str
) -> None:
    make, args = METHODS[name]
    whole = getattr(make(lucid), name)(*args(lucid)).tolist()
    a = make(lucid).to(device)
    before = a.tolist()
    view = a[1:3]
    # The operands are cut to the view's shape where they are tensors.
    sliced = tuple(x[1:3] if isinstance(x, lucid.Tensor) else x for x in args(lucid))
    getattr(view, name)(*_on(device, sliced))
    assert a.tolist() == pytest.approx([before[0], *whole[1:3], before[3]])


@pytest.mark.parametrize("name", list(METHODS))
def test_a_view_of_self_sees_the_methods_write_on_the_cpu(name: str) -> None:
    make, args = METHODS[name]
    a = make(lucid)
    view = a[:2]
    getattr(a, name)(*args(lucid))
    assert view.tolist() == a.tolist()[:2]


@pytest.mark.parametrize("name", list(KIND_DOWN))
def test_an_integer_self_refuses_a_float_result_before_writing(
    ref: ModuleType, device: str, name: str
) -> None:
    method = name.split()[0]
    theirs = _refusal(
        lambda: getattr(ref.tensor([0, 1, 2, 3]), method)(*KIND_DOWN[name](ref))
    )
    a = lucid.tensor([0, 1, 2, 3]).to(device)
    ours = _refusal(lambda: getattr(a, method)(*_on(device, KIND_DOWN[name](lucid))))
    assert isinstance(ours, type(theirs))
    assert _CANNOT_CAST.search(str(theirs))
    assert _CANNOT_CAST.search(str(ours))
    assert a.tolist() == [0, 1, 2, 3]


def test_lerp_into_an_integer_self_is_refused_as_lerp_refuses_it(device: str) -> None:
    a = lucid.tensor([0, 1]).to(device)
    with pytest.raises(NotImplementedError, match="floating-point"):
        a.lerp_(lucid.tensor([1.0, 2.0]).to(device), 0.5)
    assert a.tolist() == [0, 1]


#: The float methods, whose leaf that requires grad is refused while
#: autograd records.
_DIFFERENTIABLE = [
    "remainder_",
    "fmod_",
    "floor_divide_",
    "lerp_",
    "addcmul_",
    "addcdiv_",
    "clamp_",
    "clip_",
    "clamp_min_",
    "clamp_max_",
]


@pytest.mark.parametrize("name", _DIFFERENTIABLE)
def test_a_leaf_that_requires_grad_is_not_written(device: str, name: str) -> None:
    make, args = METHODS[name]
    leaf = make(lucid).to(device).requires_grad_()
    before = leaf.tolist()
    with pytest.raises(RuntimeError, match="leaf"):
        getattr(leaf, name)(*_on(device, args(lucid)))
    assert leaf.tolist() == before
    with lucid.no_grad():
        getattr(leaf, name)(*_on(device, args(lucid)))
    assert leaf.is_leaf and leaf.requires_grad


@pytest.mark.parametrize(
    # The reference has no derivative for floor division.
    "name",
    [n for n in _DIFFERENTIABLE if n != "floor_divide_"],
)
def test_the_gradient_through_the_write_is_the_references(
    ref: ModuleType, device: str, name: str
) -> None:
    make, args = METHODS[name]

    def grad(lib: ModuleType, to: str | None) -> list[float]:
        x = make(lib)
        x = (x.to(to) if to else x).requires_grad_()
        operands = args(lib)
        operands = _on(to, operands) if to else operands
        y = x * 1.0
        getattr(y, name)(*operands)
        (y * y).sum().backward()
        return x.grad.tolist()

    assert grad(lucid, device) == pytest.approx(grad(ref, None))


# ── clamp: the bounds promote as clamp's do ─────────────────────────────────


@pytest.mark.parametrize(
    "bounds",
    [
        (0.5, 2.5),
        (1, 2),
        ("tensor", "tensor"),
        ("0-d", "0-d"),
        (None, 2),
        (1, None),
    ],
    ids=str,
)
@pytest.mark.parametrize("dtype", ["float32", "float16", "int64", "int32"])
def test_clamp_writes_what_clamp_returns_or_refuses(
    ref: ModuleType, device: str, dtype: str, bounds: tuple
) -> None:
    def spell(lib: ModuleType, bound: object, at: float) -> object:
        if bound == "tensor":
            return lib.tensor([at, at, at, at])
        if bound == "0-d":
            return lib.tensor(at)
        return bound

    def lo(lib: ModuleType) -> object:
        return spell(lib, bounds[0], 0.5)

    def hi(lib: ModuleType) -> object:
        return spell(lib, bounds[1], 2.5)

    values = [-1, 0, 2, 4]

    def theirs() -> object:
        return ref.tensor(values, dtype=getattr(ref, dtype)).clamp_(lo(ref), hi(ref))

    a = lucid.tensor(values, dtype=getattr(lucid, dtype)).to(device)
    try:
        expected = theirs()
    except RuntimeError as exc:
        ours = _refusal(lambda: a.clamp_(*_on(device, (lo(lucid), hi(lucid)))))
        assert isinstance(ours, RuntimeError)
        assert _CANNOT_CAST.search(str(exc)) and _CANNOT_CAST.search(str(ours))
        assert a.tolist() == values
        return
    out = a.clamp_(*_on(device, (lo(lucid), hi(lucid))))
    assert out is a
    assert a.dtype == getattr(lucid, dtype)
    assert a.tolist() == pytest.approx(expected.tolist())


def test_clamp_without_a_bound_is_refused_as_clamp_refuses_it() -> None:
    a = lucid.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="at least one"):
        a.clamp_()
    with pytest.raises(ValueError, match="at least one"):
        a.clamp(None, None)
