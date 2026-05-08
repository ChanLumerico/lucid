from lucid._C import engine as _C_engine


class dtype:
    """Lucid scalar data type. Singletons compared by identity (is)."""

    def __init__(
        self,
        name: str,
        engine_dtype: _C_engine.Dtype,
        itemsize: int,
    ) -> None:
        self._name = name
        self._engine = engine_dtype
        self.itemsize = itemsize

    def __repr__(self) -> str:
        return f"lucid.{self._name}"

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


float16: dtype = dtype("float16", _C_engine.Dtype.F16, 2)
float32: dtype = dtype("float32", _C_engine.Dtype.F32, 4)
float64: dtype = dtype("float64", _C_engine.Dtype.F64, 8)
bfloat16: dtype = dtype("bfloat16", _C_engine.Dtype.F16, 2)
int8: dtype = dtype("int8", _C_engine.Dtype.I8, 1)
int16: dtype = dtype("int16", _C_engine.Dtype.I16, 2)
int32: dtype = dtype("int32", _C_engine.Dtype.I32, 4)
int64: dtype = dtype("int64", _C_engine.Dtype.I64, 8)
bool_: dtype = dtype("bool", _C_engine.Dtype.Bool, 1)
complex64: dtype = dtype("complex64", _C_engine.Dtype.C64, 8)

half: dtype = float16
double: dtype = float64
short: dtype = int16
long: dtype = int64

_ENGINE_TO_DTYPE: dict[_C_engine.Dtype, dtype] = {
    _C_engine.Dtype.F16: float16,
    _C_engine.Dtype.F32: float32,
    _C_engine.Dtype.F64: float64,
    _C_engine.Dtype.I8: int8,
    _C_engine.Dtype.I16: int16,
    _C_engine.Dtype.I32: int32,
    _C_engine.Dtype.I64: int64,
    _C_engine.Dtype.Bool: bool_,
    _C_engine.Dtype.C64: complex64,
}

_NAME_TO_DTYPE: dict[str, dtype] = {
    "float16": float16,
    "float32": float32,
    "float64": float64,
    "bfloat16": bfloat16,
    "int8": int8,
    "int16": int16,
    "int32": int32,
    "int64": int64,
    "bool": bool_,
    "bool_": bool_,
    "complex64": complex64,
    "half": float16,
    "float": float32,
    "double": float64,
    "short": int16,
    "int": int32,
    "long": int64,
}


def to_engine_dtype(
    d: dtype | _C_engine.Dtype | str | None,
) -> _C_engine.Dtype:
    """Convert lucid dtype, engine Dtype, string name, or None → engine Dtype."""
    if d is None:
        return _C_engine.Dtype.F32
    if isinstance(d, _C_engine.Dtype):
        return d
    if isinstance(d, dtype):
        return d._engine
    if isinstance(d, str):
        name = d.lower()
        if name in _NAME_TO_DTYPE:
            return _NAME_TO_DTYPE[name]._engine
        raise ValueError(f"Unknown dtype string: {d!r}")
    raise TypeError(f"Cannot convert {type(d).__name__} to engine Dtype")


# ── finfo / iinfo ────────────────────────────────────────────────────────────


import dataclasses as _dc


def _resolve_dtype_name(d: "dtype | _C_engine.Dtype | str") -> str:  # type: ignore[name-defined]
    """Map any dtype-like input to its canonical lucid dtype name.

    Used by ``finfo`` / ``iinfo`` to discriminate ``float16`` from
    ``bfloat16`` (both share engine enum ``F16``) and to produce the
    user-facing ``dtype: str`` field on the info objects.
    """
    if isinstance(d, dtype):
        return d._name
    if isinstance(d, str):
        name = d.lower()
        if name in _NAME_TO_DTYPE:
            return _NAME_TO_DTYPE[name]._name
        raise ValueError(f"Unknown dtype string: {d!r}")
    # Engine Dtype enum — best-effort reverse lookup (loses bfloat16/float16
    # distinction; both report ``float16``).
    if isinstance(d, _C_engine.Dtype):
        for nm, lucid_dt in _NAME_TO_DTYPE.items():
            if lucid_dt._engine == d:
                return lucid_dt._name
    raise TypeError(f"Cannot resolve dtype name for {type(d).__name__}: {d!r}")


@_dc.dataclass(frozen=True)
class finfo:
    """Floating-point dtype info — mirrors ``numpy.finfo`` (and the reference framework's ``finfo``).

    Constructed from a floating dtype (``float16`` / ``bfloat16`` /
    ``float32`` / ``float64``).
    """

    bits: int = 0
    eps: float = 0.0
    max: float = 0.0
    min: float = 0.0
    tiny: float = 0.0
    smallest_normal: float = 0.0
    resolution: float = 0.0
    dtype: str = ""

    def __init__(self, dt: "dtype | _C_engine.Dtype | str") -> None:  # type: ignore[name-defined]
        # Resolve to a lucid dtype name first — bfloat16 and float16 share the
        # same engine enum (F16), so we can't distinguish them via the engine.
        name = _resolve_dtype_name(dt)
        table: dict[str, tuple[int, float, float, float, float, float]] = {
            "float64": (
                64,
                2.220446049250313e-16,
                1.7976931348623157e308,
                -1.7976931348623157e308,
                2.2250738585072014e-308,
                1e-15,
            ),
            "float32": (
                32,
                1.1920929e-7,
                3.4028235e38,
                -3.4028235e38,
                1.1754944e-38,
                1e-6,
            ),
            "float16": (16, 9.7656e-4, 65504.0, -65504.0, 6.1035e-5, 1e-3),
            "bfloat16": (
                16,
                7.8125e-3,
                3.3895314e38,
                -3.3895314e38,
                1.1754944e-38,
                1e-2,
            ),
        }
        if name not in table:
            raise TypeError(f"finfo expects a floating dtype, got {dt!r}")
        bits, eps, mx, mn, tiny, res = table[name]
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "eps", eps)
        object.__setattr__(self, "max", mx)
        object.__setattr__(self, "min", mn)
        object.__setattr__(self, "tiny", tiny)
        object.__setattr__(self, "smallest_normal", tiny)
        object.__setattr__(self, "resolution", res)
        object.__setattr__(self, "dtype", name)


@_dc.dataclass(frozen=True)
class iinfo:
    """Integer dtype info — mirrors ``numpy.iinfo`` (and the reference framework's ``iinfo``).

    Constructed from an integer dtype (``int8`` / ``int16`` / ``int32`` / ``int64``).
    """

    bits: int = 0
    max: int = 0
    min: int = 0
    dtype: str = ""

    def __init__(self, dt: "dtype | _C_engine.Dtype | str") -> None:  # type: ignore[name-defined]
        name = _resolve_dtype_name(dt)
        table: dict[str, tuple[int, int, int]] = {
            "int8": (8, -(2**7), 2**7 - 1),
            "int16": (16, -(2**15), 2**15 - 1),
            "int32": (32, -(2**31), 2**31 - 1),
            "int64": (64, -(2**63), 2**63 - 1),
        }
        if name not in table:
            raise TypeError(f"iinfo expects an integer dtype, got {dt!r}")
        bits, lo, hi = table[name]
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "min", lo)
        object.__setattr__(self, "max", hi)
        object.__setattr__(self, "dtype", name)
