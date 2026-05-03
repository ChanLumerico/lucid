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
float: dtype = float32
double: dtype = float64
short: dtype = int16
int: dtype = int32
long: dtype = int64
bool: dtype = bool_

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
    d: "dtype | _C_engine.Dtype | str | None",
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
