from lucid._C import engine as _C_engine


class device:
    """
    Lucid device descriptor.
    device("metal") → Apple GPU (Metal)
    device("cpu")   → CPU (Apple Accelerate)
    """

    _METAL_TYPES: frozenset[str] = frozenset({"metal"})
    _VALID_TYPES: frozenset[str] = frozenset({"metal", "cpu"})

    def __init__(self, type_or_str: str, index: int = 0) -> None:
        if ":" in type_or_str:
            t, idx = type_or_str.split(":", 1)
            self.type: str = t.strip()
            self.index: int = int(idx)
        else:
            self.type = type_or_str.strip()
            self.index = index
        if self.type not in self._VALID_TYPES:
            raise ValueError(f"Unknown device '{self.type}'. Use 'metal' or 'cpu'.")
        self._engine: _C_engine.Device = (
            _C_engine.Device.GPU
            if self.type in self._METAL_TYPES
            else _C_engine.Device.CPU
        )

    @property
    def is_metal(self) -> bool:
        return self.type in self._METAL_TYPES

    def __repr__(self) -> str:
        return f"device('{self.type}')"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other == self.type
        if isinstance(other, device):
            return self._engine == other._engine
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._engine)


def _device_from_engine(d: _C_engine.Device) -> device:
    """Convert engine Device enum → lucid device."""
    return device("metal" if d == _C_engine.Device.GPU else "cpu")
