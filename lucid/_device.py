"""Lucid device descriptors.

Defines :class:`device` — the runtime handle that pins a tensor to either
the CPU stream (Apple Accelerate) or the Metal GPU stream (MLX). Only
``"cpu"`` and ``"metal"`` are valid device types; Lucid is Apple Silicon
only.
"""

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
        """Parse a device specifier.

        Parameters
        ----------
        type_or_str : str
            Either a bare device type (``"cpu"`` / ``"metal"``) or a
            ``"<type>:<index>"`` string (e.g. ``"metal:0"``).
        index : int, optional
            Device index — only ``0`` is meaningful on current Apple
            Silicon hardware. Ignored when ``type_or_str`` already
            contains an index suffix.

        Raises
        ------
        ValueError
            If the device type is not ``"cpu"`` or ``"metal"``.

        Examples
        --------
        >>> import lucid
        >>> lucid.device("cpu")
        device('cpu')
        >>> lucid.device("metal:0").is_metal
        True
        """
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
        """``True`` if this device targets the Metal GPU stream."""
        return self.type in self._METAL_TYPES

    def __repr__(self) -> str:
        """Return ``"device('<type>')"``."""
        return f"device('{self.type}')"

    def __eq__(self, other: object) -> bool:
        """Compare against another :class:`device` or a device-type string.

        Two devices are equal iff they map to the same engine stream
        (``CPU`` or ``GPU``).
        """
        if isinstance(other, str):
            return other == self.type
        if isinstance(other, device):
            return self._engine == other._engine
        return NotImplemented

    def __hash__(self) -> int:
        """Hash by engine stream so equal devices hash equal."""
        return hash(self._engine)


def _device_from_engine(d: _C_engine.Device) -> device:
    """Convert engine Device enum → lucid device."""
    return device("metal" if d == _C_engine.Device.GPU else "cpu")
