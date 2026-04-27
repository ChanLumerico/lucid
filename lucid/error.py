"""
lucid.error — exception classes raised by the Python wrapper layer.

The C++ engine (`lucid._C.engine`) raises its own typed exceptions
(`LucidError`, `DtypeMismatch`, `ShapeMismatch`, `DeviceMismatch`,
`OutOfMemory`, `NotImplementedError`, `IndexError`, `VersionMismatch`)
which propagate up through pybind11 unchanged. The classes here are the
Python-side counterparts that callers historically caught — they are
preserved for backward compatibility.
"""

from __future__ import annotations

from lucid.types import _DeviceType, _ShapeLike


__all__ = ["UnknownDeviceError", "DeviceMismatchError", "BackwardError"]


class UnknownDeviceError(Exception):
    """Raised when a device string is neither 'cpu' nor 'gpu'."""

    def __init__(self, device: str) -> None:
        super().__init__(
            f"Unknown device '{device}'. Must be either 'cpu' or 'gpu'."
        )


class DeviceMismatchError(Exception):
    """Raised when a tensor on one device is accessed from another."""

    def __init__(self, to: _DeviceType, from_: _DeviceType) -> None:
        super().__init__(
            f"Attempted access of '{to}' tensor from '{from_}' tensor."
        )


class BackwardError(Exception):
    """Wraps an exception raised during backward, carrying the offending
    tensor's shape and the op that emitted it."""

    def __init__(self, shape: _ShapeLike, op: object) -> None:
        super().__init__(
            f"Exception above occurred for tensor of shape {shape}"
            f" on operation {op}."
        )
