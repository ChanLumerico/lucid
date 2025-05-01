from lucid.types import _DeviceType, _ShapeLike


__all__ = ["UnknownDeviceError", "DeviceMismatchError", "BackwardError"]


class UnknownDeviceError(Exception):
    def __init__(self, device: str) -> None:
        super().__init__(f"Unknown device '{device}'. Must be either 'cpu' or 'gpu'.")


class DeviceMismatchError(Exception):
    def __init__(self, to: _DeviceType, from_: _DeviceType) -> None:
        super().__init__(f"Attempted access of '{to}' tensor from '{from_}' tensor.")


class BackwardError(Exception):
    def __init__(self, shape: _ShapeLike, op: object) -> None:
        super().__init__(
            f"Exception above occurred for tensor of shape {shape} on operation {op}."
        )
