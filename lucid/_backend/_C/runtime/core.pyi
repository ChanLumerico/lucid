from __future__ import annotations
import typing

__all__: list[str] = list()

class _C_FuncOpSpec:
    device: str
    has_gradient: bool
    def __init__(self) -> None: ...
    @property
    def n_in(self) -> int | None: ...
    @n_in.setter
    def n_in(self, arg0: typing.SupportsInt | None) -> None: ...
    @property
    def n_ret(self) -> int | None: ...
    @n_ret.setter
    def n_ret(self, arg0: typing.SupportsInt | None) -> None: ...

def _C_func_op(
    op_self: typing.Any,
    forward_func: typing.Any,
    args: tuple,
    kwargs: dict,
    spec: _C_FuncOpSpec,
) -> typing.Any: ...
def _C_func_op_raw(
    op_self: typing.Any,
    forward_func: typing.Any,
    args: tuple,
    kwargs: dict,
    n_in: typing.SupportsInt | None = None,
    n_ret: typing.SupportsInt | None = None,
    has_gradient: bool = True,
    device: str = "cpu",
) -> typing.Any: ...
