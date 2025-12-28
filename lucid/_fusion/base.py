from abc import ABC, abstractmethod
from typing import ClassVar, Callable, Never, Sequence, overload
from types import ModuleType

from functools import partial
import inspect
import numpy as np

from lucid._tensor.tensor import Tensor
from lucid._backend.core import Operation, _GradType
from lucid._backend.metal import mx
from lucid.types import _DeviceType


__all__ = ["FusedBackwardOp", "match_fusion_table"]


_lib_mapping: dict[_DeviceType, ModuleType] = {"cpu": np, "metal": mx}


class FusedBackwardOp(ABC):
    op1: ClassVar[type[Operation] | None] = None
    op2: ClassVar[type[Operation] | None] = None

    @classmethod
    def get_fused_grad_func(
        cls,
        inputs: Tensor | Sequence[Tensor],
        results: Tensor | Sequence[Tensor],
        device: _DeviceType = "cpu",
    ) -> Callable[[], _GradType]:
        if isinstance(inputs, Sequence) and not isinstance(inputs, Tensor):
            ins: tuple[Tensor, ...] = tuple(inputs)
        else:
            ins = (inputs,)

        if isinstance(results, Sequence) and not isinstance(results, Tensor):
            rets: tuple[Tensor, ...] = tuple(results)
        else:
            rets = (results,)

        sig = inspect.signature(cls.__grad__)
        params = sig.parameters
        bound: dict[str, object] = {}
        if "ins" in params:
            bound["ins"] = ins
        if "rets" in params:
            bound["rets"] = rets
        if "lib_" in params:
            bound["lib_"] = _lib_mapping[device]

        # NOTE (typing/robustness): This required-argument check is intentionally simple.
        # If you see false positives, consider narrowing this to required KEYWORD_ONLY params only
        # (p.kind is KEYWORD_ONLY and default is empty), since we bind by keyword names.
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        if not accepts_var_kw:
            required = [
                name
                for name, p in params.items()
                if name != "cls" and p.default is inspect._empty
            ]
            missing = [name for name in required if name not in bound]
            if missing:
                raise TypeError(
                    f"{cls.__name__}.__grad__ missing required argument(s): {', '.join(missing)}"
                )

        return partial(cls.__grad__, **bound)

    @classmethod
    @overload
    def __grad__(
        cls, *, ins: tuple[Tensor, ...], rets: tuple[Tensor, ...], lib_: ModuleType
    ) -> _GradType: ...

    @classmethod
    @overload
    def __grad__(
        cls, *, ins: tuple[Tensor, ...], rets: tuple[Tensor, ...]
    ) -> _GradType: ...

    @classmethod
    @overload
    def __grad__(cls, *, rets: tuple[Tensor, ...], lib_: ModuleType) -> _GradType: ...

    @classmethod
    @overload
    def __grad__(cls, *, rets: tuple[Tensor, ...]) -> _GradType: ...

    @classmethod
    @abstractmethod
    def __grad__(cls, *args, **kwargs) -> _GradType: ...

    def __new__(cls, *args, **kwargs) -> Never:
        raise TypeError(f"{cls.__name__} cannot be instantiated.")


_FusionTableEntry = tuple[type[Operation], type[Operation]]

# NOTE (registration): `_fusion_table` is built at import time from current subclasses.
# If fused-rule subclasses are imported after this module, they won't appear here.
# Recommended: build lazily inside `match_fusion_table` OR register via `__init_subclass__`.
_fusion_table: dict[_FusionTableEntry, type[FusedBackwardOp]] = {
    (fused_op.op1, fused_op.op2): fused_op
    for fused_op in FusedBackwardOp.__subclasses__()
}


def match_fusion_table(
    op1: type[Operation], op2: type[Operation]
) -> type[FusedBackwardOp] | None:
    return _fusion_table.get((op1, op2), None)
