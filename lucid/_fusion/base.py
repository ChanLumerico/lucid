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

        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if not accepts_var_kw:
            bindable = {"ins", "rets", "lib_"}
            required_kwonly = [
                name
                for name, p in params.items()
                if name in bindable
                and p.kind is inspect.Parameter.KEYWORD_ONLY
                and p.default is inspect._empty
            ]
            missing = [name for name in required_kwonly if name not in bound]
            if missing:
                raise TypeError(
                    f"{cls.__name__}.__grad__ missing required keyword-only argument(s): {', '.join(missing)}"
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

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.op1 is not None and cls.op2 is not None:
            key = (cls.op1, cls.op2)
            if key in _fusion_table:
                existing = _fusion_table[key]
                raise ValueError(
                    f"FusedBackwardOp for {cls.op1.__name__} + {cls.op2.__name__} "
                    f"already registered as {existing.__name__}."
                )
            _fusion_table[key] = cls


_FusionTableEntry = tuple[type[Operation], type[Operation]]

_fusion_table: dict[_FusionTableEntry, type[FusedBackwardOp]] = {}


def match_fusion_table(op1: Operation, op2: Operation) -> type[FusedBackwardOp] | None:
    return _fusion_table.get((type(op1), type(op2)), None)
