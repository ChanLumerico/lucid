from abc import ABC, abstractmethod
from typing import ClassVar, Callable, Sequence, overload
from types import ModuleType

from functools import partial
import inspect

from lucid._tensor.tensor import Tensor
from lucid._backend.core import Operation, _GradType


class FusedBackwardOp(ABC):
    op1: ClassVar[type[Operation] | None] = None
    op2: ClassVar[type[Operation] | None] = None

    @classmethod
    def get_fused_grad_func(
        cls,
        inputs: Tensor | Sequence[Tensor],
        results: Tensor | Sequence[Tensor],
        lib_: ModuleType,
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
            bound["lib_"] = lib_

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
