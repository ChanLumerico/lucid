"""
nn.Parameter: a Tensor that is automatically registered by Module.
"""

from lucid._tensor.tensor import Tensor
from lucid._C import engine as _C_engine
from lucid._factories.converters import _to_impl


class Parameter(Tensor):
    """
    A Tensor subclass that Module recognizes as a learnable parameter.

    When assigned as a Module attribute, it is automatically added to
    `_parameters` and yielded by `parameters()`.
    """

    _is_parameter: bool = True

    def __new__(
        cls,
        data: "Tensor | list[object] | None" = None,
        requires_grad: bool = True,
    ) -> "Parameter":
        if data is None:
            impl = _C_engine.zeros([0], _C_engine.F32, _C_engine.CPU)
            impl = impl.clone_with_grad(requires_grad)
        elif isinstance(data, Tensor):
            impl = data._impl.clone_with_grad(requires_grad)
        else:
            impl = _to_impl(data, requires_grad=requires_grad)
        obj = object.__new__(cls)
        obj._impl = impl
        return obj

    def __init__(
        self,
        data: "Tensor | list[object] | None" = None,
        requires_grad: bool = True,
    ) -> None:
        pass

    def __repr__(self) -> str:
        return f"Parameter containing:\n{super().__repr__()}"
