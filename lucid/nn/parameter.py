"""
nn.Parameter: a Tensor that is automatically registered by Module.
"""

from typing import Any
from lucid._tensor.tensor import Tensor
from lucid._C import engine as _C_engine


class Parameter(Tensor):
    """
    A Tensor subclass that Module recognizes as a learnable parameter.

    When assigned as a Module attribute, it is automatically added to
    `_parameters` and yielded by `parameters()`.
    """

    _is_parameter: bool = True

    def __new__(
        cls,
        data: Any = None,
        requires_grad: bool = True,
    ) -> "Parameter":
        if data is None:
            import numpy as np
            arr = np.array([], dtype="float32").reshape(0)
            impl = _C_engine.TensorImpl(arr, _C_engine.Device.CPU, requires_grad)
        elif isinstance(data, Tensor):
            # Create new TensorImpl with the desired requires_grad
            impl = data._impl
            if impl.requires_grad != requires_grad:
                import numpy as np
                arr = np.ascontiguousarray(np.asarray(impl.data_as_python()))
                impl = _C_engine.TensorImpl(arr, impl.device, requires_grad)
        else:
            from lucid._factories.converters import _to_impl
            impl = _to_impl(data, requires_grad=requires_grad)
        obj = object.__new__(cls)
        obj._impl = impl
        return obj

    def __init__(self, data: Any = None, requires_grad: bool = True) -> None:
        # __new__ already set self._impl; skip Tensor.__init__ to avoid overwrite
        pass

    def __repr__(self) -> str:
        return f"Parameter containing:\n{super().__repr__()}"
