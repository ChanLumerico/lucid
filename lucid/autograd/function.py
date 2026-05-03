"""
autograd.Function: base class for custom differentiable operations.
"""

from typing import Any, TYPE_CHECKING
from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class FunctionCtx:
    """Context object passed to forward/backward in custom Functions."""

    def __init__(self) -> None:
        self._saved_tensors: list[Any] = []
        self.needs_input_grad: tuple[bool, ...] = ()
        self._non_differentiable: list[Any] = []
        self._extra: dict[str, Any] = {}

    def save_for_backward(self, *tensors: Any) -> None:
        """Save tensors to be retrieved in backward()."""
        self._saved_tensors = list(tensors)

    @property
    def saved_tensors(self) -> tuple[Any, ...]:
        """Return the tensors saved by save_for_backward() as Tensors."""
        from lucid._tensor.tensor import Tensor as _T
        from lucid._C import engine as _C_engine
        from lucid._dispatch import _wrap
        result: list[Any] = []
        for t in self._saved_tensors:
            if isinstance(t, _C_engine.TensorImpl):
                result.append(_wrap(t))
            else:
                result.append(t)
        return tuple(result)

    def mark_non_differentiable(self, *tensors: Any) -> None:
        """Mark outputs as non-differentiable."""
        self._non_differentiable = list(tensors)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in ("needs_input_grad",):
            object.__setattr__(self, name, value)
        else:
            try:
                object.__setattr__(self, name, value)
            except AttributeError:
                self._extra[name] = value

    def __getattr__(self, name: str) -> Any:
        extra = object.__getattribute__(self, "_extra")
        if name in extra:
            return extra[name]
        raise AttributeError(f"FunctionCtx has no attribute '{name}'")


def _make_apply(cls: type) -> classmethod:
    def apply(klass: type, *args: Any, **kwargs: Any) -> Any:
        from lucid._tensor.tensor import Tensor as _T
        ctx = FunctionCtx()
        ctx.needs_input_grad = tuple(
            isinstance(a, _T) and a.requires_grad for a in args
        )

        output = klass.forward(ctx, *args, **kwargs)

        if _C_engine.grad_enabled() and any(ctx.needs_input_grad):
            from lucid.autograd._python_node import _register
            tensor_inputs = [a for a in args if isinstance(a, _T)]
            if isinstance(output, _T):
                _register(output, klass, ctx, tensor_inputs)

        return output

    return classmethod(apply)  # type: ignore[return-value]


class FunctionMeta(type):
    def __init__(cls, name: str, bases: tuple[type, ...], dct: dict[str, Any]) -> None:
        super().__init__(name, bases, dct)
        if name != "Function":
            cls.apply = _make_apply(cls)


class Function(metaclass=FunctionMeta):
    """
    Base class for custom differentiable functions.

    Subclass this and implement forward() and backward() as static methods.

    Example:
        class MyReLU(lucid.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return lucid.relu(x)

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                return grad_output * (x > 0)
    """

    @staticmethod
    def forward(ctx: FunctionCtx, *args: Any) -> Any:
        """Compute the forward pass. Override in subclasses."""
        raise NotImplementedError

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs: Any) -> Any:
        """Compute the backward pass. Override in subclasses."""
        raise NotImplementedError
