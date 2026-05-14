"""
autograd.Function: base class for custom differentiable operations.
"""

from typing import Protocol, cast
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._tensor.tensor import Tensor
from lucid.autograd._python_node import _register


class _FunctionClass(Protocol):
    """Protocol describing a Function subclass (has forward/backward classmethods)."""

    @classmethod
    def forward(
        cls, ctx: object, *args: object, **kwargs: object
    ) -> Tensor | tuple[Tensor, ...]: ...

    @classmethod
    def backward(
        cls, ctx: object, *grad_outputs: Tensor
    ) -> Tensor | tuple[Tensor | None, ...] | list[Tensor | None]: ...


class FunctionCtx:
    """Per-call context shared between :meth:`Function.forward` and
    :meth:`Function.backward`.

    A fresh ``FunctionCtx`` is created on every :meth:`Function.apply`
    invocation. ``forward`` populates it with anything ``backward`` will
    need: saved tensors (via :meth:`save_for_backward`), non-differentiable
    output markers (via :meth:`mark_non_differentiable`), and arbitrary
    user-defined attributes (cached shapes, axis indices, scalar
    hyperparameters, ...) set with ordinary attribute assignment.

    Attributes
    ----------
    needs_input_grad : tuple of bool
        One flag per positional Tensor input to ``forward``, indicating
        whether autograd would propagate a gradient to that input. Useful
        for skipping unneeded gradient computations.
    saved_tensors : tuple of Tensor
        Read-only view of tensors stored via :meth:`save_for_backward`.
    """

    def __init__(self) -> None:
        """Initialise an empty context with no saved tensors or extras."""
        self._saved_tensors: list[Tensor] = []
        self.needs_input_grad: tuple[bool, ...] = ()
        self._non_differentiable: list[Tensor] = []
        self._extra: dict[str, object] = {}

    def save_for_backward(self, *tensors: Tensor) -> None:
        """Store tensors needed to compute the backward pass.

        Parameters
        ----------
        *tensors : Tensor
            Activations / inputs the ``backward`` implementation will
            require. They are retrieved later via the :attr:`saved_tensors`
            property as a tuple in the same order.

        Notes
        -----
        Each call replaces any tensors previously saved on this context.
        """
        self._saved_tensors = list(tensors)

    @property
    def saved_tensors(self) -> tuple[Tensor, ...]:
        """Read back the tensors saved during ``forward``.

        Returns
        -------
        tuple of Tensor
            The tensors stored by :meth:`save_for_backward`, wrapped
            back into Python ``Tensor`` instances if the engine stored
            raw ``TensorImpl`` handles.
        """
        result: list[Tensor] = []
        for t in self._saved_tensors:
            if isinstance(t, _C_engine.TensorImpl):
                result.append(_wrap(t))
            else:
                result.append(t)
        return tuple(result)

    def mark_non_differentiable(self, *tensors: Tensor) -> None:
        """Declare that the given output tensors carry no gradient.

        Parameters
        ----------
        *tensors : Tensor
            Outputs of :meth:`Function.forward` for which autograd should
            not propagate gradients (e.g., integer indices, masks).
        """
        self._non_differentiable = list(tensors)

    def __setattr__(self, name: str, value: object) -> None:
        """Route user-defined attributes onto the ``_extra`` dict.

        Reserved private names and ``needs_input_grad`` use normal
        instance storage so they remain accessible through descriptors;
        everything else falls back to the extras bag, keeping the
        context permissive while preserving the public schema.
        """
        if name.startswith("_") or name in ("needs_input_grad",):
            object.__setattr__(self, name, value)
        else:
            try:
                object.__setattr__(self, name, value)
            except AttributeError:
                self._extra[name] = value

    def __getattr__(self, name: str) -> object:
        """Look up overflow attributes that were stored in ``_extra``.

        Called only when normal attribute resolution fails, so it
        complements :meth:`__setattr__` without shadowing class members.
        """
        extra = object.__getattribute__(self, "_extra")
        if name in extra:
            return extra[name]
        raise AttributeError(f"FunctionCtx has no attribute '{name}'")


def _make_apply(cls: type) -> classmethod:  # type: ignore[type-arg]
    """Build the per-subclass ``apply`` classmethod injected by ``FunctionMeta``."""

    def apply(
        klass: type, *args: Tensor, **kwargs: object
    ) -> Tensor | tuple[Tensor, ...]:
        """Run ``forward`` and register the autograd node when needed."""
        ctx = FunctionCtx()
        ctx.needs_input_grad = tuple(
            isinstance(a, Tensor) and a.requires_grad for a in args
        )

        output = klass.forward(ctx, *args, **kwargs)  # type: ignore[attr-defined]

        if _C_engine.grad_enabled() and any(ctx.needs_input_grad):
            tensor_inputs = [a for a in args if isinstance(a, Tensor)]
            if isinstance(output, Tensor):
                _register(output, klass, ctx, tensor_inputs)  # type: ignore[arg-type]

        return cast(Tensor | tuple[Tensor, ...], output)

    return classmethod(apply)  # type: ignore[arg-type]


class FunctionMeta(type):
    """Metaclass that wires a fresh :meth:`apply` onto each ``Function`` subclass.

    The base ``Function`` class is left untouched (so ``Function.apply`` keeps
    its explanatory stub); every concrete subclass receives a closure that
    instantiates a :class:`FunctionCtx`, calls the subclass's ``forward``,
    and — when grad is enabled and any input requests it — registers the
    autograd node.
    """

    def __init__(
        cls, name: str, bases: tuple[type, ...], dct: dict[str, object]
    ) -> None:
        """Install the dispatching :meth:`apply` on the freshly built subclass."""
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
    def forward(ctx: FunctionCtx, *args: Tensor) -> Tensor | tuple[Tensor, ...]:
        """Compute the forward result of the custom op.

        Subclasses override this static method. Save anything needed
        during backward on ``ctx`` (via ``ctx.save_for_backward(...)``
        or by setting attributes like ``ctx.shape = x.shape``); do not
        capture tensors via closure.

        Parameters
        ----------
        ctx : FunctionCtx
            Per-call context; will be passed unchanged to :meth:`backward`.
        *args : Tensor
            Positional inputs to the custom op. Non-Tensor positional
            arguments are allowed but receive no gradients.

        Returns
        -------
        Tensor or tuple of Tensor
            One or more output tensors.
        """
        raise NotImplementedError

    @staticmethod
    def backward(
        ctx: FunctionCtx, *grad_outputs: Tensor
    ) -> Tensor | tuple[Tensor, ...]:
        r"""Compute gradients of the loss w.r.t. each input of :meth:`forward`.

        Subclasses override this static method. The returned tuple must
        contain one gradient per **positional argument** of ``forward``,
        in the same order; non-Tensor arguments receive ``None``.

        Parameters
        ----------
        ctx : FunctionCtx
            The same context that was populated during ``forward``.
        *grad_outputs : Tensor
            Upstream gradients :math:`\partial L / \partial y_i`, one per
            output that ``forward`` returned.

        Returns
        -------
        Tensor or tuple of Tensor
            Downstream gradients matching the positional inputs of
            ``forward``. Use ``None`` for inputs that have no gradient.
        """
        raise NotImplementedError

    @classmethod
    def apply(cls, *args: object, **kwargs: object) -> object:
        """Run the function — autograd-tracked when ``forward`` requests grad.

        ``FunctionMeta`` rewrites this on every concrete subclass to dispatch
        through ``forward``/``backward``; the base implementation here exists
        only so ``Function.apply`` is a real attribute (matters for tooling
        and ``hasattr`` checks).  Calling it on the base class is meaningless
        and raises.
        """
        raise NotImplementedError(
            "Function.apply must be called on a concrete subclass that "
            "implements forward / backward."
        )
