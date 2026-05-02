"""lucid.autograd — Python-facing custom autograd API.

Mirrors torch.autograd.Function so users can define custom forward/backward:

    class MyOp(lucid.autograd.Function):
        @staticmethod
        def forward(ctx, x, y):
            ctx.save_for_backward(x, y)
            return x * y

        @staticmethod
        def backward(ctx, grad):
            x, y = ctx.saved_tensors
            return grad * y, grad * x

    z = MyOp.apply(x, y)
    z.backward()
"""

from lucid._C import engine as _E


def _apply(cls, *args):
    """Execute cls.forward then wire a PythonBackwardNode into the graph."""
    ctx = _E.FunctionCtx()

    # Collect TensorImpl inputs for edge building.
    tensor_inputs = [a for a in args if isinstance(a, _E.TensorImpl)]

    # Run forward (always — even with grad disabled; user controls this).
    output = cls.forward(ctx, *args)

    if not isinstance(output, _E.TensorImpl):
        return output

    # Only wire backward if grad mode is on and at least one input needs grad.
    needs_grad = _E.grad_enabled() and any(
        t.requires_grad for t in tensor_inputs
    )
    if not needs_grad:
        return output

    node = _E._PythonBackwardNode()
    node.ctx = ctx
    node.backward_fn = staticmethod(cls.backward)
    _E._register_python_backward_node(output, node, tensor_inputs)
    return output


class FunctionMeta(type):
    """Metaclass that injects a classmethod `apply` on every Function subclass."""

    def __new__(mcs, name, bases, ns):
        cls = super().__new__(mcs, name, bases, ns)
        # Only inject on concrete subclasses, not on Function itself.
        if any(isinstance(b, FunctionMeta) for b in bases):
            @classmethod
            def apply(klass, *args):
                return _apply(klass, *args)
            cls.apply = apply
        return cls


class Function(metaclass=FunctionMeta):
    """Base class for custom autograd functions.

    Subclass this and implement ``forward`` and ``backward`` as static methods.
    Call ``MyOp.apply(...)`` to execute (do not call ``forward`` directly).
    """

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError("Subclasses must implement forward()")

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Subclasses must implement backward()")
