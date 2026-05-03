"""
Bridge between Python autograd.Function and the C++ engine's backward graph.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._tensor.tensor import Tensor



def _register(
    output: Tensor,
    fn_class: type,
    py_ctx: object,
    tensor_inputs: list[Tensor],
) -> None:
    """
    Wire a Python backward function into the C++ autograd graph.

    The C++ engine calls backward_fn(cpp_ctx, grad_impl) during backprop.
    We use a closure to capture py_ctx and the user's backward() method.
    """
    # C++ FunctionCtx — used as a carrier; C++ engine needs it to be non-null
    cpp_ctx = _C_engine.FunctionCtx()

    def backward_fn(_cpp_ctx: object, grad_impl: _C_engine.TensorImpl) -> list[_C_engine.TensorImpl]:
        """Called by C++ engine: (cpp_ctx, grad_impl) → list[TensorImpl | None]"""
        grad_tensor = _wrap(grad_impl)
        grads = fn_class.backward(py_ctx, grad_tensor)
        if not isinstance(grads, (list, tuple)):
            grads = [grads]

        result: list[_C_engine.TensorImpl] = []
        for g in grads:
            if g is None:
                result.append(None)
            elif isinstance(g, Tensor):
                result.append(_unwrap(g))
            else:
                result.append(None)
        return result

    node = _C_engine._PythonBackwardNode()
    node.ctx = cpp_ctx
    node.backward_fn = backward_fn

    impl_inputs = [_unwrap(t) for t in tensor_inputs]
    _C_engine._register_python_backward_node(output._impl, node, impl_inputs)
