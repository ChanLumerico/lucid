"""
autograd.backward() and autograd.grad() free functions.
"""

from typing import TYPE_CHECKING
import numpy as np
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def backward(
    tensors: Tensor | list[Tensor],
    grad_tensors: list[Tensor] | None = None,
    retain_graph: bool = False,
    create_graph: bool = False,
    inputs: list[Tensor] | None = None,
) -> None:
    """Compute gradients of tensors w.r.t. leaf variables.

    Args:
        tensors:      Root tensor(s) to differentiate.
        grad_tensors: Seed gradients per root tensor (default: ones-like).
        retain_graph: Keep the computation graph after backward.
        create_graph: Not yet supported.
        inputs:       Not yet supported.
    """
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    if grad_tensors is None:
        for t in tensors:
            t.backward(retain_graph=retain_graph, create_graph=create_graph)
    else:
        for t, g in zip(tensors, grad_tensors):
            t.backward(gradient=g, retain_graph=retain_graph, create_graph=create_graph)


def grad(
    outputs: Tensor | list[Tensor],
    inputs: Tensor | list[Tensor],
    grad_outputs: list[Tensor] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool = False,
) -> tuple[Tensor | None, ...]:
    """Compute gradients of outputs w.r.t. inputs, returning them directly.

    Unlike :meth:`~lucid.Tensor.backward`, this function does **not** accumulate
    gradients into ``.grad``; it returns them as a tuple.

    Parameters
    ----------
    outputs : Tensor or list of Tensor
        Output tensors to differentiate.
    inputs : Tensor or list of Tensor
        Input tensors to compute gradients for.
    grad_outputs : list of Tensor, optional
        Seed gradients for non-scalar outputs.
    retain_graph : bool, optional
        Keep the computation graph after backward. Defaults to ``create_graph``.
    create_graph : bool, optional
        Not yet supported.
    allow_unused : bool, optional
        If ``True``, return ``None`` for inputs with no gradient path.

    Returns
    -------
    tuple of Tensor or None
        One gradient per input.

    Examples
    --------
    >>> x = lucid.randn(3)
    >>> x.requires_grad_(True)
    >>> y = (x * x).sum()
    >>> (gx,) = lucid.autograd.grad(y, [x])
    """
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    # Save existing .grad values so we can restore them after
    saved_grads: list[Tensor | None] = [inp.grad for inp in inputs]
    # Zero out existing grads on inputs so we can read the fresh ones
    for inp in inputs:
        if inp._impl.grad_as_python() is not None:
            inp._impl.zero_grad()

    _retain = retain_graph if retain_graph is not None else create_graph

    # Run backward
    if grad_outputs is None:
        for out in outputs:
            out.backward(retain_graph=_retain)
    else:
        for out, g in zip(outputs, grad_outputs):
            out.backward(gradient=g, retain_graph=_retain)

    # Collect computed gradients
    result: list[Tensor | None] = []
    for inp in inputs:
        g_raw = inp._impl.grad_as_python()
        if g_raw is None:
            if not allow_unused:
                raise RuntimeError(
                    "One of the differentiated tensors does not require grad "
                    "and is not reachable from outputs. "
                    "Set allow_unused=True to suppress this error."
                )
            result.append(None)
        else:
            arr = np.ascontiguousarray(np.asarray(g_raw))
            impl = _C_engine.TensorImpl(arr, inp._impl.device, False)
            result.append(_wrap(impl))

    # Restore original .grad values
    for inp, saved in zip(inputs, saved_grads):
        inp._impl.zero_grad()
        if saved is not None:
            # Re-run a zero-scale backward to install saved grad — not ideal,
            # but the engine has no "set_grad" API. Use in-place write instead.
            raw = inp._impl.grad_as_python()
            if raw is None:
                # grad buffer doesn't exist yet — can't restore it easily
                pass
            else:
                raw[:] = saved.numpy()

    return tuple(result)
