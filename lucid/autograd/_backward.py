"""
autograd.backward() free function.
"""

from typing import TYPE_CHECKING
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def backward(
    tensors: "Tensor | list[Tensor]",
    grad_tensors: "list[Tensor] | None" = None,
    retain_graph: bool = False,
    create_graph: bool = False,
    inputs: "list[Tensor] | None" = None,
) -> None:
    """
    Compute gradients of tensors w.r.t. leaf variables.

    Args:
        tensors: Root tensor(s) to differentiate.
        grad_tensors: Seed gradients for each root tensor (default: ones).
        retain_graph: If True, keep the computation graph after backward.
        create_graph: Not yet supported.
        inputs: Not yet supported.
    """
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    for t in tensors:
        _C_engine.engine_backward(_unwrap(t), retain_graph=retain_graph)
