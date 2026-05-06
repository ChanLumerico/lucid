"""Flatten parameters into a single 1-D vector and back.

Mirrors ``reference framework.nn.utils.{parameters_to_vector, vector_to_parameters}``.
Useful when an algorithm needs the full parameter set as a flat vector — most
commonly L-BFGS / second-order methods, hyper-gradient solvers, and reinforcement
learning trust-region updates.
"""

from typing import Iterable

import lucid
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter


def parameters_to_vector(parameters: Iterable[Parameter]) -> Tensor:
    """Concatenate every parameter's data into a single 1-D ``Tensor``.

    The returned tensor lives on the same device/dtype as the first parameter
    in ``parameters``; subsequent parameters are checked to match. Result has
    ``requires_grad=False`` — it's a snapshot, not a graph node — so ``vector_to_parameters``
    can be called on it without disturbing autograd.
    """
    flats: list[Tensor] = []
    target_device: str | None = None
    target_dtype: object | None = None
    for p in parameters:
        if not isinstance(p, Tensor):
            raise TypeError(f"expected Parameter or Tensor, got {type(p).__name__}")
        if target_device is None:
            target_device = str(p._impl.device)
            target_dtype = p.dtype
        elif str(p._impl.device) != target_device:
            raise ValueError(
                f"all parameters must live on the same device; saw "
                f"{target_device} then {p._impl.device}"
            )
        elif p.dtype is not target_dtype:
            raise ValueError(
                f"all parameters must share dtype; saw {target_dtype} then {p.dtype}"
            )
        flats.append(p.detach().reshape(-1))
    if not flats:
        return lucid.zeros(0)
    return lucid.cat(flats)


def vector_to_parameters(vec: Tensor, parameters: Iterable[Parameter]) -> None:
    """Slice ``vec`` back into the supplied parameters in place.

    ``vec`` must be 1-D and exactly the length implied by ``parameters``;
    mismatches raise ``ValueError``. Each parameter's storage is overwritten
    with its slice — autograd graph state is left intact (no version bump
    side-effects beyond the assignment itself).
    """
    if len(vec.shape) != 1:
        raise ValueError(f"vec must be 1-D, got shape {tuple(vec.shape)}")
    params: list[Parameter] = list(parameters)
    expected_total: int = sum(int(p._impl.numel()) for p in params)
    if int(vec._impl.numel()) != expected_total:
        raise ValueError(
            f"vec length {int(vec._impl.numel())} does not match total parameter "
            f"size {expected_total}"
        )
    offset: int = 0
    for p in params:
        n: int = int(p._impl.numel())
        slice_view: Tensor = vec[offset : offset + n].reshape(p.shape)
        p._impl.copy_from(slice_view._impl)
        offset += n
