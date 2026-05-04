import numpy as np

from typing import TYPE_CHECKING, Final
from lucid._dtype import float32 as _f32

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_REPR_THRESHOLD: Final = 1000
_REPR_EDGEITEMS: Final = 3
_REPR_PRECISION: Final = 4


def tensor_repr(t: Tensor) -> str:
    """
    Produce a human-readable tensor representation.

    Examples:
        tensor([1., 2., 3.])
        tensor([[1., 2.], [3., 4.]], device='metal', dtype=lucid.float64,
                requires_grad=True)
    """
    try:
        data = t.numpy()
        arr_str = np.array2string(
            data,
            precision=_REPR_PRECISION,
            separator=", ",
            threshold=_REPR_THRESHOLD,
            edgeitems=_REPR_EDGEITEMS,
        )
    except Exception:
        arr_str = "<data unavailable>"

    extras: list[str] = []
    if t.is_metal:
        extras.append("device='metal'")
    if t.dtype is not _f32:
        extras.append(f"dtype={t.dtype!r}")
    if t.requires_grad:
        extras.append("requires_grad=True")
    suffix = (", " + ", ".join(extras)) if extras else ""
    return f"tensor({arr_str}{suffix})"
