"""
lucid
=====
"""

from lucid.main import _ArrayLike, Tensor


def tensor(data: _ArrayLike, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad)
