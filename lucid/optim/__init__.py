"""
lucid.optim — optimizers (mirror of legacy ``lucid.optim``).

Optimizer classes are bound from the C++ engine; this package wraps them
so callers can pass ``lucid.Tensor`` / ``lucid.nn.Parameter`` lists
(unwrapping them to the engine's ``TensorImpl`` slots automatically).
"""

from __future__ import annotations

from typing import Iterable
from lucid._C import engine as _eng
from lucid._tensor import Tensor


def _unwrap_params(params: Iterable):
    out = []
    for p in params:
        if isinstance(p, Tensor):
            out.append(p._impl)
        else:
            # Already a TensorImpl, or a dict (param-group form).
            out.append(p)
    return out


def _make(cls):
    """Return a thin Python wrapper that unwraps Tensor params before
    forwarding to the C++ optimizer constructor."""
    class _Wrapped(cls):
        def __init__(self, params, *args, **kwargs):
            super().__init__(_unwrap_params(params), *args, **kwargs)
    _Wrapped.__name__ = cls.__name__
    _Wrapped.__qualname__ = cls.__qualname__
    return _Wrapped


Optimizer = _eng.Optimizer
SGD       = _make(_eng.SGD)
Adam      = _make(_eng.Adam)
AdamW     = _make(_eng.AdamW)
NAdam     = _make(_eng.NAdam)
RAdam     = _make(_eng.RAdam)
Adamax    = _make(_eng.Adamax)
Adagrad   = _make(_eng.Adagrad)
Adadelta  = _make(_eng.Adadelta)
RMSprop   = _make(_eng.RMSprop)
Rprop     = _make(_eng.Rprop)
ASGD      = _make(_eng.ASGD)

from lucid.optim import lr_scheduler  # noqa: F401

__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "NAdam",
    "RAdam",
    "Adamax",
    "Adagrad",
    "Adadelta",
    "RMSprop",
    "Rprop",
    "ASGD",
    "lr_scheduler",
]
