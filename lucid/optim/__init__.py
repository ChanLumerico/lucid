"""
lucid.optim — optimizers (mirror of legacy ``lucid.optim``).

Optimizer classes are bound from the C++ engine; this package wraps them
so callers can pass ``lucid.Tensor`` / ``lucid.nn.Parameter`` lists
(unwrapping them to the engine's ``TensorImpl`` slots automatically).
"""

from __future__ import annotations

from typing import Iterable
from lucid._C import engine as _C_engine
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


Optimizer = _C_engine.Optimizer
SGD       = _make(_C_engine.SGD)
Adam      = _make(_C_engine.Adam)
AdamW     = _make(_C_engine.AdamW)
NAdam     = _make(_C_engine.NAdam)
RAdam     = _make(_C_engine.RAdam)
Adamax    = _make(_C_engine.Adamax)
Adagrad   = _make(_C_engine.Adagrad)
Adadelta  = _make(_C_engine.Adadelta)
RMSprop   = _make(_C_engine.RMSprop)
Rprop     = _make(_C_engine.Rprop)
ASGD      = _make(_C_engine.ASGD)

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
