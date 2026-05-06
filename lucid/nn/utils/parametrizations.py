"""Parametrize-based reparametrisations.

The reference framework's ``parametrizations`` package re-implements
``weight_norm`` and ``spectral_norm`` on top of the modern
``register_parametrization`` API.  We mirror that surface so user code that
imports ``from lucid.nn.utils.parametrizations import weight_norm`` keeps
working — under the hood the new entry points just delegate to the legacy
implementations in ``weight_norm.py`` / ``spectral_norm.py``, which already
handle the same observable behaviour (a weight derived from trainable
helpers via a forward pre-hook).

A future pass can replace these delegations with proper ``Parametrization``
modules registered via ``register_parametrization``; the public surface
stays stable either way.
"""

from lucid.nn.module import Module

from lucid.nn.utils.spectral_norm import spectral_norm as _legacy_spectral_norm
from lucid.nn.utils.weight_norm import weight_norm as _legacy_weight_norm


def weight_norm(module: Module, name: str = "weight", dim: int = 0) -> Module:
    """Modern ``weight_norm`` reparametrisation entry point.

    Behaviour matches the legacy ``lucid.nn.utils.weight_norm`` — the alias
    here is for source-level parity with the reference framework, which
    moved the canonical entry point under ``parametrizations``.
    """
    return _legacy_weight_norm(module, name=name, dim=dim)


def spectral_norm(
    module: Module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: int | None = None,
) -> Module:
    """Modern ``spectral_norm`` reparametrisation entry point.

    Delegates to the legacy ``lucid.nn.utils.spectral_norm`` — same
    observable behaviour, same buffers (``weight_orig`` / ``weight_u`` /
    ``weight_v``).
    """
    return _legacy_spectral_norm(
        module,
        name=name,
        n_power_iterations=n_power_iterations,
        eps=eps,
        dim=dim,
    )
