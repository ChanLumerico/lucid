"""Parametrize-based reparametrisations.

The reference framework's ``parametrizations`` package re-implements
``weight_norm`` and ``spectral_norm`` on top of the modern
``register_parametrization`` API.  We mirror that surface so user code that
imports ``from lucid.nn.utils.parametrizations import weight_norm`` keeps
working — under the hood these point directly to the canonical implementations.
"""

from lucid.nn.utils.spectral_norm import spectral_norm as spectral_norm
from lucid.nn.utils.weight_norm import weight_norm as weight_norm
