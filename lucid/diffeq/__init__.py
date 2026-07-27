"""lucid.diffeq — differential-equation solvers.

Numerical integration of ordinary differential equations, ``dy/dt = f(t, y)``,
where the right-hand side is an arbitrary Lucid callable — most often a neural
network, as in continuous normalising flows, flow matching, and rectified flow.

Surface: :func:`odeint` integrates to a set of times you name up front and
:func:`odeint_dense` returns a continuous solution you can query afterwards, and
:func:`odeint_adjoint` trades exactness for constant memory when the solve is
long enough that retained stages dominate;
:class:`ButcherTableau` is the coefficient table they dispatch on.  Ten
built-in tableaux are exported, and a custom tableau is accepted anywhere a
method name is.

The tableau decides how ``t`` is read.  An **adaptive** method — the default
:data:`DOPRI5`, plus :data:`TSIT5`, :data:`BOSH3`, :data:`FEHLBERG2` and
:data:`ADAPTIVE_HEUN` —
carries an embedded error estimate, so it picks its own step sizes to hold the
error inside ``rtol`` / ``atol`` and interpolates to the times in ``t``.  A
**fixed-step** method — :data:`EULER`, :data:`MIDPOINT`, :data:`HEUN2`,
:data:`HEUN3`, :data:`RK4` — steps once per interval of ``t``, so ``t`` is the
integration grid itself.

Everything here is reached as ``lucid.diffeq.*`` and nowhere else — there are
no top-level aliases and no ``Tensor`` methods, so each name has exactly one
canonical path.

The O(1)-memory adjoint, event handling, implicit methods, and stochastic
differential equations are not implemented yet.

Attributes
----------
DOPRI5 : ButcherTableau
    Dormand-Prince 5(4) — seven stages, adaptive, fifth order.  The default
    method.  Method name ``"dopri5"``.
TSIT5 : ButcherTableau
    Tsitouras 5(4) — seven stages, adaptive, fifth order.  Method name
    ``"tsit5"``.
BOSH3 : ButcherTableau
    Bogacki-Shampine 3(2) — four stages, adaptive, third order.  Method name
    ``"bosh3"``.
FEHLBERG2 : ButcherTableau
    Fehlberg 2(1) — three stages, adaptive, second order.  Method name
    ``"fehlberg2"``.
ADAPTIVE_HEUN : ButcherTableau
    Heun-Euler 2(1) — two stages, adaptive, second order.  Method name
    ``"adaptive_heun"``.
EULER : ButcherTableau
    Forward Euler — one stage, first order.  Method name ``"euler"``.
MIDPOINT : ButcherTableau
    Explicit midpoint — two stages, second order.  Method name ``"midpoint"``.
HEUN2 : ButcherTableau
    Heun's method (explicit trapezoid) — two stages, second order.  Method
    name ``"heun2"``.
HEUN3 : ButcherTableau
    Heun's third-order method — three stages, third order.  Method name
    ``"heun3"``.
RK4 : ButcherTableau
    The classical Runge-Kutta method — four stages, fourth order.  Method
    name ``"rk4"``.
"""

from lucid.diffeq._adjoint import odeint_adjoint
from lucid.diffeq._solvers import odeint, odeint_dense
from lucid.diffeq._tableau import (
    ADAPTIVE_HEUN,
    BOSH3,
    DOPRI5,
    EULER,
    FEHLBERG2,
    HEUN2,
    HEUN3,
    MIDPOINT,
    RK4,
    TSIT5,
    ButcherTableau,
)

__all__ = [
    "odeint",
    "odeint_adjoint",
    "odeint_dense",
    "ButcherTableau",
    "DOPRI5",
    "TSIT5",
    "BOSH3",
    "FEHLBERG2",
    "ADAPTIVE_HEUN",
    "EULER",
    "MIDPOINT",
    "HEUN2",
    "HEUN3",
    "RK4",
]
