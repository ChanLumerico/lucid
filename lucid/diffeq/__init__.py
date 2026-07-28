"""lucid.diffeq — differential-equation solvers.

Numerical integration of ordinary differential equations, ``dy/dt = f(t, y)``,
where the right-hand side is an arbitrary Lucid callable — most often a neural
network, as in continuous normalising flows, flow matching, and rectified flow.

Surface: :func:`odeint` integrates to a set of times you name up front and
:func:`odeint_dense` returns a continuous solution you can query afterwards, and
:func:`odeint_adjoint` trades exactness for constant memory when the solve is
long enough that retained stages dominate, and :func:`odeint_event` runs until a
condition fires rather than to a time you name;
:class:`ButcherTableau` is the coefficient table they dispatch on.  Nineteen
built-in tableaux are exported, and a custom tableau is accepted anywhere a
method name is.

The tableau decides how ``t`` is read.  An **adaptive** method — the default
:data:`DOPRI5`, plus :data:`TSIT5`, :data:`BOSH3`, :data:`FEHLBERG2` and
:data:`ADAPTIVE_HEUN` —
carries an embedded error estimate, so it picks its own step sizes to hold the
error inside ``rtol`` / ``atol`` and interpolates to the times in ``t``.  A
**fixed-step** method — :data:`EULER`, :data:`MIDPOINT`, :data:`HEUN2`,
:data:`HEUN3`, :data:`RK4`, :data:`RK4_CLASSIC` — steps once per interval of
``t``, so ``t`` is the integration grid itself.

Two further fixed-step families trade a different way.  **Adams multistep** —
method names ``"explicit_adams"``, ``"implicit_adams"`` and ``"fixed_adams"``,
which are not tableaux — reaches high order by reusing derivatives from
earlier steps, so a step costs one evaluation whatever its order.  **Implicit**
methods — :data:`IMPLICIT_EULER`, :data:`IMPLICIT_MIDPOINT`, :data:`TRAPEZOID`,
:data:`RADAU_IIA3`, :data:`RADAU_IIA5`, :data:`GL4`, :data:`GL6`,
:data:`SDIRK2` and :data:`TRBDF2` — solve a nonlinear system every step, which
is expensive but is what lets them take large steps on a stiff problem, where
an explicit method is held to tiny ones by stability alone.

Everything here is reached as ``lucid.diffeq.*`` and nowhere else — there are
no top-level aliases and no ``Tensor`` methods, so each name has exactly one
canonical path.

Stochastic differential equations are not implemented yet.

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
    Kutta's 3/8 rule — four stages, fourth order.  Method name ``"rk4"``,
    matching what that name means in the reference ODE library.
RK4_CLASSIC : ButcherTableau
    The classical Runge-Kutta method — four stages, fourth order.  Method name
    ``"rk4_classic"``.  A Lucid extension: the reference has no name for this
    tableau, but it is what most other sources call "RK4".
IMPLICIT_EULER : ButcherTableau
    Backward Euler — one stage, first order, implicit.  The most stable
    method here and the least accurate.  Method name ``"implicit_euler"``.
IMPLICIT_MIDPOINT : ButcherTableau
    The implicit midpoint rule — one stage, second order, implicit.  Method
    name ``"implicit_midpoint"``.
TRAPEZOID : ButcherTableau
    The trapezoidal rule — two stages, second order, implicit.  Method name
    ``"trapezoid"``.
RADAU_IIA3 : ButcherTableau
    Radau IIA — two stages, third order, implicit.  Method name
    ``"radauIIA3"``.
RADAU_IIA5 : ButcherTableau
    Radau IIA — three stages, fifth order, implicit.  Method name
    ``"radauIIA5"``.
GL4 : ButcherTableau
    Gauss-Legendre — two stages, fourth order, implicit.  Method name
    ``"gl4"``.
GL6 : ButcherTableau
    Gauss-Legendre — three stages, sixth order, implicit.  The highest order
    per stage any Runge-Kutta method reaches.  Method name ``"gl6"``.
SDIRK2 : ButcherTableau
    Singly diagonally implicit — two stages, second order.  Method name
    ``"sdirk2"``.
TRBDF2 : ButcherTableau
    TR-BDF2 — three stages, second order, implicit.  Method name
    ``"trbdf2"``.
"""

from lucid.diffeq._adjoint import odeint_adjoint
from lucid.diffeq._solvers import odeint, odeint_dense, odeint_event
from lucid.diffeq._tableau import (
    ADAPTIVE_HEUN,
    BOSH3,
    DOPRI5,
    EULER,
    FEHLBERG2,
    GL4,
    GL6,
    HEUN2,
    HEUN3,
    IMPLICIT_EULER,
    IMPLICIT_MIDPOINT,
    MIDPOINT,
    RADAU_IIA3,
    RADAU_IIA5,
    RK4,
    RK4_CLASSIC,
    SDIRK2,
    TRAPEZOID,
    TRBDF2,
    TSIT5,
    ButcherTableau,
)

__all__ = [
    "odeint",
    "odeint_adjoint",
    "odeint_dense",
    "odeint_event",
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
    "RK4_CLASSIC",
    "IMPLICIT_EULER",
    "IMPLICIT_MIDPOINT",
    "TRAPEZOID",
    "RADAU_IIA3",
    "RADAU_IIA5",
    "GL4",
    "GL6",
    "SDIRK2",
    "TRBDF2",
]
