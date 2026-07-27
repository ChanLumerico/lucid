"""lucid.diffeq — differential-equation solvers.

Numerical integration of ordinary differential equations, ``dy/dt = f(t, y)``,
where the right-hand side is an arbitrary Lucid callable — most often a neural
network, as in continuous normalising flows, flow matching, and rectified flow.

Surface: :func:`odeint` integrates on a fixed user-supplied grid using an
explicit Runge-Kutta method, and :class:`ButcherTableau` is the coefficient
table it dispatches on.  The five built-in tableaux — :data:`EULER`,
:data:`MIDPOINT`, :data:`HEUN2`, :data:`HEUN3`, :data:`RK4` — are also
exported, and a custom tableau is accepted anywhere a method name is.

Everything here is reached as ``lucid.diffeq.*`` and nowhere else — there are
no top-level aliases and no ``Tensor`` methods, so each name has exactly one
canonical path.

Adaptive step control, the O(1)-memory adjoint, and stochastic differential
equations are not implemented; the fixed-step explicit family is what the flow
model families need, and the rest have materially different performance
characteristics.

Attributes
----------
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

from lucid.diffeq._solvers import odeint
from lucid.diffeq._tableau import ButcherTableau, EULER, MIDPOINT, HEUN2, HEUN3, RK4

__all__ = ["odeint", "ButcherTableau", "EULER", "MIDPOINT", "HEUN2", "HEUN3", "RK4"]
