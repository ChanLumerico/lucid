"""The dtype a composite special function computes in.

A special function composed of engine ops computes in whatever dtype its
operands arrive in, and two kinds of operand are not fit for that:

* a discrete one (:func:`_real`) — the engine's own ``real_valued``
  schemas promote an integer before the kernel sees it, but a composite
  has no schema, so the promotion has to be asked for;
* a 16-bit float (:func:`_in_float32`) — series, continued fractions and
  polynomial fits carry constants and intermediates outside float16's
  range (``2**24`` as a rescaling threshold, ``Γ`` and ``exp`` of the
  argument) and coefficients past bfloat16's 8 bits.  ``zeta`` and
  ``gammainc`` answered NaN in float16, ``bessel_y0`` NaN on ``(0, 8]``,
  ``bessel_j0(300)`` in bfloat16 was 30 times too small.  The reference
  framework evaluates these kernels in float32 ("opmath") and rounds the
  answer to the 16-bit dtype; so does this.
"""

from collections.abc import Callable

import lucid
from lucid._tensor.tensor import Tensor


def _real(x: Tensor) -> Tensor:
    """A float view of a discrete operand.

    It matters where the composition negates before it reaches a
    promoting op.  ``entr`` computes ``-x * log(x)``, and with a bool
    operand the negation runs first — MLX refuses to negate a bool and
    says to use logical_not, so the whole function failed on Metal while
    the CPU answered.  Promoting at the door removes the question.
    """
    if lucid.is_floating_point(x):
        return x
    return x.to(lucid.float32)


def _in_float32(fn: Callable[..., Tensor], *operands: Tensor) -> Tensor:
    """``fn(*operands)``, evaluated in float32 when the answer is 16-bit.

    The answer's dtype is the operands' common one; when that is float16
    or bfloat16 every operand is widened to float32 first and the result
    rounded back, so only the answer carries 16-bit precision.  Any other
    dtype is passed through untouched.  The casts are differentiable, so
    gradients flow as they would without them.
    """
    common = operands[0].dtype
    if len(operands) == 2:
        common = lucid.result_type(operands[0], operands[1])
    if common not in (lucid.float16, lucid.bfloat16):
        return fn(*operands)
    return fn(*(t.to(lucid.float32) for t in operands)).to(common)
