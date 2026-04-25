import numpy as np

import pytest

import lucid

from lucid.test.parity import data

from lucid.test.parity.core import TensorInput

from lucid.test.parity.gradcheck import assert_gradcheck


def _one_grad(shape, *, seed: int, factory=data.random_floats):
    return [TensorInput(factory(shape, seed=seed), requires_grad=True)]


def _two_grad(shape, *, seed: int, factory=data.random_floats):
    return [
        TensorInput(factory(shape, seed=seed), requires_grad=True),
        TensorInput(factory(shape, seed=seed + 1), requires_grad=True),
    ]


GC_RTOL = 0.0001

GC_ATOL = 1e-05


def test_gc_exp():
    assert_gradcheck(lucid.exp, _one_grad((3,), seed=10), rtol=GC_RTOL, atol=GC_ATOL)


def test_gc_log():
    assert_gradcheck(
        lucid.log,
        _one_grad((3,), seed=11, factory=data.pos_floats),
        rtol=GC_RTOL,
        atol=GC_ATOL,
    )


def test_gc_sqrt():
    assert_gradcheck(
        lucid.sqrt,
        _one_grad((3,), seed=12, factory=data.pos_floats),
        rtol=GC_RTOL,
        atol=GC_ATOL,
    )


def test_gc_sin():
    assert_gradcheck(lucid.sin, _one_grad((3,), seed=13), rtol=GC_RTOL, atol=GC_ATOL)


def test_gc_cos():
    assert_gradcheck(lucid.cos, _one_grad((3,), seed=14), rtol=GC_RTOL, atol=GC_ATOL)


def test_gc_tan():
    arr = data.random_floats((3,), seed=15, low=-1.0, high=1.0)
    assert_gradcheck(
        lucid.tan, [TensorInput(arr, requires_grad=True)], rtol=GC_RTOL, atol=GC_ATOL
    )


def test_gc_tanh():
    assert_gradcheck(lucid.tanh, _one_grad((3,), seed=16), rtol=GC_RTOL, atol=GC_ATOL)


def test_gc_sinh():
    assert_gradcheck(lucid.sinh, _one_grad((3,), seed=17), rtol=GC_RTOL, atol=GC_ATOL)


def test_gc_cosh():
    assert_gradcheck(lucid.cosh, _one_grad((3,), seed=18), rtol=GC_RTOL, atol=GC_ATOL)


def test_gc_clip_symmetric():
    arr = data.random_floats((4,), seed=20, low=-2.0, high=2.0)
    assert_gradcheck(
        lambda a: lucid.clip(a, min_value=-0.8, max_value=0.8),
        [TensorInput(arr, requires_grad=True)],
        rtol=GC_RTOL,
        atol=GC_ATOL,
    )


def test_gc_minimum_elementwise():
    assert_gradcheck(
        lucid.minimum, _two_grad((3,), seed=21), rtol=GC_RTOL, atol=GC_ATOL
    )


def test_gc_maximum_elementwise():
    assert_gradcheck(
        lucid.maximum, _two_grad((3,), seed=22), rtol=GC_RTOL, atol=GC_ATOL
    )


def test_gc_reciprocal():
    arr = data.nonzero_floats((3,), seed=23)
    assert_gradcheck(
        lucid.reciprocal,
        [TensorInput(arr, requires_grad=True)],
        rtol=GC_RTOL,
        atol=GC_ATOL,
    )


def test_gc_abs_nonzero():
    arr = data.nonzero_floats((3,), seed=24)
    assert_gradcheck(
        lucid.abs, [TensorInput(arr, requires_grad=True)], rtol=GC_RTOL, atol=GC_ATOL
    )


def test_gc_square():
    assert_gradcheck(lucid.square, _one_grad((3,), seed=25), rtol=GC_RTOL, atol=GC_ATOL)


def test_gc_cube():
    assert_gradcheck(lucid.cube, _one_grad((3,), seed=26), rtol=GC_RTOL, atol=GC_ATOL)


def test_gc_power_integer_exponent():
    assert_gradcheck(
        lambda a: a**3, _one_grad((3,), seed=30), rtol=GC_RTOL, atol=GC_ATOL
    )


def test_gc_power_fractional_exponent_pos_base():
    assert_gradcheck(
        lambda a: a**1.5,
        [TensorInput(data.pos_floats((3,), seed=31), requires_grad=True)],
        rtol=GC_RTOL,
        atol=GC_ATOL,
    )


def test_gc_power_two_input():
    assert_gradcheck(
        lucid.power,
        [
            TensorInput(data.pos_floats((3,), seed=32), requires_grad=True),
            TensorInput(
                data.random_floats((3,), seed=33, low=-1.0, high=1.0),
                requires_grad=True,
            ),
        ],
        rtol=GC_RTOL,
        atol=GC_ATOL,
    )


def test_gc_add():
    assert_gradcheck(
        lambda a, b: a + b, _two_grad((3,), seed=40), rtol=GC_RTOL, atol=GC_ATOL
    )


def test_gc_sub():
    assert_gradcheck(
        lambda a, b: a - b, _two_grad((3,), seed=41), rtol=GC_RTOL, atol=GC_ATOL
    )


def test_gc_mul():
    assert_gradcheck(
        lambda a, b: a * b, _two_grad((3,), seed=42), rtol=GC_RTOL, atol=GC_ATOL
    )


def test_gc_div():
    inputs = [
        TensorInput(data.random_floats((3,), seed=43), requires_grad=True),
        TensorInput(data.pos_floats((3,), seed=44), requires_grad=True),
    ]
    assert_gradcheck(lambda a, b: a / b, inputs, rtol=GC_RTOL, atol=GC_ATOL)


def test_gc_dot_1d():
    assert_gradcheck(lucid.dot, _two_grad((5,), seed=50), rtol=GC_RTOL, atol=GC_ATOL)


def test_gc_matmul_2d_2d():
    inputs = [
        TensorInput(data.random_floats((3, 5), seed=51), requires_grad=True),
        TensorInput(data.random_floats((5, 4), seed=52), requires_grad=True),
    ]
    assert_gradcheck(lambda a, b: a @ b, inputs, rtol=GC_RTOL, atol=GC_ATOL)


def test_gc_inner_1d():
    assert_gradcheck(lucid.inner, _two_grad((5,), seed=53), rtol=GC_RTOL, atol=GC_ATOL)


def test_gc_outer():
    inputs = [
        TensorInput(data.random_floats((4,), seed=54), requires_grad=True),
        TensorInput(data.random_floats((3,), seed=55), requires_grad=True),
    ]
    assert_gradcheck(lucid.outer, inputs, rtol=GC_RTOL, atol=GC_ATOL)


def test_gc_tensordot_axes2():
    inputs = [
        TensorInput(data.random_floats((3, 4, 5), seed=56), requires_grad=True),
        TensorInput(data.random_floats((4, 5, 2), seed=57), requires_grad=True),
    ]
    assert_gradcheck(
        lambda a, b: lucid.tensordot(a, b, axes=2), inputs, rtol=GC_RTOL, atol=GC_ATOL
    )


def test_gc_sum_axis():
    assert_gradcheck(
        lambda a: a.sum(axis=1),
        [TensorInput(data.random_floats((3, 4), seed=60), requires_grad=True)],
        rtol=GC_RTOL,
        atol=GC_ATOL,
    )


def test_gc_mean_axis_keepdims():
    assert_gradcheck(
        lambda a: a.mean(axis=0, keepdims=True),
        [TensorInput(data.random_floats((3, 4), seed=61), requires_grad=True)],
        rtol=GC_RTOL,
        atol=GC_ATOL,
    )


def test_gc_var_axis():
    assert_gradcheck(
        lambda a: a.var(axis=1),
        [TensorInput(data.random_floats((3, 4), seed=62), requires_grad=True)],
        rtol=GC_RTOL,
        atol=GC_ATOL,
    )


def test_gc_trace_2d():
    assert_gradcheck(
        lucid.trace,
        [TensorInput(data.random_floats((4, 4), seed=63), requires_grad=True)],
        rtol=GC_RTOL,
        atol=GC_ATOL,
    )


def test_gc_cumsum_axis():
    assert_gradcheck(
        lambda a: lucid.cumsum(a, axis=1),
        [TensorInput(data.random_floats((3, 4), seed=64), requires_grad=True)],
        rtol=GC_RTOL,
        atol=GC_ATOL,
    )


def test_gc_cumprod_axis():
    assert_gradcheck(
        lambda a: lucid.cumprod(a, axis=1),
        [TensorInput(data.pos_floats((3, 4), seed=65), requires_grad=True)],
        rtol=GC_RTOL,
        atol=GC_ATOL,
    )


def test_gc_norm_l2_axis():
    assert_gradcheck(
        lambda a: lucid.linalg.norm(a, ord=2, axis=1),
        [TensorInput(data.random_floats((3, 4), seed=66), requires_grad=True)],
        rtol=GC_RTOL,
        atol=GC_ATOL,
    )
