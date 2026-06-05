"""Dedicated unit coverage for ``nn`` padding modules.

Forward shape + (where deterministic) value checks, plus a backward pass so
the grad path is exercised — these families had no dedicated test file before.
"""

import lucid
import lucid.nn as nn


class TestConstantZeroPad2d:
    def test_zeropad2d_shape_and_corner(self) -> None:
        # padding is (left, right, top, bottom)
        m = nn.ZeroPad2d((1, 2, 3, 4))
        out = m(lucid.ones(2, 3, 8, 8))
        assert out.shape == (2, 3, 15, 11)  # H+top+bottom, W+left+right
        o = out.numpy()
        assert o[0, 0, 0, 0] == 0.0  # padded corner
        assert o[0, 0, 3, 1] == 1.0  # first real pixel (top=3, left=1)

    def test_zeropad2d_int_padding(self) -> None:
        out = nn.ZeroPad2d(2)(lucid.ones(1, 1, 4, 4))
        assert out.shape == (1, 1, 8, 8)

    def test_constantpad2d_value(self) -> None:
        out = nn.ConstantPad2d((1, 1, 1, 1), 5.0)(lucid.zeros(1, 1, 2, 2)).numpy()
        assert out.shape == (1, 1, 4, 4)
        assert out[0, 0, 0, 0] == 5.0  # padding region == fill value
        assert out[0, 0, 1, 1] == 0.0  # interior unchanged

    def test_backward(self) -> None:
        x = lucid.ones(1, 1, 3, 3, requires_grad=True)
        nn.ZeroPad2d(1)(x).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (1, 1, 3, 3)


class TestReflectionReplicationPad2d:
    def test_reflection_shape(self) -> None:
        out = nn.ReflectionPad2d(2)(lucid.ones(2, 3, 8, 8))
        assert out.shape == (2, 3, 12, 12)

    def test_replication_shape(self) -> None:
        out = nn.ReplicationPad2d(2)(lucid.ones(2, 3, 8, 8))
        assert out.shape == (2, 3, 12, 12)

    def test_replication_edge_value(self) -> None:
        # replication repeats the edge value outward
        x = lucid.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # (1,1,2,2)
        out = nn.ReplicationPad2d(1)(x).numpy()
        assert out.shape == (1, 1, 4, 4)
        assert out[0, 0, 0, 0] == 1.0  # top-left corner replicates [0,0]
        assert out[0, 0, 3, 3] == 4.0  # bottom-right replicates [1,1]

    def test_reflection_backward(self) -> None:
        x = lucid.ones(1, 1, 5, 5, requires_grad=True)
        nn.ReflectionPad2d(2)(x).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (1, 1, 5, 5)
