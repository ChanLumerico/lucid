"""
Tests for lucid._types — protocols, aliases, and Generic Tensor.
"""

import numpy as np
import pytest
import lucid
from lucid._types import (
    HasShape, SupportsNumpyConversion, SupportsGrad, TensorLikeProtocol,
    ParamGroupDict, DT, DV,
)


class TestProtocols:
    def test_tensor_satisfies_has_shape(self):
        t = lucid.randn(3, 4)
        assert isinstance(t, HasShape)
        assert t.shape == (3, 4)
        assert t.ndim == 2

    def test_tensor_satisfies_numpy_conversion(self):
        t = lucid.randn(2, 3)
        assert isinstance(t, SupportsNumpyConversion)
        arr = t.numpy()
        assert isinstance(arr, np.ndarray)

    def test_tensor_satisfies_supports_grad(self):
        t = lucid.randn(3, requires_grad=True)
        assert isinstance(t, SupportsGrad)

    def test_tensor_satisfies_tensor_like_protocol(self):
        t = lucid.randn(3)
        assert isinstance(t, TensorLikeProtocol)

    def test_numpy_array_not_tensor_like(self):
        arr = np.zeros((3,))
        assert not isinstance(arr, TensorLikeProtocol)

    def test_custom_class_satisfies_has_shape(self):
        class MyShape:
            @property
            def shape(self) -> tuple[int, ...]:
                return (5, 5)
            @property
            def ndim(self) -> int:
                return 2

        obj = MyShape()
        assert isinstance(obj, HasShape)

    def test_custom_class_satisfies_tensor_like(self):
        class MyArray:
            @property
            def shape(self) -> tuple[int, ...]: return (3,)
            @property
            def ndim(self) -> int: return 1
            @property
            def dtype(self): return lucid.float32
            @property
            def device(self): return lucid.device("cpu")
            def numpy(self): return np.zeros(3)
            def to(self, *a, **kw): return self

        obj = MyArray()
        assert isinstance(obj, TensorLikeProtocol)


class TestGenericTensor:
    def test_subscript_runtime(self):
        # Tensor[dtype, device] should return a GenericAlias at runtime
        alias = lucid.Tensor[lucid.float32, lucid.device]
        assert alias is not None

    def test_dtype_property_type(self):
        t = lucid.randn(3)
        assert isinstance(t.dtype, lucid.dtype)

    def test_device_property_type(self):
        t = lucid.randn(3)
        assert isinstance(t.device, lucid.device)


class TestParamGroupDict:
    def test_valid_param_group(self):
        pg: ParamGroupDict = {"params": [], "lr": 1e-3, "weight_decay": 0.0}
        assert pg["lr"] == 1e-3

    def test_total_false_optional_keys(self):
        pg: ParamGroupDict = {"params": []}
        assert "lr" not in pg


class TestPublicAliasesOnLucid:
    def test_scalar_accessible(self):
        assert lucid.Scalar is not None

    def test_tensor_like_accessible(self):
        assert lucid.TensorLike is not None

    def test_device_like_accessible(self):
        assert lucid.DeviceLike is not None

    def test_has_shape_accessible(self):
        assert lucid.HasShape is HasShape

    def test_tensor_like_protocol_accessible(self):
        assert lucid.TensorLikeProtocol is TensorLikeProtocol


class TestTypesBaseAliases:
    def test_device_like_includes_str(self):
        # DeviceLike = _Device | str | None
        # Just verify the alias is subscriptable / usable
        from lucid._types_base import DeviceLike, DTypeLike, ShapeLike, _Size2d

    def test_factory_uses_device_like(self):
        # Factories now use DeviceLike — verify they still accept str
        t = lucid.zeros(3, device="cpu")
        assert t.shape == (3,)

    def test_factory_uses_dtype_like(self):
        t = lucid.ones(3, dtype=lucid.float32)
        assert t.dtype is lucid.float32

    def test_factory_accepts_none_device(self):
        t = lucid.randn(2, 3, device=None)
        assert t is not None
