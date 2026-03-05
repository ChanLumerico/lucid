from typing import Any, Callable, ClassVar

import numpy as np
import torch

import lucid.nn as nn

from lucid.test.core import ModuleTorchBase, ModuleTorchCase


def _unwrap_callable(operation: Any) -> Callable[..., Any]:
    if isinstance(operation, staticmethod):
        return operation.__func__

    if callable(operation):
        return operation

    raise TypeError(f"Expected callable operation, got: {type(operation)!r}")


class _NNModuleTorchBase(ModuleTorchBase):
    _collect_nested_test_classes = True

    case_name: ClassVar[str]
    inputs: ClassVar[tuple[Any, ...]]

    module_ctor: ClassVar[Callable[..., Any]]
    torch_module_ctor: ClassVar[Callable[..., Any] | None]
    expected_forward: ClassVar[Any | None] = None

    module_kwargs: ClassVar[dict[str, Any]] = {}
    torch_module_kwargs: ClassVar[dict[str, Any]] = {}

    requires_grad: ClassVar[bool | tuple[bool, ...]] = True
    expected_input_grads: ClassVar[tuple[object | None, ...] | None] = None
    module_train_modes: ClassVar[tuple[bool, ...]] = (True,)

    compare_grad_with_torch: ClassVar[bool] = True
    compare_state_dict: ClassVar[bool] = True

    check_finite: ClassVar[bool] = True
    rtol: ClassVar[float] = 1e-7
    atol: ClassVar[float] = 1e-8

    def tensor_op_devices(self) -> tuple[str, ...]:
        return ("cpu", "gpu")

    def _normalize_bool_flags(
        self, flags: bool | tuple[bool, ...], expected_len: int
    ) -> tuple[bool, ...]:
        if isinstance(flags, bool):
            return (bool(flags),) * expected_len
        if len(flags) == 1 and expected_len > 1:
            return (bool(flags[0]),) * expected_len
        if len(flags) != expected_len:
            raise AssertionError(
                f"requires_grad length {len(flags)} does not match inputs "
                f"(expected {expected_len}) for case '{self.case_name}'."
            )
        return tuple(bool(v) for v in flags)

    def _build_inputs(self) -> tuple[Any, ...]:
        requires_grad = self._normalize_bool_flags(self.requires_grad, len(self.inputs))
        outputs: list[Any] = []

        for item, req_grad in zip(self.inputs, requires_grad):
            outputs.append(self.tensor(item, requires_grad=req_grad))

        return tuple(outputs)

    def _build_case(self) -> ModuleTorchCase:
        module_ctor = _unwrap_callable(self.module_ctor)
        torch_module_ctor = _unwrap_callable(self.torch_module_ctor)

        requires_grad = self._normalize_bool_flags(self.requires_grad, len(self.inputs))

        return ModuleTorchCase(
            name=self.case_name,
            build_inputs=lambda _: self._build_inputs(),
            module_ctor=module_ctor,
            torch_module_ctor=torch_module_ctor,
            module_kwargs=self.module_kwargs,
            torch_module_kwargs=self.torch_module_kwargs,
            expected_forward=self.expected_forward,
            expected_input_grads=self.expected_input_grads,
            requires_grad=requires_grad,
            module_train_modes=self.module_train_modes,
            compare_grad_with_torch=self.compare_grad_with_torch,
            compare_state_dict=self.compare_state_dict,
            check_finite=self.check_finite,
            rtol=self.rtol,
            atol=self.atol,
        )

    def module_op_cases(self) -> tuple[ModuleTorchCase, ...]:
        return (self._build_case(),)


class TestIdentity(_NNModuleTorchBase):
    __test__ = False

    class TestDefault(_NNModuleTorchBase):
        case_name = "identity"
        inputs = (np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float64),)
        module_ctor = nn.Identity
        torch_module_ctor = torch.nn.Identity
        expected_forward = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float64)
        compare_state_dict = False


class TestFlatten(_NNModuleTorchBase):
    __test__ = False

    class TestDefault(_NNModuleTorchBase):
        case_name = "flatten_default"
        inputs = (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)
        module_ctor = nn.Flatten
        torch_module_ctor = torch.nn.Flatten
        module_kwargs = {}
        torch_module_kwargs = {}
        expected_forward = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    class TestStartAxis(_NNModuleTorchBase):
        case_name = "flatten_start_axis_end_axis"
        inputs = (np.arange(12.0, dtype=np.float64).reshape(2, 2, 3),)
        module_ctor = nn.Flatten
        torch_module_ctor = torch.nn.Flatten
        module_kwargs = {"start_axis": 0, "end_axis": -1}
        torch_module_kwargs = {"start_dim": 0, "end_dim": -1}
        expected_forward = np.arange(12.0, dtype=np.float64).reshape(2, 2 * 3)


class TestActivation(_NNModuleTorchBase):
    __test__ = False

    class TestReLU(_NNModuleTorchBase):
        case_name = "relu"
        inputs = (np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64),)
        module_ctor = nn.ReLU
        torch_module_ctor = torch.nn.ReLU

    class TestLeakyReLU(_NNModuleTorchBase):
        case_name = "leaky_relu"
        inputs = (np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64),)
        module_ctor = nn.LeakyReLU
        torch_module_ctor = torch.nn.LeakyReLU
        module_kwargs = {"negative_slope": 0.2}
        torch_module_kwargs = {"negative_slope": 0.2}

    class TestTanh(_NNModuleTorchBase):
        case_name = "tanh"
        inputs = (np.array([[-1.0, 0.5], [3.0, -4.0]], dtype=np.float64),)
        module_ctor = nn.Tanh
        torch_module_ctor = torch.nn.Tanh


class TestLinear(_NNModuleTorchBase):
    __test__ = False

    class TestDefault(_NNModuleTorchBase):
        case_name = "linear"
        inputs = (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)
        module_ctor = nn.Linear
        torch_module_ctor = torch.nn.Linear
        module_kwargs = {"in_features": 2, "out_features": 3, "bias": True}
        torch_module_kwargs = {"in_features": 2, "out_features": 3, "bias": True}

    class TestNoBias(_NNModuleTorchBase):
        case_name = "linear_no_bias"
        inputs = (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)
        module_ctor = nn.Linear
        torch_module_ctor = torch.nn.Linear
        module_kwargs = {"in_features": 2, "out_features": 3, "bias": False}
        torch_module_kwargs = {"in_features": 2, "out_features": 3, "bias": False}


class TestConv2d(_NNModuleTorchBase):
    case_name = "conv2d"
    inputs = (np.arange(1.0 * 1 * 4 * 4, dtype=np.float64).reshape(1, 1, 4, 4),)
    module_ctor = nn.Conv2d
    torch_module_ctor = torch.nn.Conv2d
    module_kwargs = {
        "in_channels": 1,
        "out_channels": 2,
        "kernel_size": 3,
        "padding": 1,
    }
    torch_module_kwargs = {
        "in_channels": 1,
        "out_channels": 2,
        "kernel_size": 3,
        "padding": 1,
    }
    rtol = 1e-5
    atol = 1e-6


class TestBatchNorm(_NNModuleTorchBase):
    __test__ = False

    class TestBatchNorm1dTrain(_NNModuleTorchBase):
        case_name = "batch_norm1d_train"
        inputs = (np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64),)
        module_ctor = nn.BatchNorm1d
        torch_module_ctor = torch.nn.BatchNorm1d
        module_kwargs = {"num_features": 1, "momentum": 0.1}
        torch_module_kwargs = {"num_features": 1, "momentum": 0.1}
        module_train_modes = (True,)

    class TestBatchNorm1dEval(_NNModuleTorchBase):
        case_name = "batch_norm1d_eval"
        inputs = (np.array([[[1.0, 2.0, 3.0]]], dtype=np.float64),)
        module_ctor = nn.BatchNorm1d
        torch_module_ctor = torch.nn.BatchNorm1d
        module_kwargs = {"num_features": 1, "momentum": 0.1}
        torch_module_kwargs = {"num_features": 1, "momentum": 0.1}
        module_train_modes = (False,)

    class TestBatchNorm2dTrain(_NNModuleTorchBase):
        case_name = "batch_norm2d_train"
        inputs = (
            np.array(
                [
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ]
                ],
                dtype=np.float64,
            ),
        )
        module_ctor = nn.BatchNorm2d
        torch_module_ctor = torch.nn.BatchNorm2d
        module_kwargs = {"num_features": 2, "momentum": 0.1}
        torch_module_kwargs = {"num_features": 2, "momentum": 0.1}
        module_train_modes = (True,)

    class TestBatchNorm2dEval(_NNModuleTorchBase):
        case_name = "batch_norm2d_eval"
        inputs = (
            np.array(
                [
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ]
                ],
                dtype=np.float64,
            ),
        )
        module_ctor = nn.BatchNorm2d
        torch_module_ctor = torch.nn.BatchNorm2d
        module_kwargs = {"num_features": 2, "momentum": 0.1}
        torch_module_kwargs = {"num_features": 2, "momentum": 0.1}
        module_train_modes = (False,)


class TestLayerNorm(_NNModuleTorchBase):
    case_name = "layer_norm"
    inputs = (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),)
    module_ctor = nn.LayerNorm
    torch_module_ctor = torch.nn.LayerNorm
    module_kwargs = {"normalized_shape": (3,)}
    torch_module_kwargs = {"normalized_shape": (3,)}
    rtol = 1e-5
    atol = 1e-6


class TestGroupNorm(_NNModuleTorchBase):
    case_name = "group_norm"
    inputs = (np.arange(1.0 * 2 * 2 * 2, dtype=np.float64).reshape(1, 2, 2, 2),)
    module_ctor = nn.GroupNorm
    torch_module_ctor = torch.nn.GroupNorm
    module_kwargs = {"num_groups": 1, "num_channels": 2}
    torch_module_kwargs = {"num_groups": 1, "num_channels": 2}


class TestEmbedding(_NNModuleTorchBase):
    case_name = "embedding"
    inputs = (np.array([[1, 3, 4], [2, 0, 5]], dtype=np.int64),)
    module_ctor = nn.Embedding
    torch_module_ctor = torch.nn.Embedding
    module_kwargs = {"num_embeddings": 10, "embedding_dim": 4}
    torch_module_kwargs = {"num_embeddings": 10, "embedding_dim": 4}
    requires_grad = (False,)
    module_train_modes = (False,)

    def tensor_op_devices(self) -> tuple[str, ...]:
        return ("cpu",)


class TestDropout(_NNModuleTorchBase):
    case_name = "dropout_eval"
    inputs = (np.array([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]], dtype=np.float64),)
    module_ctor = nn.Dropout
    torch_module_ctor = torch.nn.Dropout
    module_kwargs = {"p": 0.2}
    torch_module_kwargs = {"p": 0.2}
    module_train_modes = (False,)


class TestPooling(_NNModuleTorchBase):
    __test__ = False

    class TestAvgPool2d(_NNModuleTorchBase):
        case_name = "avg_pool2d"
        inputs = (
            np.array(
                [
                    [
                        [
                            [1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 10.0, 11.0, 12.0],
                            [13.0, 14.0, 15.0, 16.0],
                        ]
                    ]
                ],
                dtype=np.float64,
            ),
        )
        module_ctor = nn.AvgPool2d
        torch_module_ctor = torch.nn.AvgPool2d
        module_kwargs = {"kernel_size": 2, "stride": 2}
        torch_module_kwargs = {"kernel_size": 2, "stride": 2}

    class TestMaxPool2d(_NNModuleTorchBase):
        case_name = "max_pool2d"
        inputs = (
            np.array(
                [
                    [
                        [
                            [1.0, 2.0, 3.0, 4.0],
                            [5.0, 6.0, 7.0, 8.0],
                            [9.0, 10.0, 11.0, 12.0],
                            [13.0, 14.0, 15.0, 16.0],
                        ]
                    ]
                ],
                dtype=np.float64,
            ),
        )
        module_ctor = nn.MaxPool2d
        torch_module_ctor = torch.nn.MaxPool2d
        module_kwargs = {"kernel_size": 2, "stride": 2}
        torch_module_kwargs = {"kernel_size": 2, "stride": 2}
