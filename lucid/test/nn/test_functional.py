from typing import Any, Callable, ClassVar

import numpy as np
import torch

import lucid
from lucid.test._utils.test_utils import _unwrap_callable
from lucid.test.core import TensorOpTorchCase, TensorOpWithTorchBase


class _NNFunctionalTorchBase(TensorOpWithTorchBase):
    _collect_nested_test_classes = True
    case_name: ClassVar[str]
    inputs: ClassVar[tuple[Any, ...]]

    forward_op: ClassVar[Callable[..., Any]]
    torch_forward: ClassVar[Callable[..., Any]]
    expected_forward: ClassVar[Any | None] = None

    requires_grad: ClassVar[bool | tuple[bool, ...]] = True
    input_dtypes: ClassVar[tuple[Any | None, ...] | None] = None

    expected_input_grads: ClassVar[tuple[object | None, ...] | None] = None
    compare_grad_with_torch: ClassVar[bool] = True
    check_finite: ClassVar[bool] = True

    rtol: ClassVar[float] = 1e-7
    atol: ClassVar[float] = 1e-8

    def tensor_op_devices(self) -> tuple[str, ...]:
        return ("cpu",)

    def _normalize_bool_flags(
        self,
        flags: bool | tuple[bool, ...],
        expected_len: int,
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

    def _normalize_input_dtypes(
        self,
        dtypes: tuple[Any | None, ...] | None,
        expected_len: int,
    ) -> tuple[Any | None, ...]:
        if dtypes is None:
            return (None,) * expected_len
        if len(dtypes) == 1 and expected_len > 1:
            return (dtypes[0],) * expected_len
        if len(dtypes) != expected_len:
            raise AssertionError(
                f"input_dtypes length {len(dtypes)} does not match inputs "
                f"(expected {expected_len}) for case '{self.case_name}'."
            )
        return tuple(dtypes)

    def _build_inputs(self) -> tuple[Any, ...]:
        requires = self._normalize_bool_flags(self.requires_grad, len(self.inputs))
        dtypes = self._normalize_input_dtypes(self.input_dtypes, len(self.inputs))
        outputs: list[Any] = []
        for item, requires_grad, dtype in zip(self.inputs, requires, dtypes):
            if item is None:
                outputs.append(None)
                continue
            outputs.append(self.tensor(item, requires_grad=requires_grad, dtype=dtype))
        return tuple(outputs)

    def _to_torch_inputs(self, inputs: tuple[Any, ...], torch: Any) -> tuple[Any, ...]:
        output: list[Any] = []
        for item in inputs:
            if item is None:
                output.append(None)
                continue
            output.extend(super()._to_torch_inputs((item,), torch))
        return tuple(output)

    def _to_numpy(self, value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(self._to_numpy(item) for item in value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if hasattr(value, "data"):
            return np.array(value.data)
        return np.array(value)

    def _build_expected_forward(self, torch_forward: Callable[..., Any]) -> Any:
        torch_inputs = self._to_torch_inputs(self._build_inputs(), torch)
        torch_out = torch_forward(*torch_inputs)
        return self._to_numpy(torch_out)

    def tensor_op_cases(self) -> tuple[TensorOpTorchCase, ...]:
        forward_op = _unwrap_callable(self.forward_op)
        torch_forward = _unwrap_callable(self.torch_forward)

        return (
            TensorOpTorchCase(
                name=self.case_name,
                build_inputs=lambda _: self._build_inputs(),
                forward_op=forward_op,
                expected_forward=(
                    self.expected_forward
                    if self.expected_forward is not None
                    else self._build_expected_forward(torch_forward)
                ),
                expected_input_grads=self.expected_input_grads,
                requires_grad=self._normalize_bool_flags(
                    self.requires_grad, len(self.inputs)
                ),
                check_finite=self.check_finite,
                torch_forward=torch_forward,
                compare_grad_with_torch=self.compare_grad_with_torch,
                rtol=self.rtol,
                atol=self.atol,
            ),
        )


class TestActivation(_NNFunctionalTorchBase):
    __test__ = False

    class TestRelu(_NNFunctionalTorchBase):
        case_name = "relu"
        inputs = (np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float64),)
        forward_op = staticmethod(lucid.nn.functional.relu)
        torch_forward = staticmethod(torch.relu)

    class TestLeakyRelu(_NNFunctionalTorchBase):
        case_name = "leaky_relu"
        inputs = (np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float64),)
        forward_op = staticmethod(
            lambda a: lucid.nn.functional.leaky_relu(a, negative_slope=0.2)
        )
        torch_forward = staticmethod(
            lambda a: torch.nn.functional.leaky_relu(a, negative_slope=0.2)
        )

    class TestElu(_NNFunctionalTorchBase):
        case_name = "elu"
        inputs = (np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.nn.functional.elu(a, alpha=1.2))
        torch_forward = staticmethod(lambda a: torch.nn.functional.elu(a, alpha=1.2))

    class TestSelu(_NNFunctionalTorchBase):
        case_name = "selu"
        inputs = (np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float64),)
        forward_op = staticmethod(lucid.nn.functional.selu)
        torch_forward = staticmethod(torch.selu)

    class TestSilu(_NNFunctionalTorchBase):
        case_name = "silu"
        inputs = (np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float64),)
        forward_op = staticmethod(lucid.nn.functional.silu)
        torch_forward = staticmethod(torch.nn.functional.silu)

    class TestSoftmax(_NNFunctionalTorchBase):
        case_name = "softmax"
        inputs = (np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.nn.functional.softmax(a, axis=1))
        torch_forward = staticmethod(lambda a: torch.nn.functional.softmax(a, dim=1))


class TestLinear(_NNFunctionalTorchBase):
    __test__ = False

    class TestLinear(_NNFunctionalTorchBase):
        case_name = "linear"
        inputs = (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        )
        forward_op = staticmethod(lucid.nn.functional.linear)
        torch_forward = staticmethod(torch.nn.functional.linear)

    class TestLinearNoBias(_NNFunctionalTorchBase):
        case_name = "linear_no_bias"
        inputs = (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64),
        )
        forward_op = staticmethod(lucid.nn.functional.linear)
        torch_forward = staticmethod(torch.nn.functional.linear)

    class TestBilinear(_NNFunctionalTorchBase):
        case_name = "bilinear"
        inputs = (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
            np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )
        forward_op = staticmethod(lucid.nn.functional.bilinear)
        torch_forward = staticmethod(torch.bilinear)


class TestLoss(_NNFunctionalTorchBase):
    __test__ = False

    class TestMSEMean(_NNFunctionalTorchBase):
        case_name = "mse_mean"
        inputs = (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[1.5, 1.0], [2.5, 3.5]], dtype=np.float64),
        )
        forward_op = staticmethod(lambda a, b: lucid.nn.functional.mse_loss(a, b))
        torch_forward = staticmethod(
            lambda a, b: torch.nn.functional.mse_loss(a, b, reduction="mean")
        )
        requires_grad = (True, False)

    class TestMSESum(_NNFunctionalTorchBase):
        case_name = "mse_sum"
        inputs = (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[2.0, 1.0], [0.0, 4.0]], dtype=np.float64),
        )
        forward_op = staticmethod(
            lambda a, b: lucid.nn.functional.mse_loss(a, b, reduction="sum")
        )
        torch_forward = staticmethod(
            lambda a, b: torch.nn.functional.mse_loss(a, b, reduction="sum")
        )
        requires_grad = (True, False)

    class TestHuber(_NNFunctionalTorchBase):
        case_name = "huber"
        inputs = (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[1.2, 1.9], [2.0, 5.0]], dtype=np.float64),
        )
        forward_op = staticmethod(
            lambda a, b: lucid.nn.functional.huber_loss(a, b, delta=1.0)
        )
        torch_forward = staticmethod(
            lambda a, b: torch.nn.functional.smooth_l1_loss(
                a, b, beta=1.0, reduction="mean"
            )
        )
        requires_grad = (True, False)
        atol = 1e-6
        rtol = 1e-6

    class TestBinaryCrossEntropy(_NNFunctionalTorchBase):
        case_name = "bce"
        inputs = (
            np.array([[0.1, 0.7], [0.2, 0.9]], dtype=np.float64),
            np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        )
        forward_op = staticmethod(
            lambda a, b: lucid.nn.functional.binary_cross_entropy(a, b)
        )
        torch_forward = staticmethod(torch.nn.functional.binary_cross_entropy)
        requires_grad = (True, False)

    class TestBCEWithLogits(_NNFunctionalTorchBase):
        case_name = "bce_with_logits"
        inputs = (
            np.array([[0.2, -1.0], [1.5, 0.1]], dtype=np.float64),
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        )
        forward_op = staticmethod(
            lambda a, b: lucid.nn.functional.binary_cross_entropy_with_logits(
                a, b, pos_weight=lucid.ones_like(a)
            )
        )
        torch_forward = staticmethod(
            lambda a, b: torch.nn.functional.binary_cross_entropy_with_logits(
                a, b, pos_weight=torch.ones_like(a)
            )
        )
        requires_grad = (True, False)

    class TestCrossEntropy(_NNFunctionalTorchBase):
        case_name = "cross_entropy"
        inputs = (
            np.array([[2.0, 1.0, 0.1], [0.1, 0.2, 3.0]], dtype=np.float64),
            np.array([1, 2], dtype=np.int64),
        )
        forward_op = staticmethod(
            lambda a, b: lucid.nn.functional.cross_entropy(a, b, reduction="mean")
        )
        torch_forward = staticmethod(
            lambda a, b: torch.nn.functional.cross_entropy(a, b, reduction="mean")
        )
        requires_grad = (True, False)


class TestNormalization(_NNFunctionalTorchBase):
    __test__ = False

    class TestNormalize(_NNFunctionalTorchBase):
        case_name = "normalize"
        inputs = (np.array([[3.0, 4.0], [1.0, 2.0]], dtype=np.float64),)
        forward_op = staticmethod(
            lambda a: lucid.nn.functional.normalize(a, ord=2, axis=1)
        )
        torch_forward = staticmethod(
            lambda a: torch.nn.functional.normalize(a, p=2, dim=1, eps=1e-12)
        )

    class TestLayerNorm(_NNFunctionalTorchBase):
        case_name = "layer_norm"
        inputs = (
            np.array(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                dtype=np.float64,
            ),
        )
        forward_op = staticmethod(
            lambda a: lucid.nn.functional.layer_norm(a, normalized_shape=(4,))
        )
        torch_forward = staticmethod(
            lambda a: torch.nn.functional.layer_norm(
                a, normalized_shape=(4,), weight=None, bias=None
            )
        )

    class TestBatchNorm(_NNFunctionalTorchBase):
        case_name = "batch_norm"
        inputs = (
            np.array(
                [[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [2.0, 1.0, 0.0, 2.0]],
                dtype=np.float64,
            ).reshape(1, 3, 2, 2),
            np.array([0.5, 0.4, 0.3], dtype=np.float64),
            np.array([1.5, 1.2, 1.1], dtype=np.float64),
            np.array([1.0, 1.0, 1.0], dtype=np.float64),
            np.array([0.1, 0.0, -0.1], dtype=np.float64),
        )
        forward_op = staticmethod(
            lambda a, b, c, d, e: lucid.nn.functional.batch_norm(
                a, b, c, d, e, training=False, momentum=0.1, eps=1e-5
            )
        )
        torch_forward = staticmethod(
            lambda a, b, c, d, e: torch.nn.functional.batch_norm(
                a, b, c, d, e, training=False, momentum=0.1, eps=1e-5
            )
        )
        requires_grad = (True, False, False, True, True)

    class TestGroupNorm(_NNFunctionalTorchBase):
        case_name = "group_norm"
        inputs = (
            np.arange(1, 33, dtype=np.float64).reshape(2, 4, 2, 2),
            np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        forward_op = staticmethod(
            lambda a, c, d: lucid.nn.functional.group_norm(a, 2, c, d, eps=1e-5)
        )
        torch_forward = staticmethod(
            lambda a, c, d: torch.nn.functional.group_norm(
                a, num_groups=2, weight=c, bias=d, eps=1e-5
            )
        )
        requires_grad = (True, True, True)


class TestPool(_NNFunctionalTorchBase):
    __test__ = False

    class TestAvgPool2d(_NNFunctionalTorchBase):
        case_name = "avg_pool2d"
        inputs = (np.arange(1, 17, dtype=np.float64).reshape(1, 1, 4, 4),)
        forward_op = staticmethod(
            lambda a: lucid.nn.functional.avg_pool2d(
                a, kernel_size=(2, 2), stride=(2, 2)
            )
        )
        torch_forward = staticmethod(
            lambda a: torch.nn.functional.avg_pool2d(
                a, kernel_size=(2, 2), stride=(2, 2)
            )
        )

    class TestMaxPool2d(_NNFunctionalTorchBase):
        case_name = "max_pool2d"
        inputs = (np.arange(1, 17, dtype=np.float64).reshape(1, 1, 4, 4),)
        forward_op = staticmethod(
            lambda a: lucid.nn.functional.max_pool2d(
                a, kernel_size=(2, 2), stride=(2, 2)
            )
        )
        torch_forward = staticmethod(
            lambda a: torch.nn.functional.max_pool2d(
                a, kernel_size=(2, 2), stride=(2, 2)
            )
        )


class TestConvolution(_NNFunctionalTorchBase):
    __test__ = False

    class TestConv2d(_NNFunctionalTorchBase):
        case_name = "conv2d"
        inputs = (
            np.arange(16.0, dtype=np.float64).reshape(1, 1, 4, 4),
            np.array(
                [[[[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]]],
                dtype=np.float64,
            ),
            np.array([0.25], dtype=np.float64),
        )
        forward_op = staticmethod(lucid.nn.functional.conv2d)
        torch_forward = staticmethod(
            lambda a, b, c: torch.nn.functional.conv2d(
                a, b, c, stride=1, padding=0, dilation=1
            )
        )


class TestEmbedding(_NNFunctionalTorchBase):
    __test__ = False

    class TestEmbedding(_NNFunctionalTorchBase):
        case_name = "embedding"
        inputs = (
            np.array([[0, 1, 2], [1, 3, 4]], dtype=np.int64),
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                ],
                dtype=np.float64,
            ),
        )
        forward_op = staticmethod(
            lambda a, b: lucid.nn.functional.embedding(
                a, b, padding_idx=0, max_norm=3.0, norm_type=2.0
            )
        )
        torch_forward = staticmethod(
            lambda a, b: torch.nn.functional.embedding(
                a, b, padding_idx=0, max_norm=3.0, norm_type=2.0
            )
        )
        requires_grad = (False, True)

    class TestOneHot(_NNFunctionalTorchBase):
        case_name = "one_hot"
        inputs = (np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int64),)
        forward_op = staticmethod(
            lambda a: lucid.nn.functional.one_hot(a, num_classes=5)
        )
        torch_forward = staticmethod(
            lambda a: torch.nn.functional.one_hot(a, num_classes=5)
        )
        requires_grad = (False,)
        compare_grad_with_torch = False
