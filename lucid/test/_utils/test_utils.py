from typing import Any, Callable, ClassVar

import numpy as np
import torch

import lucid
from lucid.types import _NumPyArray
from lucid.test.core import TensorOpWithTorchBase, TensorOpTorchCase


def _to_numpy_or_float(x: Any) -> _NumPyArray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def _torch_histogramdd_2d(
    a: Any,
    bins: int | list[int],
    ranges: list[tuple[float, float]],
    density: bool = False,
) -> tuple[Any, Any]:
    hist_np, edges_np = np.histogramdd(
        np.array(_to_numpy_or_float(a), dtype=np.float64),
        bins=bins,
        range=ranges,
        density=density,
    )
    hist = torch.as_tensor(hist_np, device=a.device if hasattr(a, "device") else None)
    edges = torch.as_tensor(np.stack(tuple(edges_np), axis=0), device=a.device)
    return hist, edges


def _torch_histogram_1d(a: Any) -> tuple[Any, Any]:
    hist, edges = torch.histogram(a, bins=4, range=(0.0, 2.0), density=False)
    return hist, torch.stack((edges,), dim=0)


def _torch_histogram2d(
    a: Any,
    b: Any,
    bins: list[int],
    ranges: list[tuple[float, float]],
    density: bool = False,
) -> tuple[Any, Any]:
    data = torch.stack((a, b), dim=1)
    hist_np, edges_np = np.histogramdd(
        np.array(_to_numpy_or_float(data), dtype=np.float64),
        bins=bins,
        range=ranges,
        density=density,
    )
    hist = torch.as_tensor(hist_np, device=data.device)
    edges = torch.as_tensor(np.stack(tuple(edges_np), axis=0), device=data.device)
    return hist, edges


def _torch_unique_sorted_inverse(a: Any) -> tuple[Any, Any]:
    unique, inverse = torch.unique(a, sorted=True, return_inverse=True)
    return unique, inverse


def _unwrap_callable(operation: Any) -> Callable[..., Any]:
    if isinstance(operation, staticmethod):
        return operation.__func__
    if callable(operation):
        return operation
    raise TypeError(f"Expected callable operation, got: {type(operation)!r}")


def _as_numpy(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_as_numpy(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "data"):
        return np.array(value.data)
    return np.array(value)


def _assert_nested_allclose(
    tester: Any, lhs: Any, rhs: Any, *, rtol: float, atol: float
) -> None:
    lhs_np = _as_numpy(lhs)
    rhs_np = _as_numpy(rhs)

    if isinstance(lhs_np, tuple) or isinstance(rhs_np, tuple):
        assert isinstance(lhs_np, tuple)
        assert isinstance(rhs_np, tuple)
        assert len(lhs_np) == len(rhs_np)
        for lhs_item, rhs_item in zip(lhs_np, rhs_np):
            _assert_nested_allclose(tester, lhs_item, rhs_item, rtol=rtol, atol=atol)
        return

    tester.assert_tensor_allclose(lhs_np, rhs_np, rtol=rtol, atol=atol)


def _assert_nested_finite(tester: Any, value: Any) -> None:
    value_np = _as_numpy(value)

    if isinstance(value_np, tuple):
        for item in value_np:
            _assert_nested_finite(tester, item)
        return

    np.testing.assert_(np.isfinite(np.array(value_np)).all())


class _UtilsTorchOpBase(TensorOpWithTorchBase):
    case_name: ClassVar[str]
    inputs: ClassVar[tuple[Any, ...]]
    input_dtypes: ClassVar[tuple[Any | None, ...] | None] = None
    forward_op: ClassVar[Callable[..., Any]]
    torch_forward: ClassVar[Callable[..., Any] | None]
    expected_forward: ClassVar[Any]

    requires_grad: ClassVar[bool | tuple[bool, ...]] = True
    expected_input_grads: ClassVar[tuple[object | None, ...] | None] = None
    check_finite: ClassVar[bool] = True
    rtol: ClassVar[float] = 1e-7
    atol: ClassVar[float] = 1e-8
    compare_grad_with_torch: ClassVar[bool] = True

    def tensor_op_devices(self) -> tuple[str, ...]:
        return ("cpu", "gpu")

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
        for data, requires_grad, dtype in zip(self.inputs, requires, dtypes):
            outputs.append(self.tensor(data, requires_grad=requires_grad, dtype=dtype))
        return tuple(outputs)

    def _build_case(self) -> TensorOpTorchCase:
        forward_raw = vars(type(self)).get("forward_op")
        torch_raw = vars(type(self)).get("torch_forward")

        forward_op = _unwrap_callable(forward_raw)
        torch_forward = None if torch_raw is None else _unwrap_callable(torch_raw)

        return TensorOpTorchCase(
            name=self.case_name,
            build_inputs=lambda _case: self._build_inputs(),
            forward_op=forward_op,
            expected_forward=self.expected_forward,
            expected_input_grads=self.expected_input_grads,
            requires_grad=self._normalize_bool_flags(
                self.requires_grad, len(self.inputs)
            ),
            check_finite=self.check_finite,
            rtol=self.rtol,
            atol=self.atol,
            torch_forward=torch_forward,
            compare_grad_with_torch=self.compare_grad_with_torch,
        )

    def tensor_op_cases(self) -> tuple[TensorOpTorchCase, ...]:
        return (self._build_case(),)


class _UtilsTupleTorchOpBase(_UtilsTorchOpBase):
    compare_grad_with_torch: ClassVar[bool] = False

    def _run_tensor_op_case(
        self,
        case: TensorOpTorchCase,
        device: str = "cpu",
    ) -> None:
        inputs = self._prepare_inputs(case, case.build_inputs(self), device=device)
        out = case.forward_op(*inputs)

        _assert_nested_allclose(
            self,
            out,
            case.expected_forward,
            rtol=case.rtol,
            atol=case.atol,
        )

        if case.check_finite:
            _assert_nested_finite(self, out)

        if case.torch_forward is None:
            return

        torch = self.import_module("torch")
        torch_inputs = self._to_torch_inputs(inputs, torch)
        torch_out = case.torch_forward(*torch_inputs)

        _assert_nested_allclose(
            self,
            out,
            _as_numpy(torch_out),
            rtol=case.rtol,
            atol=case.atol,
        )

        if case.expected_input_grads is not None:
            raise AssertionError(
                "Tuple-output utils tests do not support explicit expected_input_grads."
            )


class TestReshape(_UtilsTorchOpBase):
    case_name = "reshape"
    inputs = (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)
    forward_op = staticmethod(lambda a: lucid.reshape(a, shape=(1, 4)))
    torch_forward = staticmethod(lambda a: torch.reshape(a, (1, 4)))
    expected_forward = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)


class TestSqueeze(_UtilsTorchOpBase):
    __test__ = False

    class TestDefault(_UtilsTorchOpBase):
        case_name = "squeeze_default"
        inputs = (np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.squeeze(a))
        torch_forward = staticmethod(torch.squeeze)
        expected_forward = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    class TestAxis(_UtilsTorchOpBase):
        case_name = "squeeze_axis"
        inputs = (np.array([[[1.0, 2.0]]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.squeeze(a, axis=1))
        torch_forward = staticmethod(lambda a: torch.squeeze(a, dim=1))
        expected_forward = np.array([[1.0, 2.0]], dtype=np.float64)


class TestUnsqueeze(_UtilsTorchOpBase):
    __test__ = False

    class TestAxis0(_UtilsTorchOpBase):
        case_name = "unsqueeze_axis0"
        inputs = (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.unsqueeze(a, axis=0))
        torch_forward = staticmethod(lambda a: torch.unsqueeze(a, dim=0))
        expected_forward = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64)

    class TestAxis1(_UtilsTorchOpBase):
        case_name = "unsqueeze_axis1"
        inputs = (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.unsqueeze(a, axis=1))
        torch_forward = staticmethod(lambda a: torch.unsqueeze(a, dim=1))
        expected_forward = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float64)


class TestExpandDims(TestUnsqueeze.TestAxis1):
    case_name = "expand_dims"
    forward_op = staticmethod(lambda a: lucid.expand_dims(a, axis=1))
    torch_forward = staticmethod(lambda a: torch.unsqueeze(a, dim=1))


class TestRavel(_UtilsTorchOpBase):
    case_name = "ravel"
    inputs = (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)
    forward_op = staticmethod(lucid.ravel)
    torch_forward = staticmethod(lambda a: torch.reshape(a, (-1,)))
    expected_forward = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)


class TestStack(_UtilsTorchOpBase):
    __test__ = False

    class TestDefault(_UtilsTorchOpBase):
        case_name = "stack_default"
        inputs = (
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0], dtype=np.float64),
        )
        forward_op = staticmethod(lambda a, b: lucid.stack((a, b), axis=0))
        torch_forward = staticmethod(lambda a, b: torch.stack((a, b), dim=0))
        expected_forward = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    class TestAxis(_UtilsTorchOpBase):
        case_name = "stack_axis1"
        inputs = (
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0], dtype=np.float64),
        )
        forward_op = staticmethod(lambda a, b: lucid.stack((a, b), axis=1))
        torch_forward = staticmethod(lambda a, b: torch.stack((a, b), dim=1))
        expected_forward = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64)


class TestHstack(_UtilsTorchOpBase):
    case_name = "hstack"
    inputs = (
        np.array([[1.0], [2.0]], dtype=np.float64),
        np.array([[3.0], [4.0]], dtype=np.float64),
    )
    forward_op = staticmethod(lambda a, b: lucid.hstack((a, b)))
    torch_forward = staticmethod(lambda a, b: torch.hstack((a, b)))
    expected_forward = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64)


class TestVstack(_UtilsTorchOpBase):
    case_name = "vstack"
    inputs = (
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([3.0, 4.0], dtype=np.float64),
    )
    forward_op = staticmethod(lambda a, b: lucid.vstack((a, b)))
    torch_forward = staticmethod(lambda a, b: torch.vstack((a, b)))
    expected_forward = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)


class TestConcatenate(_UtilsTorchOpBase):
    __test__ = False

    class TestAxis0(_UtilsTorchOpBase):
        case_name = "concatenate_axis0"
        inputs = (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
        )
        forward_op = staticmethod(lambda a, b: lucid.concatenate((a, b), axis=0))
        torch_forward = staticmethod(lambda a, b: torch.cat((a, b), dim=0))
        expected_forward = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            dtype=np.float64,
        )

    class TestAxis1(_UtilsTorchOpBase):
        case_name = "concatenate_axis1"
        inputs = (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
        )
        forward_op = staticmethod(lambda a, b: lucid.concatenate((a, b), axis=1))
        torch_forward = staticmethod(lambda a, b: torch.cat((a, b), dim=1))
        expected_forward = np.array(
            [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]],
            dtype=np.float64,
        )


class TestPad(_UtilsTorchOpBase):
    __test__ = False

    class TestPadTuple(_UtilsTorchOpBase):
        case_name = "pad_tuple"
        inputs = (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.pad(a, pad_width=((1, 1), (2, 0))))
        torch_forward = staticmethod(
            lambda a: torch.nn.functional.pad(a, (2, 0, 1, 1), mode="constant", value=0)
        )
        expected_forward = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 3.0, 4.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

    class TestPadInt(_UtilsTorchOpBase):
        case_name = "pad_int"
        inputs = (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.pad(a, pad_width=1))
        torch_forward = staticmethod(
            lambda a: torch.nn.functional.pad(a, (1, 1, 1, 1), mode="constant", value=0)
        )
        expected_forward = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 0.0],
                [0.0, 3.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )


class TestRepeat(_UtilsTorchOpBase):
    __test__ = False

    class TestFlattened(_UtilsTorchOpBase):
        case_name = "repeat_axis_none"
        inputs = (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.repeat(a, repeats=2, axis=None))
        torch_forward = staticmethod(
            lambda a: torch.repeat_interleave(a.reshape(-1), repeats=2)
        )
        expected_forward = np.array(
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], dtype=np.float64
        )

    class TestAxis1(_UtilsTorchOpBase):
        case_name = "repeat_axis1"
        inputs = (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.repeat(a, repeats=2, axis=1))
        torch_forward = staticmethod(
            lambda a: torch.repeat_interleave(a, repeats=2, dim=1)
        )
        expected_forward = np.array(
            [[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]],
            dtype=np.float64,
        )


class TestTile(_UtilsTorchOpBase):
    case_name = "tile"
    inputs = (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)
    forward_op = staticmethod(lambda a: lucid.tile(a, reps=(2, 1)))
    torch_forward = staticmethod(lambda a: torch.tile(a, (2, 1)))
    expected_forward = np.array(
        [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float64
    )


class TestFlatten(_UtilsTorchOpBase):
    __test__ = False

    class TestDefault(_UtilsTorchOpBase):
        case_name = "flatten_default"
        inputs = (
            np.array(
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64
            ),
        )
        forward_op = staticmethod(lambda a: lucid.flatten(a))
        torch_forward = staticmethod(lambda a: torch.flatten(a))
        expected_forward = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64
        )

    class TestPartial(_UtilsTorchOpBase):
        case_name = "flatten_partial"
        inputs = (
            np.array(
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float64
            ),
        )
        forward_op = staticmethod(lambda a: lucid.flatten(a, start_axis=1, end_axis=-1))
        torch_forward = staticmethod(lambda a: torch.flatten(a, start_dim=1, end_dim=2))
        expected_forward = np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            dtype=np.float64,
        )


class TestMeshgrid(_UtilsTupleTorchOpBase):
    __test__ = False

    class TestIJ(_UtilsTupleTorchOpBase):
        case_name = "meshgrid_ij"
        inputs = (np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0]))
        forward_op = staticmethod(lambda a, b: lucid.meshgrid(a, b, indexing="ij"))
        torch_forward = staticmethod(lambda a, b: torch.meshgrid(a, b, indexing="ij"))
        expected_forward = (
            np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64),
            np.array([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]], dtype=np.float64),
        )

    class TestXY(_UtilsTupleTorchOpBase):
        case_name = "meshgrid_xy"
        inputs = (np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0]))
        forward_op = staticmethod(lambda a, b: lucid.meshgrid(a, b, indexing="xy"))
        torch_forward = staticmethod(lambda a, b: torch.meshgrid(a, b, indexing="xy"))
        expected_forward = (
            np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float64),
            np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]], dtype=np.float64),
        )


class TestSplit(_UtilsTupleTorchOpBase):
    __test__ = False

    class TestSectionByInt(_UtilsTupleTorchOpBase):
        case_name = "split_section_int"
        inputs = (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.split(a, size_or_sections=1, axis=0))
        torch_forward = staticmethod(
            lambda a: torch.split(a, split_size_or_sections=1, dim=0)
        )
        expected_forward = (
            np.array([[1.0, 2.0]], dtype=np.float64),
            np.array([[3.0, 4.0]], dtype=np.float64),
            np.array([[5.0, 6.0]], dtype=np.float64),
        )

    class TestSectionByListAxis1(_UtilsTupleTorchOpBase):
        case_name = "split_section_list_axis1"
        inputs = (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),)
        forward_op = staticmethod(
            lambda a: lucid.split(a, size_or_sections=[1, 2], axis=1)
        )
        torch_forward = staticmethod(
            lambda a: torch.split(a, split_size_or_sections=[1, 2], dim=1)
        )
        expected_forward = (
            np.array([[1.0], [4.0]], dtype=np.float64),
            np.array([[2.0, 3.0], [5.0, 6.0]], dtype=np.float64),
        )


class TestTril(_UtilsTorchOpBase):
    __test__ = False

    class TestDefault(_UtilsTorchOpBase):
        case_name = "tril_default"
        inputs = (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.tril(a))
        torch_forward = staticmethod(lambda a: torch.tril(a))
        expected_forward = np.array(
            [[1.0, 0.0, 0.0], [4.0, 5.0, 0.0]], dtype=np.float64
        )

    class TestDiagonal(_UtilsTorchOpBase):
        case_name = "tril_diagonal1"
        inputs = (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.tril(a, diagonal=1))
        torch_forward = staticmethod(lambda a: torch.tril(a, diagonal=1))
        expected_forward = np.array(
            [[1.0, 2.0, 0.0], [4.0, 5.0, 6.0]], dtype=np.float64
        )


class TestTriu(_UtilsTorchOpBase):
    __test__ = False

    class TestDefault(_UtilsTorchOpBase):
        case_name = "triu_default"
        inputs = (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.triu(a))
        torch_forward = staticmethod(lambda a: torch.triu(a))
        expected_forward = np.array(
            [[1.0, 2.0, 3.0], [0.0, 5.0, 6.0]], dtype=np.float64
        )

    class TestDiagonal(_UtilsTorchOpBase):
        case_name = "triu_diagonal_neg1"
        inputs = (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.triu(a, diagonal=-1))
        torch_forward = staticmethod(lambda a: torch.triu(a, diagonal=-1))
        expected_forward = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64
        )


class TestBroadcastTo(_UtilsTorchOpBase):
    case_name = "broadcast_to"
    inputs = (np.array([[1.0, 2.0]], dtype=np.float64),)
    forward_op = staticmethod(lambda a: lucid.broadcast_to(a, shape=(2, 2)))
    torch_forward = staticmethod(lambda a: torch.broadcast_to(a, (2, 2)))
    expected_forward = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float64)


class TestExpand(_UtilsTorchOpBase):
    __test__ = False

    class TestDefault(_UtilsTorchOpBase):
        case_name = "expand"
        inputs = (np.array([[1.0, 2.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.expand(a, 3, 2))
        torch_forward = staticmethod(lambda a: torch.broadcast_to(a, (3, 2)))
        expected_forward = np.array(
            [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
            dtype=np.float64,
        )

    class TestKeepDim(_UtilsTorchOpBase):
        case_name = "expand_keep_dim"
        inputs = (np.array([[1.0, 2.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.expand(a, 1, -1))
        torch_forward = staticmethod(lambda a: torch.broadcast_to(a, (1, 2)))
        expected_forward = np.array([[1.0, 2.0]], dtype=np.float64)


class TestChunk(_UtilsTupleTorchOpBase):
    __test__ = False

    class TestAxis1(_UtilsTupleTorchOpBase):
        case_name = "chunk_axis1"
        inputs = (
            np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64),
        )
        forward_op = staticmethod(lambda a: lucid.chunk(a, chunks=2, axis=1))
        torch_forward = staticmethod(lambda a: torch.chunk(a, chunks=2, dim=1))
        expected_forward = (
            np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float64),
            np.array([[3.0, 4.0], [7.0, 8.0]], dtype=np.float64),
        )

    class TestAxis0(_UtilsTupleTorchOpBase):
        case_name = "chunk_axis0"
        inputs = (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.chunk(a, chunks=3, axis=0))
        torch_forward = staticmethod(lambda a: torch.chunk(a, chunks=3, dim=0))
        expected_forward = (
            np.array([[1.0, 2.0]], dtype=np.float64),
            np.array([[3.0, 4.0]], dtype=np.float64),
            np.array([[5.0, 6.0]], dtype=np.float64),
        )


class TestMaskedFill(_UtilsTorchOpBase):
    case_name = "masked_fill"
    inputs = (
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        np.array([[True, False], [False, True]]),
    )
    forward_op = staticmethod(
        lambda a, mask: lucid.masked_fill(a, mask=mask, value=0.0)
    )
    torch_forward = staticmethod(
        lambda a, mask: torch.where(
            mask, torch.tensor(0.0, device=a.device, dtype=a.dtype), a
        )
    )
    expected_forward = np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float64)
    input_dtypes = (None, lucid.Bool if hasattr(lucid, "Bool") else None)
    requires_grad = (True, False)


class TestGather(_UtilsTorchOpBase):
    case_name = "gather"
    inputs = (
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        np.array([[2, 0, 1], [1, 2, 0]], dtype=np.int64),
    )
    forward_op = staticmethod(lambda a, idx: lucid.gather(a, dim=1, index=idx))
    torch_forward = staticmethod(lambda a, idx: torch.gather(a, dim=1, index=idx))
    expected_forward = np.array([[3.0, 1.0, 2.0], [5.0, 6.0, 4.0]], dtype=np.float64)
    input_dtypes = (None, lucid.Int64)
    requires_grad = (True, False)


class TestRoll(_UtilsTorchOpBase):
    __test__ = False

    class TestDefault(_UtilsTorchOpBase):
        case_name = "roll_default"
        inputs = (np.array([1.0, 2.0, 3.0, 4.0]),)
        forward_op = staticmethod(lambda a: lucid.roll(a, shifts=1, axis=None))
        torch_forward = staticmethod(lambda a: torch.roll(a, shifts=1, dims=None))
        expected_forward = np.array([4.0, 1.0, 2.0, 3.0], dtype=np.float64)

    class TestTupleAxes(_UtilsTorchOpBase):
        case_name = "roll_tuple_axes"
        inputs = (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),)
        forward_op = staticmethod(lambda a: lucid.roll(a, shifts=(1, 1), axis=(0, 1)))
        torch_forward = staticmethod(
            lambda a: torch.roll(a, shifts=(1, 1), dims=(0, 1))
        )
        expected_forward = np.array(
            [[6.0, 4.0, 5.0], [3.0, 1.0, 2.0]], dtype=np.float64
        )


class TestUnbind(_UtilsTupleTorchOpBase):
    __test__ = False

    class TestAxis0(_UtilsTupleTorchOpBase):
        case_name = "unbind_axis0"
        inputs = (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.unbind(a, axis=0))
        torch_forward = staticmethod(lambda a: torch.unbind(a, dim=0))
        expected_forward = (
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0], dtype=np.float64),
            np.array([5.0, 6.0], dtype=np.float64),
        )

    class TestAxis1(_UtilsTupleTorchOpBase):
        case_name = "unbind_axis1"
        inputs = (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.unbind(a, axis=1))
        torch_forward = staticmethod(lambda a: torch.unbind(a, dim=1))
        expected_forward = (
            np.array([1.0, 3.0], dtype=np.float64),
            np.array([2.0, 4.0], dtype=np.float64),
        )


class TestSort(_UtilsTupleTorchOpBase):
    __test__ = False

    class TestAscending(_UtilsTupleTorchOpBase):
        case_name = "sort_ascending"
        inputs = (np.array([[3.0, 1.0], [2.0, 4.0]], dtype=np.float64),)
        forward_op = staticmethod(
            lambda a: lucid.sort(
                a, axis=-1, descending=False, kind="quicksort", stable=False
            )
        )
        torch_forward = staticmethod(
            lambda a: torch.sort(a, dim=-1, descending=False, stable=False)
        )
        expected_forward = (
            np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64),
            np.array([[1, 0], [0, 1]], dtype=np.int32),
        )

    class TestDescending(_UtilsTupleTorchOpBase):
        case_name = "sort_descending"
        inputs = (np.array([[3.0, 1.0], [2.0, 4.0]], dtype=np.float64),)
        forward_op = staticmethod(
            lambda a: lucid.sort(
                a, axis=-1, descending=True, kind="quicksort", stable=False
            )
        )
        torch_forward = staticmethod(
            lambda a: torch.sort(a, dim=-1, descending=True, stable=False)
        )
        expected_forward = (
            np.array([[3.0, 1.0], [4.0, 2.0]], dtype=np.float64),
            np.array([[0, 1], [1, 0]], dtype=np.int32),
        )


class TestArgsort(_UtilsTorchOpBase):
    case_name = "argsort"
    inputs = (np.array([[3.0, 1.0], [2.0, 4.0]], dtype=np.float64),)
    forward_op = staticmethod(lambda a: lucid.argsort(a, axis=-1, descending=False))
    torch_forward = staticmethod(lambda a: torch.argsort(a, dim=-1, descending=False))
    expected_forward = np.array([[1, 0], [0, 1]], dtype=np.int32)
    compare_grad_with_torch = False


class TestArgmin(_UtilsTorchOpBase):
    __test__ = False

    class TestAxisNone(_UtilsTorchOpBase):
        case_name = "argmin_axis_none"
        inputs = (np.array([[5.0, 1.0, 7.0], [2.0, 9.0, 0.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.argmin(a))
        torch_forward = staticmethod(lambda a: torch.argmin(a, dim=0))
        expected_forward = np.array([1, 0, 1], dtype=np.int32)
        compare_grad_with_torch = False

    class TestKeepDims(_UtilsTorchOpBase):
        case_name = "argmin_keepdims"
        inputs = (np.array([[5.0, 1.0, 7.0], [2.0, 9.0, 0.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.argmin(a, axis=-1, keepdims=True))
        torch_forward = staticmethod(lambda a: torch.argmin(a, dim=-1, keepdim=True))
        expected_forward = np.array([[1], [2]], dtype=np.int32)
        compare_grad_with_torch = False


class TestArgmax(_UtilsTorchOpBase):
    __test__ = False

    class TestAxisNone(_UtilsTorchOpBase):
        case_name = "argmax_axis_none"
        inputs = (np.array([[5.0, 1.0, 7.0], [2.0, 9.0, 0.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.argmax(a))
        torch_forward = staticmethod(lambda a: torch.argmax(a, dim=0))
        expected_forward = np.array([0, 1, 0], dtype=np.int32)
        compare_grad_with_torch = False

    class TestKeepDims(_UtilsTorchOpBase):
        case_name = "argmax_keepdims"
        inputs = (np.array([[5.0, 1.0, 7.0], [2.0, 9.0, 0.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.argmax(a, axis=-1, keepdims=True))
        torch_forward = staticmethod(lambda a: torch.argmax(a, dim=-1, keepdim=True))
        expected_forward = np.array([[2], [1]], dtype=np.int32)
        compare_grad_with_torch = False


class TestNonzero(_UtilsTorchOpBase):
    case_name = "nonzero"
    inputs = (np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], dtype=np.float64),)
    forward_op = staticmethod(lambda a: lucid.nonzero(a))
    torch_forward = staticmethod(lambda a: torch.nonzero(a))
    expected_forward = np.array([[0, 1], [1, 0], [1, 2]], dtype=np.int32)
    compare_grad_with_torch = False


class TestUnique(_UtilsTorchOpBase):
    __test__ = False

    class TestSorted(_UtilsTorchOpBase):
        case_name = "unique_sorted"
        inputs = (np.array([3.0, 1.0, 2.0, 3.0, 2.0, 4.0], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.unique(a, sorted=True))
        torch_forward = staticmethod(lambda a: torch.unique(a, sorted=True))
        expected_forward = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        compare_grad_with_torch = False

    class TestUnsorted(_UtilsTorchOpBase):
        case_name = "unique_unsorted"
        inputs = (np.array([3.0, 1.0, 2.0, 3.0, 2.0, 4.0], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.unique(a, sorted=False))
        torch_forward = staticmethod(lambda a: torch.unique(a, sorted=False))
        expected_forward = np.array([3.0, 1.0, 2.0, 4.0], dtype=np.float64)
        compare_grad_with_torch = False


class TestUniqueWithInverse(_UtilsTupleTorchOpBase):
    case_name = "unique_with_inverse"
    inputs = (np.array([1.0, 2.0, 1.0, 3.0], dtype=np.float64),)
    forward_op = staticmethod(
        lambda a: lucid.unique(a, sorted=True, return_inverse=True)
    )
    torch_forward = staticmethod(_torch_unique_sorted_inverse)
    expected_forward = (
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([0, 1, 0, 2], dtype=np.int64),
    )
    compare_grad_with_torch = False


class TestTopk(_UtilsTupleTorchOpBase):
    __test__ = False

    class TestLargest(_UtilsTupleTorchOpBase):
        case_name = "topk_largest"
        inputs = (np.array([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.topk(a, k=2, axis=-1, largest=True))
        torch_forward = staticmethod(lambda a: torch.topk(a, k=2, dim=-1, largest=True))
        expected_forward = (
            np.array([[4.0, 2.0], [5.0, 3.0]], dtype=np.float64),
            np.array([[1, 2], [2, 0]], dtype=np.int32),
        )

    class TestSmallest(_UtilsTupleTorchOpBase):
        case_name = "topk_smallest"
        inputs = (np.array([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]], dtype=np.float64),)
        forward_op = staticmethod(lambda a: lucid.topk(a, k=2, axis=-1, largest=False))
        torch_forward = staticmethod(
            lambda a: torch.topk(a, k=2, dim=-1, largest=False)
        )
        expected_forward = (
            np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float64),
            np.array([[0, 2], [1, 0]], dtype=np.int32),
        )


class TestHistogramdd(_UtilsTupleTorchOpBase):
    case_name = "histogramdd"
    inputs = (
        np.array(
            [
                [0.2, 0.2],
                [1.2, 0.7],
                [1.8, 0.9],
                [0.6, 1.7],
            ],
            dtype=np.float64,
        ),
    )
    forward_op = staticmethod(
        lambda a: lucid.histogramdd(
            a,
            bins=[2, 2],
            range=[(0.0, 2.0), (0.0, 2.0)],
            density=False,
        )
    )
    torch_forward = staticmethod(
        lambda a: _torch_histogramdd_2d(a, bins=(2, 2), ranges=[(0.0, 2.0), (0.0, 2.0)])
    )
    expected_forward = (
        np.array([[1.0, 1.0], [2.0, 0.0]], dtype=np.float64),
        np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        ),
    )
    compare_grad_with_torch = False


class TestHistogram(_UtilsTupleTorchOpBase):
    case_name = "histogram"
    inputs = (np.array([0.2, 0.6, 1.2, 1.8], dtype=np.float64),)
    forward_op = staticmethod(
        lambda a: lucid.histogram(a, bins=4, range=(0.0, 2.0), density=False)
    )
    torch_forward = staticmethod(_torch_histogram_1d)
    expected_forward = (
        np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        np.array([[0.0, 0.5, 1.0, 1.5, 2.0]], dtype=np.float64),
    )
    compare_grad_with_torch = False


class TestHistogram2d(_UtilsTupleTorchOpBase):
    case_name = "histogram2d"
    inputs = (
        np.array([0.2, 0.6, 1.2, 1.8], dtype=np.float64),
        np.array([0.2, 0.7, 0.9, 1.7], dtype=np.float64),
    )
    forward_op = staticmethod(
        lambda a, b: lucid.histogram2d(
            a,
            b,
            bins=[2, 2],
            range=[(0.0, 2.0), (0.0, 2.0)],
            density=False,
        )
    )
    torch_forward = staticmethod(
        lambda a, b: _torch_histogram2d(
            a, b, bins=(2, 2), ranges=[(0.0, 2.0), (0.0, 2.0)]
        )
    )
    expected_forward = (
        np.array([[2.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        ),
    )
    compare_grad_with_torch = False
    compare_grad_with_torch = False


class TestWhere(_UtilsTorchOpBase):
    case_name = "where"
    inputs = (
        np.array([[True, False, True], [False, True, False]], dtype=np.bool_),
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64),
    )
    forward_op = staticmethod(lambda cond, a, b: lucid.where(cond, a, b))
    torch_forward = staticmethod(lambda cond, a, b: torch.where(cond, a, b))
    expected_forward = np.array([[1.0, 20.0, 3.0], [40.0, 5.0, 60.0]], dtype=np.float64)
    requires_grad = (False, True, True)


class TestDiagonal(_UtilsTorchOpBase):
    __test__ = False

    class TestDefault(_UtilsTorchOpBase):
        case_name = "diagonal_default"
        inputs = (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),)
        forward_op = staticmethod(
            lambda a: lucid.diagonal(a, offset=0, axis1=0, axis2=1)
        )
        torch_forward = staticmethod(
            lambda a: torch.diagonal(a, offset=0, dim1=0, dim2=1)
        )
        expected_forward = np.array([1.0, 5.0], dtype=np.float64)

    class TestOffset(_UtilsTorchOpBase):
        case_name = "diagonal_offset1"
        inputs = (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),)
        forward_op = staticmethod(
            lambda a: lucid.diagonal(a, offset=1, axis1=0, axis2=1)
        )
        torch_forward = staticmethod(
            lambda a: torch.diagonal(a, offset=1, dim1=0, dim2=1)
        )
        expected_forward = np.array([2.0, 6.0], dtype=np.float64)
