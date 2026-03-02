import importlib
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest


__all__ = [
    "LucidBaseCase",
    "TensorOpCase",
    "TensorOpTorchCase",
    "TensorOpBase",
    "TensorOpTorchBase",
    "ModuleImportBase",
    "TensorFactoryBase",
]


class LucidBaseCase:
    random_seed: int | None = 42

    @pytest.fixture(autouse=True)
    def _ensure_seed(self) -> Generator[None, None, None]:
        if self.random_seed is None:
            yield
            return

        random_mod = importlib.import_module("lucid.random")
        previous_seed = random_mod.get_seed()
        random_mod.seed(self.random_seed)

        try:
            yield
        finally:
            random_mod.seed(previous_seed)

    def import_module(self, module_name: str) -> Any:
        return importlib.import_module(module_name)

    def tensor(self, data: Any, **kwargs: Any) -> Any:
        root = self.import_module("lucid")
        return root.tensor(data, **kwargs)

    def assert_tensor_array_equal(
        self,
        lhs: object,
        rhs: object,
    ) -> None:
        lhs_arr = np.array(lhs.data if hasattr(lhs, "data") else lhs)
        rhs_arr = np.array(rhs.data if hasattr(rhs, "data") else rhs)
        np.testing.assert_array_equal(lhs_arr, rhs_arr)

    def assert_tensor_allclose(
        self,
        lhs: object,
        rhs: object,
        *,
        rtol: float = 1e-7,
        atol: float = 1e-8,
    ) -> None:
        lhs_arr = np.array(lhs.data if hasattr(lhs, "data") else lhs)
        rhs_arr = np.array(rhs.data if hasattr(rhs, "data") else rhs)
        np.testing.assert_allclose(lhs_arr, rhs_arr, rtol=rtol, atol=atol)


@dataclass(frozen=True)
class TensorOpCase:
    name: str
    build_inputs: Callable[["LucidBaseCase"], tuple[Any, ...]]
    forward_op: Callable[..., Any]
    expected_forward: object
    expected_input_grads: tuple[object | None, ...] | None = None
    requires_grad: tuple[bool, ...] = ()
    check_finite: bool = True
    rtol: float = 1e-7
    atol: float = 1e-8


@dataclass(frozen=True)
class TensorOpTorchCase(TensorOpCase):
    torch_forward: Callable[..., Any] | None = None
    compare_grad_with_torch: bool = True


class TensorOpBase(LucidBaseCase):
    def tensor_op_devices(self) -> tuple[str, ...]:
        return ("cpu",)

    def _is_gpu_device_supported(self) -> bool:
        try:
            importlib.import_module("mlx.core")
        except ModuleNotFoundError:
            return False
        except Exception:
            return False

        try:
            import lucid._backend.metal as metal

            mx = getattr(metal, "mx", None)
            if mx is None or not hasattr(mx, "metal"):
                return False
            return bool(mx.metal.is_available())
        except Exception:
            return False

    def _is_device_supported(self, device: str) -> bool:
        if device == "gpu":
            return self._is_gpu_device_supported()
        return True

    def tensor_op_cases(self) -> tuple[TensorOpCase, ...]:
        return ()

    def _prepare_inputs(
        self,
        case: TensorOpCase,
        inputs: tuple[Any, ...],
        device: str = "cpu",
    ) -> tuple[Any, ...]:
        if case.requires_grad and len(case.requires_grad) != len(inputs):
            raise AssertionError(
                f"requires_grad length {len(case.requires_grad)} does not match inputs"
                f" {len(inputs)} for case '{case.name}'."
            )

        flags = (
            case.requires_grad
            if case.requires_grad
            else tuple(hasattr(item, "requires_grad") for item in inputs)
        )

        prepared = []
        for item, need_grad in zip(inputs, flags):
            if not hasattr(item, "requires_grad"):
                prepared.append(item.to(device) if hasattr(item, "to") else item)
                continue

            item.grad = None
            item.requires_grad = bool(need_grad)
            item = item.to(device)
            prepared.append(item)

        return tuple(prepared)

    def test_tensor_ops_forward_backward_sanity(self) -> None:
        cases = self.tensor_op_cases()
        assert len(cases) > 0, f"No TensorOpCase defined in {type(self).__name__}."

        for case in cases:
            for device in self.tensor_op_devices():
                if not self._is_device_supported(device):
                    continue
                self._run_tensor_op_case(case, device=device)

    def _run_tensor_op_case(self, case: TensorOpCase, device: str = "cpu") -> None:
        inputs = self._prepare_inputs(case, case.build_inputs(self), device=device)
        out = case.forward_op(*inputs)

        self._assert_forward_output(
            case, out, case.expected_forward, rtol=case.rtol, atol=case.atol
        )

        if case.expected_input_grads is None:
            return

        assert hasattr(
            out, "backward"
        ), f"Expected backward-capable tensor output for {case.name}."
        self._run_backward_and_check_grads(case, out, inputs, case.expected_input_grads)

    def _assert_forward_output(
        self,
        case: TensorOpCase,
        out: Any,
        expected_forward: object,
        *,
        rtol: float = 1e-7,
        atol: float = 1e-8,
    ) -> None:
        self.assert_tensor_allclose(out, expected_forward, rtol=rtol, atol=atol)

        if case.check_finite:
            np.testing.assert_(
                np.isfinite(np.array(out.data if hasattr(out, "data") else out)).all()
            )

    def _run_backward_and_check_grads(
        self,
        case: TensorOpCase,
        out: Any,
        inputs: tuple[Any, ...],
        expected_input_grads: tuple[object | None, ...],
    ) -> None:
        out.backward()

        for input_, expected_grad in zip(inputs, expected_input_grads):
            if expected_grad is None:
                if hasattr(input_, "grad"):
                    assert input_.grad is None
                continue

            got = input_.grad
            assert got is not None, f"Expected grad for {case.name}, but got None."
            self.assert_tensor_allclose(
                got, expected_grad, rtol=case.rtol, atol=case.atol
            )
            np.testing.assert_(np.isfinite(np.array(got)).all())


class TensorOpTorchBase(TensorOpBase):
    def _run_tensor_op_case(
        self,
        case: TensorOpTorchCase,
        device: str = "cpu",
    ) -> None:
        if case.torch_forward is None:
            return super()._run_tensor_op_case(case, device=device)

        inputs = self._prepare_inputs(case, case.build_inputs(self), device=device)
        out = case.forward_op(*inputs)

        torch = self.import_module("torch")
        torch_inputs = self._to_torch_inputs(inputs, torch)
        torch_out = case.torch_forward(*torch_inputs)

        self.assert_tensor_allclose(
            out,
            torch_out.detach().cpu().numpy(),
            rtol=case.rtol,
            atol=case.atol,
        )

        if case.check_finite:
            np.testing.assert_(
                np.isfinite(np.array(out.data if hasattr(out, "data") else out)).all()
            )

        if case.expected_input_grads is None and case.compare_grad_with_torch:
            self._run_torch_backward_check(
                case, out, inputs, torch_inputs, torch_out, torch
            )
        elif case.expected_input_grads is not None:
            self._run_backward_and_check_grads(
                case, out, inputs, case.expected_input_grads
            )

    def _to_torch_inputs(self, inputs: tuple[Any, ...], torch: Any) -> tuple[Any, ...]:
        output = []
        for item in inputs:
            if hasattr(item, "data"):
                requires_grad = bool(item.requires_grad)
                data = np.array(item.data)
                if data.dtype == bool or np.issubdtype(data.dtype, np.bool_):
                    dtype = torch.bool
                elif np.issubdtype(data.dtype, np.integer):
                    dtype = torch.int64
                elif np.issubdtype(data.dtype, np.floating):
                    dtype = torch.float32 if data.dtype == np.float32 else torch.float64
                elif np.issubdtype(data.dtype, np.complexfloating):
                    dtype = torch.complex128
                else:
                    dtype = torch.float64

                output.append(
                    torch.tensor(
                        data,
                        dtype=dtype,
                        requires_grad=requires_grad,
                    )
                )
            else:
                data = np.array(item)
                dtype = (
                    torch.float32
                    if data.dtype == np.float32
                    else torch.float64 if data.dtype == np.float64 else torch.float64
                )
                output.append(torch.tensor(data, dtype=dtype, requires_grad=False))

        return tuple(output)

    def _run_torch_backward_check(
        self,
        case: TensorOpTorchCase,
        out: Any,
        inputs: tuple[Any, ...],
        torch_inputs: tuple[Any, ...],
        torch_out: Any,
        torch: Any,
    ) -> None:
        _ = torch
        torch_target = torch_out.sum() if torch_out.ndim else torch_out
        torch_target.backward()

        out.backward()

        for input_, torch_input in zip(inputs, torch_inputs):
            torch_grad = None if not torch_input.requires_grad else torch_input.grad
            if torch_grad is None:
                if hasattr(input_, "grad"):
                    assert input_.grad is None
                continue

            assert input_.grad is not None
            self.assert_tensor_allclose(
                input_.grad,
                torch_grad.detach().cpu().numpy(),
                rtol=case.rtol,
                atol=case.atol,
            )
            np.testing.assert_(np.isfinite(np.array(input_.grad)).all())


class ModuleImportBase(LucidBaseCase):
    modules: tuple[str, ...] = ()

    def test_import_all_declared_modules(self) -> None:
        assert len(self.modules) > 0
        for module_name in self.modules:
            imported = self.import_module(module_name)
            assert imported is not None


class TensorFactoryBase(LucidBaseCase):
    def make_scalar_tensor(self, value: float | int) -> Any:
        return self.tensor(value)

    def make_matrix_tensor(self, rows: int, cols: int) -> Any:
        data = np.arange(rows * cols).reshape(rows, cols)
        return self.tensor(data)
