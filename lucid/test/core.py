import importlib
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

import numpy as np
import pytest

from lucid.types import _NumPyArray

__all__ = [
    "LucidBaseCase",
    "TensorOpCase",
    "TensorOpTorchCase",
    "ModuleTorchCase",
    "ModuleTorchBase",
    "TensorOpBase",
    "TensorOpWithTorchBase",
    "ModuleImportBase",
    "TensorFactoryBase",
]


class LucidBaseCase:
    random_seed: int | None = 42

    @staticmethod
    def _to_numpy(value: Any) -> _NumPyArray:
        try:
            import torch

            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
        except ModuleNotFoundError:
            pass

        if hasattr(value, "data"):
            return np.array(value.data)
        return np.array(value)

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
        lhs_arr = self._to_numpy(lhs)
        rhs_arr = self._to_numpy(rhs)
        np.testing.assert_array_equal(lhs_arr, rhs_arr)

    def assert_tensor_allclose(
        self,
        lhs: object,
        rhs: object,
        *,
        rtol: float = 1e-7,
        atol: float = 1e-8,
    ) -> None:
        lhs_arr = self._to_numpy(lhs)
        rhs_arr = self._to_numpy(rhs)
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


@dataclass(frozen=True)
class ModuleTorchCase:
    name: str
    build_inputs: Callable[["LucidBaseCase"], tuple[Any, ...]]
    module_ctor: Callable[..., Any]
    torch_module_ctor: Callable[..., Any] | None
    module_kwargs: dict[str, Any] = field(default_factory=dict)
    torch_module_kwargs: dict[str, Any] = field(default_factory=dict)
    expected_forward: object | None = None
    expected_input_grads: tuple[object | None, ...] | None = None
    requires_grad: tuple[bool, ...] = ()
    module_train_modes: tuple[bool, ...] = (True,)
    compare_grad_with_torch: bool = True
    compare_state_dict: bool = True
    check_finite: bool = True
    rtol: float = 1e-7
    atol: float = 1e-8


class TensorOpBase(LucidBaseCase):
    _collect_nested_test_classes: ClassVar[bool] = False

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if not cls.__name__.startswith("Test"):
            return

        qualname = cls.__qualname__
        if "." in qualname:
            return

        if not getattr(cls, "_collect_nested_test_classes", False):
            return

        if cls.__dict__.get("__test__") is not False:
            return

        outer_suffix = cls.__name__[4:]
        module = importlib.import_module(cls.__module__)
        for nested_name, nested_cls in vars(cls).items():
            if not isinstance(nested_cls, type):
                continue
            if not nested_name.startswith("Test"):
                continue

            nested_suffix = nested_name[4:]
            if nested_suffix.startswith(outer_suffix):
                nested_suffix = nested_suffix[len(outer_suffix) :]
            if not nested_suffix:
                nested_suffix = "Case"

            exported_name = f"Test{outer_suffix}{nested_suffix}"
            existing = getattr(module, exported_name, None)
            if existing is not None and existing is not nested_cls:
                continue
            setattr(module, exported_name, nested_cls)

    def tensor_op_devices(self) -> tuple[str, ...]:
        return ("cpu",)

    def _is_gpu_device_supported(self) -> bool:
        try:
            torch = importlib.import_module("torch")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return True
            if torch.cuda.is_available():
                return True
        except Exception:
            return False

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

    @staticmethod
    def _torch_device(device: str) -> str:
        return "mps" if device == "gpu" else device

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

    def test_tensor_op_forward_backward(self) -> None:
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


class TensorOpWithTorchBase(TensorOpBase):
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

    def _to_torch_inputs(
        self, inputs: tuple[Any, ...], torch: Any, device: str = "cpu"
    ) -> tuple[Any, ...]:
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
                output.append(
                    torch.tensor(
                        data,
                        dtype=dtype,
                        requires_grad=False,
                        device=device,
                    )
                )

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
            if torch_input is None:
                continue
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


class ModuleTorchBase(LucidBaseCase):
    _collect_nested_test_classes: ClassVar[bool] = False

    @classmethod
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if not cls.__name__.startswith("Test"):
            return

        qualname = cls.__qualname__
        if "." not in qualname:
            return

        if not getattr(cls, "_collect_nested_test_classes", False):
            return

        if cls.__dict__.get("__test__") is not False:
            return

        outer_suffix = cls.__name__[4:]
        module = importlib.import_module(cls.__module__)
        for nested_name, nested_cls in vars(cls).items():
            if not isinstance(nested_cls, type):
                continue
            if not nested_name.startswith("Test"):
                continue

            nested_suffix = nested_name[4:]
            if nested_suffix.startswith(outer_suffix):
                nested_suffix = nested_suffix[len(outer_suffix) :]
            if not nested_suffix:
                nested_suffix = "Case"

            exported_name = f"Test{outer_suffix}{nested_suffix}"
            existing = getattr(module, exported_name, None)
            if existing is not None and existing is not nested_cls:
                continue
            setattr(module, exported_name, nested_cls)

    def tensor_op_devices(self) -> tuple[str, ...]:
        return ("cpu", "gpu")

    def _is_gpu_device_supported(self) -> bool:
        try:
            torch = importlib.import_module("torch")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                pass
            elif torch.cuda.is_available():
                pass
            else:
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

    @staticmethod
    def _torch_device(device: str) -> str:
        return "mps" if device == "gpu" else device

    def module_op_cases(self) -> tuple[ModuleTorchCase, ...]:
        return ()

    def _prepare_inputs(
        self,
        case: TensorOpCase,
        inputs: tuple[Any, ...],
        device: str = "cpu",
    ) -> tuple[Any, ...]:
        if case.requires_grad and len(case.requires_grad) != len(inputs):
            raise AssertionError(
                f"requires_grad length {len(case.requires_grad)} does not match inputs "
                f"{len(inputs)} for case '{case.name}'."
            )

        flags = (
            case.requires_grad
            if case.requires_grad
            else tuple(hasattr(item, "requires_grad") for item in inputs)
        )

        prepared: list[Any] = []
        for item, need_grad in zip(inputs, flags):
            if not hasattr(item, "requires_grad"):
                prepared.append(item.to(device) if hasattr(item, "to") else item)
                continue

            item.grad = None
            item.requires_grad = bool(need_grad)
            prepared.append(item.to(device))

        return tuple(prepared)

    def _to_torch_inputs(
        self, inputs: tuple[Any, ...], torch: Any, device: str = "cpu"
    ) -> tuple[Any, ...]:
        output: list[Any] = []
        for item in inputs:
            if not hasattr(item, "data"):
                data = np.array(item)
                dtype = torch.float64
                if np.issubdtype(data.dtype, np.float32):
                    dtype = torch.float32
                elif np.issubdtype(data.dtype, np.floating):
                    dtype = torch.float64
                elif np.issubdtype(data.dtype, np.integer):
                    dtype = torch.int64
                elif np.issubdtype(data.dtype, np.bool_):
                    dtype = torch.bool
                elif np.issubdtype(data.dtype, np.complexfloating):
                    dtype = torch.complex128

                output.append(
                    torch.tensor(
                        data,
                        dtype=dtype,
                        requires_grad=False,
                        device=device,
                    )
                )
                continue

            data = np.array(item.data)
            requires_grad = bool(item.requires_grad)
            if np.issubdtype(data.dtype, np.bool_):
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
                    data, dtype=dtype, requires_grad=requires_grad, device=device
                )
            )

        return tuple(output)

    def test_module_op_forward_backward(self) -> None:
        cases = self.module_op_cases()
        assert len(cases) > 0, f"No ModuleTorchCase defined in {type(self).__name__}."

        for case in cases:
            for device in self.tensor_op_devices():
                if not self._is_device_supported(device):
                    continue
                self._run_module_op_case(case, device=device)

    def _run_module_op_case(self, case: ModuleTorchCase, device: str = "cpu") -> None:
        torch = self.import_module("torch")
        build_inputs = case.build_inputs(self)
        requires_grad = self._normalize_input_grad_flags(
            case.requires_grad, len(build_inputs)
        )

        inputs = self._prepare_inputs(
            TensorOpCase(
                name=case.name,
                build_inputs=lambda _case: build_inputs,
                forward_op=lambda *_: None,
                expected_forward=0.0,
                requires_grad=requires_grad,
            ),
            build_inputs,
            device=device,
        )
        torch_device = self._torch_device(device)
        torch_inputs_base = self._to_torch_inputs(inputs, torch, device=torch_device)

        for mode in case.module_train_modes:
            module = case.module_ctor(**case.module_kwargs)
            module.to(device)
            module.train(mode)

            if case.torch_module_ctor is None:
                if case.expected_forward is None:
                    raise AssertionError(
                        f"Module case '{case.name}' requires expected_forward when "
                        f"no torch module is provided."
                    )

                module_out = module(*inputs)
                self._assert_nested_allclose(
                    module_out,
                    case.expected_forward,
                    rtol=case.rtol,
                    atol=case.atol,
                )
                if case.check_finite:
                    self._assert_nested_finite(module_out)
                continue

            torch_module = case.torch_module_ctor(**case.torch_module_kwargs)
            torch_inputs = tuple(torch_inputs_base)
            self._set_torch_module_dtype(torch_inputs, torch_module)

            self._sync_module_state(module, torch_module)
            torch_module.to(torch_device)
            torch_module.train(mode)

            torch_out = torch_module(*torch_inputs)
            module_out = module(*inputs)

            if case.expected_forward is None:
                self._assert_nested_allclose(
                    module_out,
                    torch_out,
                    rtol=case.rtol,
                    atol=case.atol,
                )
            else:
                self._assert_nested_allclose(
                    module_out,
                    case.expected_forward,
                    rtol=case.rtol,
                    atol=case.atol,
                )

            if case.check_finite:
                self._assert_nested_finite(module_out)

            if case.expected_input_grads is None and case.compare_grad_with_torch:
                self._run_module_backward_and_grad_check(
                    module=module,
                    torch_module=torch_module,
                    inputs=inputs,
                    torch_inputs=torch_inputs,
                    output=module_out,
                    torch_output=torch_out,
                    rtol=case.rtol,
                    atol=case.atol,
                )
            elif case.expected_input_grads is not None:
                self._run_module_backward_and_expected_grad_check(
                    inputs=inputs,
                    expected_input_grads=case.expected_input_grads,
                    rtol=case.rtol,
                    atol=case.atol,
                )

            if case.compare_state_dict:
                self._assert_state_dict_match(module, torch_module)

    def _normalize_input_grad_flags(
        self, flags: tuple[bool, ...], expected_len: int
    ) -> tuple[bool, ...]:
        if not flags:
            return tuple(False for _ in range(expected_len))
        if len(flags) == 1 and expected_len > 1:
            return tuple(bool(flags[0]) for _ in range(expected_len))
        if len(flags) != expected_len:
            raise AssertionError(
                "Invalid requires_grad length "
                f"{len(flags)} (expected {expected_len}) for module case."
            )
        return tuple(bool(v) for v in flags)

    def _set_torch_module_dtype(
        self, torch_inputs: tuple[Any, ...], torch_module: Any
    ) -> None:
        import torch

        target_dtype = None
        for item in torch_inputs:
            if not hasattr(item, "dtype"):
                continue

            if item.dtype.is_floating_point or item.dtype.is_complex:
                target_dtype = item.dtype
                break

        if target_dtype is None:
            target_dtype = torch.float64

        torch_module.to(dtype=target_dtype)

    def _sync_module_state(self, module: Any, torch_module: Any) -> None:
        lucid_params = tuple(module.parameters())
        torch_params = tuple(torch_module.parameters())

        if len(lucid_params) != len(torch_params):
            raise AssertionError(
                "Parameter count mismatch between lucid and torch modules: "
                f"lucid={len(lucid_params)} torch={len(torch_params)}"
            )

        for lucid_param, torch_param in zip(lucid_params, torch_params):
            torch_param.data = self._to_torch_param_like(torch_param, lucid_param)

        lucid_buffers = tuple(module.buffers())
        torch_buffers = tuple(torch_module.buffers())

        if len(lucid_buffers) != len(torch_buffers):
            raise AssertionError(
                "Buffer count mismatch between lucid and torch modules: "
                f"lucid={len(lucid_buffers)} torch={len(torch_buffers)}"
            )

        for lucid_buffer, torch_buffer in zip(lucid_buffers, torch_buffers):
            if lucid_buffer is None and torch_buffer is None:
                continue
            if lucid_buffer is None or torch_buffer is None:
                raise AssertionError("State mismatch between lucid and torch buffers.")

            torch_buffer.data = self._to_torch_param_like(torch_buffer, lucid_buffer)

    def _to_torch_param_like(self, torch_target: Any, source: Any) -> Any:
        import torch

        source_data = np.array(source.data if hasattr(source, "data") else source)
        return torch.as_tensor(
            source_data,
            device=torch_target.device,
            dtype=torch_target.dtype,
        )

    def _run_module_backward_and_expected_grad_check(
        self,
        inputs: tuple[Any, ...],
        expected_input_grads: tuple[object | None, ...],
        *,
        rtol: float = 1e-7,
        atol: float = 1e-8,
    ) -> None:
        if len(inputs) != len(expected_input_grads):
            raise AssertionError(
                "Expected input grad length mismatch with number of module inputs."
            )
        for input_, expected_grad in zip(inputs, expected_input_grads):
            if expected_grad is None:
                if hasattr(input_, "grad"):
                    assert input_.grad is None
                continue

            if input_.grad is None:
                raise AssertionError("Expected input grad, got None.")

            self.assert_tensor_allclose(
                input_.grad, expected_grad, rtol=rtol, atol=atol
            )

    def _run_module_backward_and_grad_check(
        self,
        module: Any,
        torch_module: Any,
        inputs: tuple[Any, ...],
        torch_inputs: tuple[Any, ...],
        output: Any,
        torch_output: Any,
        *,
        rtol: float = 1e-7,
        atol: float = 1e-8,
    ) -> None:
        output = self._to_scalar_output(output)
        torch_output = self._to_scalar_torch(torch_output)

        for input_ in inputs:
            if hasattr(input_, "grad"):
                input_.grad = None
        for param in module.parameters():
            if hasattr(param, "grad"):
                param.grad = None
        for input_ in torch_inputs:
            if hasattr(input_, "grad"):
                input_.grad = None
        for param in torch_module.parameters():
            if hasattr(param, "grad"):
                param.grad = None

        output.backward()
        torch_output.backward()

        for input_, torch_input in zip(inputs, torch_inputs):
            if not torch_input.requires_grad:
                if hasattr(input_, "grad"):
                    assert input_.grad is None
                continue

            assert input_.grad is not None
            assert torch_input.grad is not None

            self.assert_tensor_allclose(
                input_.grad,
                torch_input.grad.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
            )
            np.testing.assert_(np.isfinite(np.array(input_.grad)).all())

        for lucid_param, torch_param in zip(
            module.parameters(), torch_module.parameters()
        ):
            assert hasattr(torch_param, "grad")
            if torch_param.grad is None:
                assert not hasattr(lucid_param, "grad") or lucid_param.grad is None
                continue

            assert lucid_param.grad is not None

            self.assert_tensor_allclose(
                lucid_param.grad,
                torch_param.grad.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
            )
            np.testing.assert_(np.isfinite(np.array(lucid_param.grad)).all())

    @staticmethod
    def _to_scalar_output(value: Any) -> Any:
        if isinstance(value, tuple):
            output = None
            for item in value:
                item_scalar = ModuleTorchBase._to_scalar_output(item)
                output = item_scalar if output is None else output + item_scalar
            return output

        if hasattr(value, "sum"):
            return value if getattr(value, "shape", None) == () else value.sum()
        return value

    @staticmethod
    def _to_scalar_torch(value: Any) -> Any:
        if isinstance(value, tuple):
            output = None
            for item in value:
                item_scalar = ModuleTorchBase._to_scalar_torch(item)
                output = item_scalar if output is None else output + item_scalar
            return output

        if hasattr(value, "dim") and hasattr(value, "sum"):
            return value if value.dim() == 0 else value.sum()
        return value

    def _assert_nested_allclose(
        self, lhs: object, rhs: object, *, rtol: float, atol: float
    ) -> None:
        if isinstance(lhs, tuple):
            if not isinstance(rhs, tuple):
                raise AssertionError("Nested output mismatch: tuple vs non-tuple.")
            if len(lhs) != len(rhs):
                raise AssertionError(
                    f"Nested output length mismatch: {len(lhs)} vs {len(rhs)}"
                )

            for lhs_item, rhs_item in zip(lhs, rhs):
                self._assert_nested_allclose(lhs_item, rhs_item, rtol=rtol, atol=atol)
            return

        if isinstance(rhs, tuple):
            raise AssertionError("Nested output mismatch: non-tuple vs tuple.")

        self.assert_tensor_allclose(lhs, rhs, rtol=rtol, atol=atol)

    def _assert_nested_finite(self, value: object) -> None:
        if isinstance(value, tuple):
            for item in value:
                self._assert_nested_finite(item)
            return

        arr = self._to_numpy(value)
        np.testing.assert_(np.isfinite(arr).all())

    def _assert_state_dict_match(self, module: Any, torch_module: Any) -> None:
        lucid_state = module.state_dict(keep_vars=True)
        torch_state = torch_module.state_dict()

        if lucid_state.keys() != torch_state.keys():
            raise AssertionError(
                "State-dict keys mismatch between lucid and torch modules."
            )

        for key in lucid_state:
            self.assert_tensor_allclose(
                lucid_state[key], torch_state[key], rtol=1e-7, atol=1e-8
            )


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
