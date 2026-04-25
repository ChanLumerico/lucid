import importlib

from dataclasses import dataclass, field

from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

import pytest

import lucid

from lucid.test.parity import tolerances as _tol

_Array = np.ndarray


@dataclass(frozen=True)
class TensorInput:
    array: _Array
    requires_grad: bool = False
    dtype_override: Any | None = None


@dataclass(frozen=True)
class ScalarInput:
    value: Any


Input = TensorInput | ScalarInput | Any

BuildInputsFn = Callable[[int], Sequence[Input]]

OpFn = Callable[..., Any]


@dataclass(frozen=True)
class ParityCase:
    name: str
    build_inputs: BuildInputsFn
    lucid_fn: OpFn
    torch_fn: OpFn
    tol_class: str = "elementwise_f64"
    rtol: float | None = None
    atol: float | None = None
    tolerance_reason: str | None = None
    check_backward: bool = True
    backward_target: Callable[[Any], Any] | None = None
    output_index: int | None = None
    compare_grad: bool = True
    gradcheck: bool = False
    check_finite: bool = True
    devices: tuple[str, ...] = ("cpu",)
    xfail: str | None = None
    seed: int = 0

    def resolved_tol(self) -> _tol.TolPair:
        if self.tol_class == "custom":
            if self.rtol is None or self.atol is None:
                raise ValueError(
                    f"{self.name}: tol_class='custom' requires explicit rtol/atol."
                )
            if not self.tolerance_reason:
                raise ValueError(
                    f"{self.name}: tol_class='custom' requires a tolerance_reason."
                )
            return (self.rtol, self.atol)
        if self.rtol is not None or self.atol is not None:
            raise ValueError(
                f"{self.name}: rtol/atol may only be set when tol_class='custom'."
            )
        return _tol.tol_for(self.tol_class)

    def resolved_grad_tol(self) -> _tol.TolPair:
        rtol, atol = self.resolved_tol()
        if self.tol_class == "custom":
            return (rtol * _tol.GRAD_MULTIPLIER, atol * _tol.GRAD_MULTIPLIER)
        return _tol.grad_tol_for(self.tol_class)


def _import_torch() -> Any:
    return importlib.import_module("torch")


def _np_to_torch_dtype(torch: Any, array: _Array) -> Any:
    if array.dtype == bool or np.issubdtype(array.dtype, np.bool_):
        return torch.bool
    if np.issubdtype(array.dtype, np.integer):
        return torch.int64
    if array.dtype == np.float32:
        return torch.float32
    if array.dtype == np.float64:
        return torch.float64
    if np.issubdtype(array.dtype, np.complexfloating):
        return torch.complex128
    return torch.float64


def _wrap_lucid(inputs: Sequence[Input]) -> list[Any]:
    wrapped: list[Any] = []
    for item in inputs:
        if isinstance(item, TensorInput):
            kwargs: dict[str, Any] = {"requires_grad": item.requires_grad}
            if item.dtype_override is not None:
                kwargs["dtype"] = item.dtype_override
            t = lucid.tensor(item.array.copy(), **kwargs)
            wrapped.append(t)
        elif isinstance(item, ScalarInput):
            wrapped.append(item.value)
        else:
            wrapped.append(item)
    return wrapped


def _wrap_torch(torch: Any, inputs: Sequence[Input]) -> list[Any]:
    wrapped: list[Any] = []
    for item in inputs:
        if isinstance(item, TensorInput):
            t = torch.tensor(
                item.array.copy(),
                dtype=_np_to_torch_dtype(torch, item.array),
                requires_grad=item.requires_grad,
            )
            wrapped.append(t)
        elif isinstance(item, ScalarInput):
            wrapped.append(item.value)
        else:
            wrapped.append(item)
    return wrapped


def _to_numpy(value: Any) -> _Array:
    if value is None:
        raise ValueError("cannot convert None to numpy array")
    try:
        torch = _import_torch()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    if hasattr(value, "data"):
        return np.asarray(value.data)
    return np.asarray(value)


def _select_output(value: Any, index: int | None) -> Any:
    if index is None:
        return value
    return value[index]


def _default_backward_target(out: Any) -> Any:
    if hasattr(out, "ndim") and out.ndim == 0:
        return out
    return out.sum()


def _gpu_supported() -> bool:
    try:
        torch = _import_torch()
    except Exception:
        return False
    if not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        or torch.cuda.is_available()
    ):
        return False
    try:
        import lucid._backend.metal as metal

        mx = getattr(metal, "mx", None)
        if mx is None or not hasattr(mx, "metal"):
            return False
        return bool(mx.metal.is_available())
    except Exception:
        return False


def _maybe_xfail(case: ParityCase) -> None:
    if case.xfail:
        pytest.xfail(case.xfail)


def _apply_backward(out: Any, target_fn: Callable[[Any], Any] | None) -> None:
    target = (target_fn or _default_backward_target)(out)
    target.backward()


def _assert_allclose(
    name: str, got: _Array, expected: _Array, rtol: float, atol: float
) -> None:
    try:
        np.testing.assert_allclose(got, expected, rtol=rtol, atol=atol)
    except AssertionError as err:
        raise AssertionError(f"[{name}] {err}") from None


def run_parity_case(case: ParityCase) -> None:
    _maybe_xfail(case)
    torch = _import_torch()
    fwd_rtol, fwd_atol = case.resolved_tol()
    grad_rtol, grad_atol = case.resolved_grad_tol()
    for device in case.devices:
        if device == "gpu" and (not _gpu_supported()):
            pytest.skip(f"[{case.name}] gpu device unavailable")
            continue
        raw_inputs = list(case.build_inputs(case.seed))
        lucid_inputs = _wrap_lucid(raw_inputs)
        torch_inputs = _wrap_torch(torch, raw_inputs)
        if device == "gpu":
            lucid_inputs = [
                x.to("gpu") if hasattr(x, "to") and hasattr(x, "device") else x
                for x in lucid_inputs
            ]
        lucid_out = case.lucid_fn(*lucid_inputs)
        torch_out = case.torch_fn(*torch_inputs)
        lucid_cmp = _select_output(lucid_out, case.output_index)
        torch_cmp = _select_output(torch_out, case.output_index)
        _assert_allclose(
            f"{case.name}[{device}][forward]",
            _to_numpy(lucid_cmp),
            _to_numpy(torch_cmp),
            fwd_rtol,
            fwd_atol,
        )
        if case.check_finite:
            finite = np.isfinite(_to_numpy(lucid_cmp)).all()
            assert finite, f"[{case.name}][{device}] forward output not finite"
        if not case.check_backward:
            continue
        _apply_backward(lucid_cmp, case.backward_target)
        _apply_backward(torch_cmp, case.backward_target)
        if case.compare_grad:
            _compare_input_grads(
                case.name,
                device,
                raw_inputs,
                lucid_inputs,
                torch_inputs,
                grad_rtol,
                grad_atol,
            )
        if case.gradcheck:
            from lucid.test.parity.gradcheck import assert_gradcheck

            assert_gradcheck(
                case.lucid_fn,
                raw_inputs,
                rtol=max(grad_rtol, 0.0005),
                atol=max(grad_atol, 5e-05),
            )


def _compare_input_grads(
    case_name: str,
    device: str,
    raw_inputs: Sequence[Input],
    lucid_inputs: Sequence[Any],
    torch_inputs: Sequence[Any],
    rtol: float,
    atol: float,
) -> None:
    for idx, (raw, lucid_arg, torch_arg) in enumerate(
        zip(raw_inputs, lucid_inputs, torch_inputs)
    ):
        if not isinstance(raw, TensorInput):
            continue
        if not raw.requires_grad:
            continue
        lucid_grad = getattr(lucid_arg, "grad", None)
        torch_grad = getattr(torch_arg, "grad", None)
        if torch_grad is None and lucid_grad is None:
            continue
        assert (
            lucid_grad is not None
        ), f"[{case_name}][{device}] lucid input[{idx}] grad is None but torch produced a grad"
        assert (
            torch_grad is not None
        ), f"[{case_name}][{device}] torch input[{idx}] grad is None but lucid produced a grad"
        _assert_allclose(
            f"{case_name}[{device}][grad#{idx}]",
            _to_numpy(lucid_grad),
            _to_numpy(torch_grad),
            rtol,
            atol,
        )
        assert np.isfinite(
            _to_numpy(lucid_grad)
        ).all(), f"[{case_name}][{device}] lucid grad[{idx}] not finite"


@dataclass(frozen=True)
class OptimTrajectoryCase:
    name: str
    build_models: Callable[[int], tuple[Any, Any]]
    "seed -> (lucid_module, torch_module) with synced initial parameters."
    build_step_inputs: Callable[[int, int], tuple[Any, Any, Any, Any]]
    "(step, seed) -> (lucid_x, lucid_y, torch_x, torch_y) for one forward."
    lucid_optim: Callable[[Iterable[Any]], Any]
    torch_optim: Callable[[Iterable[Any]], Any]
    lucid_loss: Callable[[Any, Any], Any]
    torch_loss: Callable[[Any, Any], Any]
    steps: int = 16
    tol_class_param: str = "optim_param"
    tol_class_grad: str = "optim_param"
    xfail: str | None = None
    seed: int = 0


def _sync_torch_from_lucid(lucid_model: Any, torch_model: Any) -> None:
    import torch

    lucid_state = lucid_model.state_dict()
    torch_state = torch_model.state_dict()
    with torch.no_grad():
        for name, torch_val in torch_state.items():
            if name not in lucid_state:
                continue
            src = lucid_state[name]
            arr = np.asarray(src.data if hasattr(src, "data") else src)
            torch_val.copy_(torch.as_tensor(arr, dtype=torch_val.dtype))


def run_optim_trajectory_case(case: OptimTrajectoryCase) -> None:
    if case.xfail:
        pytest.xfail(case.xfail)
    lucid.random.seed(case.seed)
    import torch

    torch.manual_seed(case.seed)
    lucid_model, torch_model = case.build_models(case.seed)
    _sync_torch_from_lucid(lucid_model, torch_model)
    lucid_opt = case.lucid_optim(lucid_model.parameters())
    torch_opt = case.torch_optim(torch_model.parameters())
    param_rtol, param_atol = _tol.tol_for(case.tol_class_param)
    grad_rtol, grad_atol = _tol.tol_for(case.tol_class_grad)
    for step in range(case.steps):
        x_l, y_l, x_t, y_t = case.build_step_inputs(step, case.seed)
        lucid_opt.zero_grad()
        torch_opt.zero_grad()
        l_out = lucid_model(x_l)
        t_out = torch_model(x_t)
        l_loss = case.lucid_loss(l_out, y_l)
        t_loss = case.torch_loss(t_out, y_t)
        _assert_allclose(
            f"{case.name}[step{step}][loss]",
            _to_numpy(l_loss),
            _to_numpy(t_loss),
            param_rtol,
            param_atol,
        )
        l_loss.backward()
        t_loss.backward()
        _compare_parameter_grads(
            case.name, step, lucid_model, torch_model, grad_rtol, grad_atol
        )
        lucid_opt.step()
        torch_opt.step()
        _compare_parameters(
            case.name, step, lucid_model, torch_model, param_rtol, param_atol
        )


def _zip_params(lucid_model: Any, torch_model: Any):
    torch_named = dict(torch_model.named_parameters())
    lucid_state = lucid_model.state_dict()
    torch_lookup = {name: torch_named[name] for name in torch_named}
    lucid_params = list(lucid_model.parameters())
    lucid_param_names = [
        name
        for (name, val) in lucid_state.items()
        if any((val is p for p in lucid_params))
    ]
    for name, lparam in zip(lucid_param_names, lucid_params):
        tparam = torch_lookup.get(name)
        if tparam is None:
            continue
        yield (name, lparam, tparam)


def _compare_parameter_grads(
    case_name: str,
    step: int,
    lucid_model: Any,
    torch_model: Any,
    rtol: float,
    atol: float,
) -> None:
    for name, lparam, tparam in _zip_params(lucid_model, torch_model):
        if lparam.grad is None and tparam.grad is None:
            continue
        if lparam.grad is None or tparam.grad is None:
            raise AssertionError(
                f"[{case_name}][step{step}] grad presence mismatch on {name}: lucid={lparam.grad is not None}, torch={tparam.grad is not None}"
            )
        tg = _to_numpy(tparam.grad)
        if np.max(np.abs(tg)) < _tol.OPTIM_PARAM_GRAD_SKIP:
            continue
        _assert_allclose(
            f"{case_name}[step{step}][grad:{name}]",
            _to_numpy(lparam.grad),
            tg,
            rtol,
            atol,
        )


def _compare_parameters(
    case_name: str,
    step: int,
    lucid_model: Any,
    torch_model: Any,
    rtol: float,
    atol: float,
) -> None:
    for name, lparam, tparam in _zip_params(lucid_model, torch_model):
        _assert_allclose(
            f"{case_name}[step{step}][param:{name}]",
            _to_numpy(lparam.data),
            _to_numpy(tparam.data),
            rtol,
            atol,
        )


@dataclass(frozen=True)
class SchedulerTrajectoryCase:
    name: str
    build_schedulers: Callable[[float], tuple[Any, Any, Any, Any]]
    "lr -> (lucid_opt, lucid_sched, torch_opt, torch_sched)."
    initial_lr: float = 0.01
    steps: int = 32
    per_step_inputs: Callable[[int], Mapping[str, Any] | None] = field(
        default=lambda s: None
    )
    tol_class: str = "scheduler_lr"
    xfail: str | None = None


def run_scheduler_case(case: SchedulerTrajectoryCase) -> None:
    if case.xfail:
        pytest.xfail(case.xfail)
    rtol, atol = _tol.tol_for(case.tol_class)
    lucid_opt, lucid_sched, torch_opt, torch_sched = case.build_schedulers(
        case.initial_lr
    )
    for step in range(case.steps):
        payload = case.per_step_inputs(step) or {}
        if payload:
            lucid_sched.step(**payload)
            torch_sched.step(**payload)
        else:
            lucid_sched.step()
            torch_sched.step()
        lucid_lrs = [pg["lr"] for pg in lucid_opt.param_groups]
        torch_lrs = [pg["lr"] for pg in torch_opt.param_groups]
        _assert_allclose(
            f"{case.name}[step{step}]",
            np.array(lucid_lrs, dtype=np.float64),
            np.array(torch_lrs, dtype=np.float64),
            rtol,
            atol,
        )


def _mlx_gpu_available() -> bool:
    try:
        import lucid._backend.metal as metal

        mx = getattr(metal, "mx", None)
        if mx is None or not hasattr(mx, "metal"):
            return False
        return bool(mx.metal.is_available())
    except Exception:
        return False


@dataclass(frozen=True)
class DeviceParityCase:
    name: str
    build_inputs: BuildInputsFn
    lucid_fn: OpFn
    tol_class: str = "elementwise_f32"
    rtol: float | None = None
    atol: float | None = None
    tolerance_reason: str | None = None
    check_backward: bool = True
    backward_target: Callable[[Any], Any] | None = None
    output_index: int | None = None
    xfail: str | None = None
    seed: int = 0


def _to_gpu_inputs(lucid_inputs: list[Any]) -> list[Any]:
    out = []
    for t in lucid_inputs:
        if hasattr(t, "to") and hasattr(t, "device"):
            out.append(t.to("gpu"))
        else:
            out.append(t)
    return out


def run_device_parity_case(case: DeviceParityCase) -> None:
    if case.xfail:
        pytest.xfail(case.xfail)
    if not _mlx_gpu_available():
        pytest.skip(f"[{case.name}] MLX GPU unavailable")
    if case.tol_class == "custom":
        if case.rtol is None or case.atol is None or (not case.tolerance_reason):
            raise ValueError(
                f"{case.name}: tol_class='custom' requires rtol/atol + tolerance_reason"
            )
        fwd_rtol, fwd_atol = (case.rtol, case.atol)
    else:
        fwd_rtol, fwd_atol = _tol.tol_for(case.tol_class)
    grad_rtol = fwd_rtol * _tol.GRAD_MULTIPLIER
    grad_atol = fwd_atol * _tol.GRAD_MULTIPLIER
    raw_inputs = list(case.build_inputs(case.seed))
    cpu_inputs = _wrap_lucid(raw_inputs)
    cpu_out = case.lucid_fn(*cpu_inputs)
    cpu_cmp = _select_output(cpu_out, case.output_index)
    gpu_inputs = _to_gpu_inputs(_wrap_lucid(raw_inputs))
    gpu_out = case.lucid_fn(*gpu_inputs)
    gpu_cmp = _select_output(gpu_out, case.output_index)
    _assert_allclose(
        f"{case.name}[forward]",
        _to_numpy(cpu_cmp),
        _to_numpy(gpu_cmp),
        fwd_rtol,
        fwd_atol,
    )
    if not case.check_backward:
        return
    _apply_backward(cpu_cmp, case.backward_target)
    _apply_backward(gpu_cmp, case.backward_target)
    for idx, (raw, cpu_arg, gpu_arg) in enumerate(
        zip(raw_inputs, cpu_inputs, gpu_inputs)
    ):
        if not isinstance(raw, TensorInput) or not raw.requires_grad:
            continue
        cpu_grad = getattr(cpu_arg, "grad", None)
        gpu_grad = getattr(gpu_arg, "grad", None)
        if cpu_grad is None and gpu_grad is None:
            continue
        assert cpu_grad is not None, f"[{case.name}] cpu grad[{idx}] is None"
        assert gpu_grad is not None, f"[{case.name}] gpu grad[{idx}] is None"
        _assert_allclose(
            f"{case.name}[grad#{idx}]",
            _to_numpy(cpu_grad),
            _to_numpy(gpu_grad),
            grad_rtol,
            grad_atol,
        )


@dataclass(frozen=True)
class ModuleParityCase:
    name: str
    build_modules: Callable[[int], tuple[Any, Any]]
    build_inputs: BuildInputsFn
    tol_class: str = "norm_f32"
    rtol: float | None = None
    atol: float | None = None
    tolerance_reason: str | None = None
    train_mode: bool = True
    backward_target: Callable[[Any], Any] | None = None
    output_index: int | None = None
    xfail: str | None = None
    seed: int = 0


def _set_train(module: Any, train: bool) -> None:
    if hasattr(module, "train"):
        module.train(train)
    if hasattr(module, "eval") and (not train):
        module.eval()


def run_module_parity_case(case: ModuleParityCase) -> None:
    if case.xfail:
        pytest.xfail(case.xfail)
    if case.tol_class == "custom":
        if case.rtol is None or case.atol is None or (not case.tolerance_reason):
            raise ValueError(
                f"{case.name}: tol_class='custom' requires rtol/atol + tolerance_reason"
            )
        fwd_rtol, fwd_atol = (case.rtol, case.atol)
        grad_rtol = fwd_rtol * _tol.GRAD_MULTIPLIER
        grad_atol = fwd_atol * _tol.GRAD_MULTIPLIER
    else:
        fwd_rtol, fwd_atol = _tol.tol_for(case.tol_class)
        grad_rtol, grad_atol = _tol.grad_tol_for(case.tol_class)
    torch = _import_torch()
    lucid_mod, torch_mod = case.build_modules(case.seed)
    _sync_torch_from_lucid(lucid_mod, torch_mod)
    _set_train(lucid_mod, case.train_mode)
    _set_train(torch_mod, case.train_mode)
    raw_inputs = list(case.build_inputs(case.seed))
    lucid_inputs = _wrap_lucid(raw_inputs)
    torch_inputs = _wrap_torch(torch, raw_inputs)
    lucid_out = lucid_mod(*lucid_inputs)
    torch_out = torch_mod(*torch_inputs)
    lucid_cmp = _select_output(lucid_out, case.output_index)
    torch_cmp = _select_output(torch_out, case.output_index)
    _assert_allclose(
        f"{case.name}[forward]",
        _to_numpy(lucid_cmp),
        _to_numpy(torch_cmp),
        fwd_rtol,
        fwd_atol,
    )
    _apply_backward(lucid_cmp, case.backward_target)
    _apply_backward(torch_cmp, case.backward_target)
    _compare_input_grads(
        case.name, "cpu", raw_inputs, lucid_inputs, torch_inputs, grad_rtol, grad_atol
    )
    _compare_parameter_grads(case.name, 0, lucid_mod, torch_mod, grad_rtol, grad_atol)
