from typing import Final

TolPair = tuple[float, float]

DEFAULT_TOL: Final[dict[str, TolPair]] = {
    "elementwise_f32": (1e-06, 1e-07),
    "elementwise_f64": (1e-12, 1e-13),
    "reduction_f32": (5e-06, 1e-06),
    "reduction_f64": (1e-10, 1e-12),
    "matmul_f32": (1e-05, 1e-06),
    "matmul_f64": (1e-10, 1e-12),
    "linalg_f32": (0.0001, 1e-05),
    "linalg_f64": (1e-08, 1e-10),
    "norm_f32": (5e-06, 5e-07),
    "softmax_f32": (1e-06, 1e-07),
    "attention_f32": (5e-06, 1e-06),
    "loss_f32": (5e-06, 1e-06),
    "optim_param": (0.0005, 1e-05),
    "scheduler_lr": (1e-07, 1e-09),
    "shape_exact": (0.0, 0.0),
    "integer_exact": (0.0, 0.0),
    "boolean_exact": (0.0, 0.0),
}

GRAD_MULTIPLIER: Final[float] = 5.0

OPTIM_PARAM_GRAD_SKIP: Final[float] = 1e-07


def tol_for(tol_class: str) -> TolPair:
    if tol_class == "custom":
        raise ValueError(
            "tol_class='custom' requires explicit rtol/atol on the case; do not call tol_for() for custom cases."
        )
    try:
        return DEFAULT_TOL[tol_class]
    except KeyError as err:
        raise KeyError(
            f"Unknown tol_class {tol_class!r}. Valid keys: {sorted(DEFAULT_TOL)}. Add a new class here rather than sprinkling ad-hoc tolerances across test files."
        ) from err


def grad_tol_for(tol_class: str) -> TolPair:
    rtol, atol = tol_for(tol_class)
    return (rtol * GRAD_MULTIPLIER, atol * GRAD_MULTIPLIER)
