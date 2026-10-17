"""The autocast policy table and how the engine applies it (LCD-500).

``lucid/_C/core/AmpPolicyTable.def`` is the one place an op's autocast
behaviour is decided: one row per op, a Metal column (the reference
framework's MPS list) and a CPU column (its CPU list).  An op with no row
keeps its inputs' dtype.  Before the table each primitive carried one
policy for both devices, and a composite re-applied its parts' policies:
inside ``autocast(metal, float16)`` a float32 ``x * y`` came back float16,
``log1p`` raised ``DtypeMismatch``, and the CPU computed ``exp`` and
``softmax`` in float32 where the reference keeps bfloat16.

The checks:

- the table is sorted, unique and names ops Lucid has;
- each row an engine op applies today gives the dtype the row says, for a
  float32 and a half input, under ``autocast(metal, float16)`` and
  ``autocast(cpu, bfloat16)`` — and, with ``-m parity``, the dtype the
  reference gives;
- the rows not wired yet are strict xfails naming the card that wires them;
- autocast is back on after an op returns or raises, since the engine
  turns it off inside the op.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.amp import autocast
from lucid.test._fixtures.devices import metal_available

_P = _C_engine.AmpPolicy

# Rows whose names are schemas the engine declares but does not register,
# with the engine function that runs each one.
_UNREGISTERED_SCHEMAS = {
    "cumprod": "cumprod",
    "cumsum": "cumsum",
    "dot_1d": "dot",
    "trace": "trace",
}

# Rows that name a composite rather than a schema, with its public path.
# The engine composites apply their row through ``AmpScope``; the Python
# ones are wired by LCD-500-A1.
_COMPOSITES = {
    "adaptive_avg_pool3d": "lucid.nn.functional.adaptive_avg_pool3d",
    "adaptive_max_pool3d": "lucid.nn.functional.adaptive_max_pool3d",
    "addbmm": "lucid.addbmm",
    "addcdiv": "lucid.addcdiv",
    "addcmul": "lucid.addcmul",
    "addmm": "lucid.addmm",
    "addmv": "lucid.addmv",
    "addr": "lucid.addr",
    "atan2": "lucid.atan2",
    "baddbmm": "lucid.baddbmm",
    "bmm": "lucid.bmm",
    "cdist": "lucid.cdist",
    "cosine_embedding_loss": "lucid.nn.functional.cosine_embedding_loss",
    "cosine_similarity": "lucid.nn.functional.cosine_similarity",
    "cross": "lucid.linalg.cross",
    "einsum": "lucid.einops.einsum",
    "expm1": "lucid.expm1",
    "fft": "lucid.fft.fft",
    "gaussian_nll_loss": "lucid.nn.functional.gaussian_nll_loss",
    "hinge_embedding_loss": "lucid.nn.functional.hinge_embedding_loss",
    "index_copy": "lucid.index_copy",
    "index_put": "lucid.index_put",
    "instance_norm": "lucid.nn.functional.instance_norm",
    "kl_div": "lucid.nn.functional.kl_div",
    "l1_loss": "lucid.nn.functional.l1_loss",
    "local_response_norm": "lucid.nn.functional.local_response_norm",
    "log10": "lucid.log10",
    "log1p": "lucid.log1p",
    "logsumexp": "lucid.logsumexp",
    "margin_ranking_loss": "lucid.nn.functional.margin_ranking_loss",
    "matrix_norm": "lucid.linalg.matrix_norm",
    "mm": "lucid.mm",
    "multi_dot": "lucid.linalg.multi_dot",
    "multi_margin_loss": "lucid.nn.functional.multi_margin_loss",
    "multilabel_margin_loss": "lucid.nn.functional.multilabel_margin_loss",
    "mv": "lucid.mv",
    "poisson_nll_loss": "lucid.nn.functional.poisson_nll_loss",
    "polar": "lucid.polar",
    "prelu": "lucid.nn.functional.prelu",
    "quantile": "lucid.quantile",
    "smooth_l1_loss": "lucid.nn.functional.smooth_l1_loss",
    "soft_margin_loss": "lucid.nn.functional.soft_margin_loss",
    "softmin": "lucid.nn.functional.softmin",
    "stft": "lucid.stft",
    "tensordot": "lucid.tensordot",
    "triplet_margin_loss": "lucid.nn.functional.triplet_margin_loss",
    "vdot": "lucid.vdot",
    "vecdot": "lucid.linalg.vecdot",
    "vector_norm": "lucid.linalg.vector_norm",
}

# The two autocast settings the table is checked under: the Metal column
# with float16 and the CPU column with bfloat16 (CPU float16 computes in
# float32, so it would hide the LOWER rows).
_SETTINGS = [
    pytest.param(
        "metal",
        "float16",
        id="metal-float16",
        marks=pytest.mark.skipif(not metal_available(), reason="Metal unavailable"),
    ),
    pytest.param("cpu", "bfloat16", id="cpu-bfloat16"),
]
_INPUTS = ["float32", "half"]


def _rows() -> dict[str, tuple[Any, Any]]:
    return {name: (metal, cpu) for name, metal, cpu in _C_engine.amp_policy_table()}


def _resolve(path: str) -> object:
    module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


# ── the table itself ─────────────────────────────────────────────────────────


def test_rows_are_sorted_unique_and_valid() -> None:
    names = [name for name, _, _ in _C_engine.amp_policy_table()]
    assert names == sorted(names)
    assert len(names) == len(set(names))
    valid = {_P.Promote, _P.KeepInput, _P.ForceFP32, _P.PromoteWidest}
    for name, (metal, cpu) in _rows().items():
        assert metal in valid and cpu in valid, name
        # A row that keeps on both devices is the default, written out.
        assert (metal, cpu) != (_P.KeepInput, _P.KeepInput), name


def test_every_row_names_an_op_lucid_has() -> None:
    for name in _rows():
        if _C_engine.op_lookup(name) is not None:
            continue
        if name in _UNREGISTERED_SCHEMAS:
            assert hasattr(_C_engine, _UNREGISTERED_SCHEMAS[name]), name
            continue
        assert name in _COMPOSITES, f"{name}: no schema and no composite of that name"
        assert callable(_resolve(_COMPOSITES[name])), name


def test_the_name_lists_hold_only_rows() -> None:
    rows = _rows()
    for name in _COMPOSITES:
        assert name in rows and _C_engine.op_lookup(name) is None, name
    for name in _UNREGISTERED_SCHEMAS:
        # Registered now: drop it from the list.
        assert name in rows and _C_engine.op_lookup(name) is None, name


def test_amp_policy_for_reads_the_table() -> None:
    for name, (metal, cpu) in _rows().items():
        assert _C_engine.amp_policy_for(name, _C_engine.Device.GPU) == metal, name
        assert _C_engine.amp_policy_for(name, _C_engine.Device.CPU) == cpu, name
    for device in (_C_engine.Device.GPU, _C_engine.Device.CPU):
        assert _C_engine.amp_policy_for("not_an_op", device) == _P.KeepInput


def test_schemas_report_the_table_not_their_declaration() -> None:
    rows = _rows()
    for schema in _C_engine.op_registry_all():
        metal, cpu = rows.get(schema.name, (_P.KeepInput, _P.KeepInput))
        assert (schema.amp_policy, schema.amp_policy_cpu) == (metal, cpu), schema.name
    # ``add`` was declared Promote; it has no row, so autocast keeps it.
    add = _C_engine.op_lookup("add")
    assert add.declared_amp_policy == _P.Promote
    assert (add.amp_policy, add.amp_policy_cpu) == (_P.KeepInput, _P.KeepInput)


# ── cases: one call per op, the same code for Lucid and the reference ───────


class _Make:
    """Inputs for one case, in one library, on one device and dtype."""

    def __init__(self, lib: ModuleType, device: str, dtype: object) -> None:
        self.lib, self.device, self.dtype = lib, device, dtype

    def __call__(self, *shape: int, pos: bool = False) -> Any:
        t = self.lib.rand(*shape, device=self.device) + (0.5 if pos else -0.5)
        return t.to(self.dtype)

    def unit(self, *shape: int) -> Any:
        """Values in (0.1, 0.9) — probabilities."""
        return (self.lib.rand(*shape, device=self.device) * 0.8 + 0.1).to(self.dtype)

    def signs(self, n: int) -> Any:
        return self.lib.tensor([1.0, -1.0] * (n // 2), device=self.device).to(self.dtype)

    def ints(self, values: list[Any]) -> Any:
        return self.lib.tensor(values, device=self.device)

    def eye_plus(self, n: int) -> Any:
        """A well-conditioned square matrix."""
        eye = self.lib.eye(n, device=self.device).to(self.dtype)
        return self(n, n) + eye * 3


_Fn = Callable[[ModuleType, Any, _Make], Any]


@dataclass(frozen=True)
class _Case:
    fn: _Fn
    # The reference's spelling, where it differs from Lucid's.
    ref_fn: _Fn | None = None


def _bn(train: bool, rank: int) -> _Fn:
    def fn(lib: ModuleType, F: Any, m: _Make) -> Any:
        x = m(2, 3, *([4] * (rank - 2)))
        if train:
            return F.batch_norm(x, None, None, m(3), m(3), training=True)
        return F.batch_norm(x, m(3), m(3, pos=True), m(3), m(3), training=False)

    return fn


_CASES: dict[str, _Case] = {
    # ── LOWER ──
    "bmm": _Case(lambda lib, F, m: lib.bmm(m(1, 3, 4), m(1, 4, 2))),
    "conv1d": _Case(lambda lib, F, m: F.conv1d(m(1, 2, 8), m(3, 2, 3))),
    "conv2d": _Case(lambda lib, F, m: F.conv2d(m(1, 2, 8, 8), m(3, 2, 3, 3))),
    "conv3d": _Case(lambda lib, F, m: F.conv3d(m(1, 2, 4, 4, 4), m(3, 2, 3, 3, 3))),
    "einsum": _Case(
        lambda lib, F, m: lib.einops.einsum("ij,jk->ik", m(3, 4), m(4, 2)),
        lambda lib, F, m: lib.einsum("ij,jk->ik", m(3, 4), m(4, 2)),
    ),
    "linear": _Case(lambda lib, F, m: F.linear(m(2, 4), m(3, 4), m(3))),
    "matmul": _Case(lambda lib, F, m: m(3, 4) @ m(4, 2)),
    "mm": _Case(lambda lib, F, m: lib.mm(m(3, 4), m(4, 2))),
    "multi_dot": _Case(lambda lib, F, m: lib.linalg.multi_dot([m(3, 4), m(4, 2), m(2, 2)])),
    # ── FP32 on both ──
    "bce_with_logits": _Case(
        lambda lib, F, m: F.binary_cross_entropy_with_logits(m(4), m.unit(4))
    ),
    "cosine_embedding_loss": _Case(
        lambda lib, F, m: F.cosine_embedding_loss(m(2, 4), m(2, 4), m.signs(2))
    ),
    "cross_entropy_loss": _Case(lambda lib, F, m: F.cross_entropy(m(3, 5), m.ints([0, 2, 4]))),
    "hinge_embedding_loss": _Case(
        lambda lib, F, m: F.hinge_embedding_loss(m(4), m.signs(4))
    ),
    "huber_loss": _Case(lambda lib, F, m: F.huber_loss(m(4), m(4))),
    "l1_loss": _Case(lambda lib, F, m: F.l1_loss(m(4), m(4))),
    "margin_ranking_loss": _Case(
        lambda lib, F, m: F.margin_ranking_loss(m(4), m(4), m.signs(4))
    ),
    "mse_loss": _Case(lambda lib, F, m: F.mse_loss(m(4), m(4))),
    "multi_margin_loss": _Case(lambda lib, F, m: F.multi_margin_loss(m(3, 5), m.ints([0, 2, 4]))),
    "multilabel_margin_loss": _Case(
        lambda lib, F, m: F.multilabel_margin_loss(
            m(2, 4), m.ints([[3, 0, -1, 1], [0, 1, 2, -1]])
        )
    ),
    "nll_loss": _Case(lambda lib, F, m: F.nll_loss(m(3, 5), m.ints([0, 2, 4]))),
    "prod": _Case(lambda lib, F, m: m(4, 3).prod(0)),
    "smooth_l1_loss": _Case(lambda lib, F, m: F.smooth_l1_loss(m(4), m(4))),
    "triplet_margin_loss": _Case(
        lambda lib, F, m: F.triplet_margin_loss(m(2, 4), m(2, 4), m(2, 4))
    ),
    # ── FP32 on Metal, KEEP on the CPU ──
    "arccos": _Case(lambda lib, F, m: lib.arccos(m(4))),
    "arcsin": _Case(lambda lib, F, m: lib.arcsin(m(4))),
    "batch_norm": _Case(_bn(train=True, rank=4)),
    "cosh": _Case(lambda lib, F, m: lib.cosh(m(4))),
    "cosine_similarity": _Case(lambda lib, F, m: F.cosine_similarity(m(2, 4), m(2, 4))),
    "erfinv": _Case(lambda lib, F, m: lib.erfinv(m(4))),
    "exp": _Case(lambda lib, F, m: lib.exp(m(4))),
    "expm1": _Case(lambda lib, F, m: lib.expm1(m(4))),
    "group_norm": _Case(lambda lib, F, m: F.group_norm(m(2, 4, 3), 2)),
    "layer_norm": _Case(lambda lib, F, m: F.layer_norm(m(2, 4), (4,))),
    "log": _Case(lambda lib, F, m: lib.log(m(4, pos=True))),
    "log10": _Case(lambda lib, F, m: lib.log10(m(4, pos=True))),
    "log1p": _Case(lambda lib, F, m: lib.log1p(m(4, pos=True))),
    "log2": _Case(lambda lib, F, m: lib.log2(m(4, pos=True))),
    "log_softmax": _Case(lambda lib, F, m: F.log_softmax(m(4), 0)),
    "logsumexp": _Case(lambda lib, F, m: lib.logsumexp(m(4), 0)),
    "pow": _Case(lambda lib, F, m: lib.pow(m(4, pos=True), m(4))),
    "reciprocal": _Case(lambda lib, F, m: lib.reciprocal(m(4, pos=True))),
    "rpow_scalar": _Case(lambda lib, F, m: 2.0 ** m(4)),
    "rsqrt": _Case(lambda lib, F, m: lib.rsqrt(m(4, pos=True))),
    "sinh": _Case(lambda lib, F, m: lib.sinh(m(4))),
    "softmax": _Case(lambda lib, F, m: F.softmax(m(4), 0)),
    "softmin": _Case(lambda lib, F, m: F.softmin(m(4), 0)),
    "square": _Case(lambda lib, F, m: lib.square(m(4))),
    "sum": _Case(lambda lib, F, m: m(4, 3).sum(0)),
    "tan": _Case(lambda lib, F, m: lib.tan(m(4))),
    # ── KEEP on Metal, FP32 on the CPU ──
    "grid_sample": _Case(
        lambda lib, F, m: F.grid_sample(m(1, 1, 4, 4), m(1, 3, 3, 2), align_corners=False)
    ),
    # ── WIDEST: one dtype in, that dtype out ──
    "addcdiv": _Case(lambda lib, F, m: lib.addcdiv(m(3), m(3), m(3, pos=True))),
    "addcmul": _Case(lambda lib, F, m: lib.addcmul(m(3), m(3), m(3))),
    "atan2": _Case(lambda lib, F, m: lib.atan2(m(4), m(4))),
    "bilinear_layer": _Case(lambda lib, F, m: F.bilinear(m(2, 3), m(2, 4), m(5, 3, 4))),
    "concatenate": _Case(lambda lib, F, m: lib.cat([m(4), m(4)])),
    "cross": _Case(lambda lib, F, m: lib.linalg.cross(m(3), m(3))),
    "index_put": _Case(
        lambda lib, F, m: lib.index_put(m(4), (m.ints([0, 2, 1]),), m(3)),
        lambda lib, F, m: m(4).index_put((m.ints([0, 2, 1]),), m(3)),
    ),
    "index_scatter": _Case(lambda lib, F, m: m(4).scatter_add(0, m.ints([0, 2, 1]), m(3))),
    "stack": _Case(lambda lib, F, m: lib.stack([m(4), m(4)])),
    "vdot": _Case(lambda lib, F, m: lib.vdot(m(4), m(4))),
}

# Ops with no row keep their inputs' dtype — several of them were LOWER or
# FP32 for both devices before the table.
_KEEP_CASES: dict[str, _Case] = {
    "abs": _Case(lambda lib, F, m: lib.abs(m(4))),
    "add": _Case(lambda lib, F, m: m(4) + m(4)),
    "div": _Case(lambda lib, F, m: m(4) / m(4, pos=True)),
    "elu": _Case(lambda lib, F, m: F.elu(m(4))),
    "exp2": _Case(lambda lib, F, m: lib.exp2(m(4))),
    "floordiv": _Case(lambda lib, F, m: lib.floor_divide(m(4), m(4, pos=True))),
    "fmod": _Case(lambda lib, F, m: lib.fmod(m(4), m(4, pos=True))),
    "frac": _Case(lambda lib, F, m: lib.frac(m(4))),
    "gelu": _Case(lambda lib, F, m: F.gelu(m(4))),
    "hypot": _Case(lambda lib, F, m: lib.hypot(m(4), m(4))),
    "interpolate_bilinear": _Case(
        lambda lib, F, m: F.interpolate(m(1, 1, 4, 4), scale_factor=2, mode="bilinear")
    ),
    "kron": _Case(lambda lib, F, m: lib.kron(m(2, 2), m(2, 2))),
    "logaddexp": _Case(lambda lib, F, m: lib.logaddexp(m(4), m(4))),
    "lp_normalize": _Case(lambda lib, F, m: F.normalize(m(3, 4))),
    "maximum": _Case(lambda lib, F, m: lib.maximum(m(4), m(4))),
    "mean": _Case(lambda lib, F, m: m(4).mean()),
    "mish": _Case(lambda lib, F, m: F.mish(m(4))),
    "mul": _Case(lambda lib, F, m: m(4) * m(4)),
    "mul_scalar": _Case(lambda lib, F, m: m(4) * 0.5),
    "neg": _Case(lambda lib, F, m: -m(4)),
    "rms_norm": _Case(lambda lib, F, m: F.rms_norm(m(2, 4), (4,))),
    "selu": _Case(lambda lib, F, m: F.selu(m(4))),
    "sigmoid": _Case(lambda lib, F, m: lib.sigmoid(m(4))),
    "silu": _Case(lambda lib, F, m: F.silu(m(4))),
    "sin": _Case(lambda lib, F, m: lib.sin(m(4))),
    "sqrt": _Case(lambda lib, F, m: lib.sqrt(m(4, pos=True))),
    "sub": _Case(lambda lib, F, m: m(4) - m(4)),
    "tanh": _Case(lambda lib, F, m: lib.tanh(m(4))),
}

# Rows the engine does not apply yet, with the card that wires each.  The
# xfails are strict: wiring a row flips its test.
_E2 = "LCD-500-E2: the op bypasses SchemaGuard"
_A1 = "LCD-500-A1: Python composite, wired by @amp_op"
_NOT_WIRED: dict[str, tuple[str, _Case]] = {
    "addbmm": (_A1, _Case(lambda lib, F, m: lib.addbmm(m(3, 2), m(1, 3, 4), m(1, 4, 2)))),
    "addmm": (_A1, _Case(lambda lib, F, m: lib.addmm(m(3, 2), m(3, 4), m(4, 2)))),
    "addmv": (_A1, _Case(lambda lib, F, m: lib.addmv(m(3), m(3, 4), m(4)))),
    "addr": (_A1, _Case(lambda lib, F, m: lib.addr(m(3, 4), m(3), m(4)))),
    "adaptive_avg_pool3d": (
        _A1,
        _Case(lambda lib, F, m: F.adaptive_avg_pool3d(m(1, 2, 4, 4, 4), 2)),
    ),
    "affine_grid": (
        _E2,
        _Case(lambda lib, F, m: F.affine_grid(m(1, 2, 3), (1, 1, 4, 4), align_corners=False)),
    ),
    "avg_pool3d": (_E2, _Case(lambda lib, F, m: F.avg_pool3d(m(1, 2, 4, 4, 4), 2))),
    "baddbmm": (_A1, _Case(lambda lib, F, m: lib.baddbmm(m(1, 3, 2), m(1, 3, 4), m(1, 4, 2)))),
    "bce_loss": (_A1, _Case(lambda lib, F, m: F.binary_cross_entropy(m.unit(4), m.unit(4)))),
    "cdist": (_A1, _Case(lambda lib, F, m: lib.cdist(m(3, 4), m(2, 4)))),
    "conv_transpose1d": (
        _E2,
        _Case(lambda lib, F, m: F.conv_transpose1d(m(1, 2, 8), m(2, 3, 3))),
    ),
    "conv_transpose2d": (
        _E2,
        _Case(lambda lib, F, m: F.conv_transpose2d(m(1, 2, 8, 8), m(2, 3, 3, 3))),
    ),
    "conv_transpose3d": (
        _E2,
        _Case(lambda lib, F, m: F.conv_transpose3d(m(1, 2, 4, 4, 4), m(2, 3, 3, 3, 3))),
    ),
    "cumsum": (_E2, _Case(lambda lib, F, m: lib.cumsum(m(4), 0))),
    "cumprod": (_E2, _Case(lambda lib, F, m: lib.cumprod(m(4), 0))),
    "gaussian_nll_loss": (
        _A1,
        _Case(lambda lib, F, m: F.gaussian_nll_loss(m(4), m(4), m(4, pos=True))),
    ),
    "instance_norm": (_A1, _Case(lambda lib, F, m: F.instance_norm(m(2, 3, 4)))),
    "kl_div": (_A1, _Case(lambda lib, F, m: F.kl_div(m(4), m(4, pos=True)))),
    "local_response_norm": (
        _A1,
        _Case(lambda lib, F, m: F.local_response_norm(m(1, 4, 3), 2)),
    ),
    "max_pool3d": (_E2, _Case(lambda lib, F, m: F.max_pool3d(m(1, 2, 4, 4, 4), 2))),
    "mv": (_A1, _Case(lambda lib, F, m: lib.mv(m(3, 4), m(4)))),
    "poisson_nll_loss": (
        _A1,
        _Case(lambda lib, F, m: F.poisson_nll_loss(m(4), m(4, pos=True))),
    ),
    "pow_scalar": (_E2, _Case(lambda lib, F, m: m(4, pos=True) ** 2.5)),
    "prelu": (_A1, _Case(lambda lib, F, m: F.prelu(m(1, 2, 3), m(2)))),
    "quantile": (_A1, _Case(lambda lib, F, m: lib.quantile(m(4), 0.5))),
    "scaled_dot_product_attention": (
        _E2,
        _Case(
            lambda lib, F, m: F.scaled_dot_product_attention(
                m(1, 2, 4, 8), m(1, 2, 4, 8), m(1, 2, 4, 8)
            )
        ),
    ),
    "soft_margin_loss": (_A1, _Case(lambda lib, F, m: F.soft_margin_loss(m(4), m.signs(4)))),
    "softplus": (_E2, _Case(lambda lib, F, m: F.softplus(m(4)))),
    "tensordot": (_E2, _Case(lambda lib, F, m: lib.tensordot(m(3, 4), m(4, 2), 1))),
    "trace": (_E2, _Case(lambda lib, F, m: lib.trace(m(3, 3)))),
    "vecdot": (_A1, _Case(lambda lib, F, m: lib.linalg.vecdot(m(3, 4), m(3, 4)))),
    "vector_norm": (_A1, _Case(lambda lib, F, m: lib.linalg.vector_norm(m(4)))),
}

_POLICY_DTYPE = {
    _P.Promote: lambda autocast_dt, input_dt: autocast_dt,
    _P.ForceFP32: lambda autocast_dt, input_dt: "float32",
    _P.KeepInput: lambda autocast_dt, input_dt: input_dt,
    _P.PromoteWidest: lambda autocast_dt, input_dt: input_dt,
}


def _name(dtype: object) -> str:
    return str(dtype).split(".")[-1]


def _run_lucid(case: _Case, device: str, autocast_dt: str, input_dt: str) -> str:
    dtype = getattr(lucid, autocast_dt if input_dt == "half" else "float32")
    lucid.manual_seed(0)
    with autocast(device_type=device, dtype=getattr(lucid, autocast_dt)):
        out = case.fn(lucid, lucid.nn.functional, _Make(lucid, device, dtype))
    return _name(out.dtype)


def _want(name: str, device: str, autocast_dt: str, input_dt: str) -> str:
    metal, cpu = _rows().get(name, (_P.KeepInput, _P.KeepInput))
    policy = metal if device == "metal" else cpu
    return _POLICY_DTYPE[policy](autocast_dt, autocast_dt if input_dt == "half" else "float32")


@pytest.mark.parametrize("input_dt", _INPUTS)
@pytest.mark.parametrize(("device", "autocast_dt"), _SETTINGS)
@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_row_gives_its_dtype(name: str, device: str, autocast_dt: str, input_dt: str) -> None:
    assert name in _rows(), f"{name} is a case for a row it does not have"
    got = _run_lucid(_CASES[name], device, autocast_dt, input_dt)
    assert got == _want(name, device, autocast_dt, input_dt)


@pytest.mark.parametrize("input_dt", _INPUTS)
@pytest.mark.parametrize(("device", "autocast_dt"), _SETTINGS)
@pytest.mark.parametrize("name", sorted(_KEEP_CASES))
def test_an_op_without_a_row_keeps_its_dtype(
    name: str, device: str, autocast_dt: str, input_dt: str
) -> None:
    got = _run_lucid(_KEEP_CASES[name], device, autocast_dt, input_dt)
    assert got == (autocast_dt if input_dt == "half" else "float32")


def _not_wired_params() -> list[Any]:
    return [
        pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=reason))
        for name, (reason, _) in sorted(_NOT_WIRED.items())
    ]


@pytest.mark.parametrize("name", _not_wired_params())
def test_a_row_not_wired_yet(name: str) -> None:
    case = _NOT_WIRED[name][1]
    settings = [("cpu", "bfloat16")] + ([("metal", "float16")] if metal_available() else [])
    for device, autocast_dt in settings:
        for input_dt in _INPUTS:
            got = _run_lucid(case, device, autocast_dt, input_dt)
            assert got == _want(name, device, autocast_dt, input_dt), (device, input_dt)


def test_the_not_wired_list_holds_only_rows() -> None:
    rows = _rows()
    for name in _NOT_WIRED:
        assert name in rows, name
    assert not set(_NOT_WIRED) & set(_CASES)


# ── documented deviations ────────────────────────────────────────────────────


def test_cpu_batch_norm_eval_computes_in_float32() -> None:
    """``batch_norm_eval`` departs from the reference on the CPU (FP32 where
    it keeps the input): the CPU eval kernel has no half path (LCD-487), and
    a CNN in eval mode under a CPU bfloat16 autocast hands it the conv's
    bfloat16 output.  Drop this test with the row when LCD-487 lands."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU()).eval()
    x = lucid.randn(2, 3, 8, 8)
    with autocast(device_type="cpu", dtype=lucid.bfloat16):
        y = model(x)
    assert y.dtype == lucid.float32
    assert _rows()["batch_norm_eval"][1] == _P.ForceFP32


def test_cpu_ctc_loss_takes_a_bfloat16_log_softmax() -> None:
    """Under a CPU autocast ``log_softmax`` keeps bfloat16 now; ``ctc_loss``
    is FP32 and casts it, where it used to reach a kernel without a half
    path."""
    logits = lucid.randn(6, 1, 4).to(lucid.bfloat16).requires_grad_()
    targets = lucid.tensor([[1, 2]])
    with autocast(device_type="cpu", dtype=lucid.bfloat16):
        log_probs = logits.log_softmax(-1)
        loss = lucid.nn.functional.ctc_loss(
            log_probs, targets, lucid.tensor([6]), lucid.tensor([2])
        )
    assert log_probs.dtype == lucid.bfloat16
    assert loss.dtype == lucid.float32
    loss.backward()
    assert logits.grad is not None and logits.grad.dtype == lucid.bfloat16

    want = lucid.nn.functional.ctc_loss(
        logits.detach().float().log_softmax(-1), targets, lucid.tensor([6]), lucid.tensor([2])
    )
    assert abs(loss.item() - want.item()) < 5e-2


# ── the engine turns autocast off inside an op, and back on after it ────────


@pytest.mark.parametrize(("device", "autocast_dt"), _SETTINGS)
def test_autocast_is_back_on_after_an_op_returns_or_raises(
    device: str, autocast_dt: str
) -> None:
    dt = getattr(lucid, autocast_dt)
    x = lucid.randn(3, 4, device=device)
    with autocast(device_type=device, dtype=dt):
        (x @ x.T).sum()
        lucid.log1p(x.abs())
        assert autocast.is_autocast_enabled(device)
        assert autocast.get_autocast_dtype(device) == dt
        with pytest.raises(ValueError):
            x @ x
        assert autocast.is_autocast_enabled(device)
        assert (x @ x.T).dtype == dt


# ── parity: the dtype the reference's autocast gives ────────────────────────


def _ref_device(ref: ModuleType, device: str) -> str:
    if device == "metal":
        if not ref.backends.mps.is_available():
            pytest.skip("the reference's MPS backend is unavailable")
        return "mps"
    return "cpu"


@pytest.mark.parity
@pytest.mark.parametrize("input_dt", _INPUTS)
@pytest.mark.parametrize(("device", "autocast_dt"), _SETTINGS)
@pytest.mark.parametrize("name", sorted(_CASES) + sorted(_KEEP_CASES))
def test_the_dtype_matches_the_reference(
    ref: ModuleType, name: str, device: str, autocast_dt: str, input_dt: str
) -> None:
    case = _CASES.get(name) or _KEEP_CASES[name]
    got = _run_lucid(case, device, autocast_dt, input_dt)

    ref_device = _ref_device(ref, device)
    ref_dt = getattr(ref, autocast_dt)
    dtype = ref_dt if input_dt == "half" else ref.float32
    try:
        with ref.autocast(device_type=ref_device, dtype=ref_dt):
            out = (case.ref_fn or case.fn)(ref, ref.nn.functional, _Make(ref, ref_device, dtype))
    except NotImplementedError:
        # Its MPS backend lacks some losses; nothing to compare against.
        pytest.skip(f"the reference has no {ref_device} kernel for {name}")
    assert got == _name(out.dtype)
