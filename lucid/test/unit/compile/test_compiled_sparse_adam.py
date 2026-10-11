"""The compiled SparseAdam is the eager one: lazy per row, one count per parameter.

The equivalence suite's models have a gradient everywhere, so a dense Adam
passed it (LCD-427).  Each model here leaves rows untouched: an embedding
looked up at a few indices, a 3-d parameter whose gradient zeroes whole
dim-0 slices, a 0-d parameter, and a parameter with no gradient at all.
``compile_optimizer`` reads the eager gradients and follows the eager
arithmetic order, so it is exact in float32; in float16 MPSGraph and the
eager kernels round intermediates differently (every compiled optimizer,
SGD included, lands a few ulp apart), so float16 gets a few ulp.
``fused_step`` derives its gradients in MPSGraph, so it gets the
equivalence suite's 1e-6.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import compile_optimizer, fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE

# Index batches: row 3 is never looked up; rows 1 and 2 sit out some steps.
BATCHES = ([0, 1], [2, 2], [0, 0], [0, 2], [1, 1], [0, 0])
FUSED_TOL = 1e-6
# Exact in float32; a few float16 ulp around |x| ~ 2 (see the docstring).
TOL = {lucid.float32: 0.0, lucid.float16: 4e-3}


class _Rows(nn.Module):
    """An embedding, a 3-d table indexed on dim 0, a 0-d scale, an unused bias."""

    def __init__(self, dtype: lucid.dtype) -> None:
        super().__init__()
        self.emb = nn.Embedding(5, 3)
        self.table = nn.Parameter(lucid.randn(5, 2, 2))
        self.scale = nn.Parameter(lucid.tensor(1.5))
        self.unused = nn.Parameter(lucid.randn(4))
        self.to(device=COMPILE_DEVICE, dtype=dtype)

    def forward(self, idx: lucid.Tensor) -> lucid.Tensor:
        rows = self.table[idx].reshape(idx.shape[0], -1)[:, :3]
        return self.emb(idx) * self.scale + rows


def _model(dtype: lucid.dtype = lucid.float32) -> _Rows:
    lucid.manual_seed(0)
    return _Rows(dtype)


def _idx(batch: list[int]) -> lucid.Tensor:
    return lucid.tensor(batch).to(COMPILE_DEVICE)


def _target(batch: list[int], dtype: lucid.dtype) -> lucid.Tensor:
    lucid.manual_seed(len(batch))
    return lucid.randn(len(batch), 3).to(device=COMPILE_DEVICE, dtype=dtype)


def _loss(model: nn.Module, idx: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
    return F.mse_loss(model(idx), t)


def _step(model: nn.Module, stepper: object, batch: list[int]) -> None:
    dtype = model.emb.weight.dtype  # type: ignore[union-attr]
    stepper.zero_grad()  # type: ignore[attr-defined]
    _loss(model, _idx(batch), _target(batch, dtype)).backward()
    stepper.step()  # type: ignore[attr-defined]


def _params(model: nn.Module) -> list[lucid.Tensor]:
    return [p.detach().to("cpu").float() for p in model.parameters()]


def _max_diff(a: list[lucid.Tensor], b: list[lucid.Tensor]) -> float:
    """Largest elementwise gap; a NaN in both at one place counts as equal.

    float16 rounds the default ``eps`` (1e-8) to zero, so an element of a
    touched row whose gradient is zero divides 0 by 0 — in eager as well.
    """
    out = 0.0
    for x, y in zip(a, b, strict=True):
        assert (x.isnan() == y.isnan()).all().item(), "NaN in one path only"
        gap = lucid.where(x.isnan(), lucid.zeros_like(x), (x - y).abs())
        out = max(out, float(gap.max().item()) if gap.numel() else 0.0)
    return out


Make = Callable[[nn.Module], optim.Optimizer]


def _sparse_adam(model: nn.Module) -> optim.Optimizer:
    return optim.SparseAdam(model.parameters(), lr=0.1)


@pytest.mark.parametrize("dtype", [lucid.float32, lucid.float16], ids=["f32", "f16"])
def test_compiled_matches_eager_with_untouched_rows(dtype: lucid.dtype) -> None:
    eager, comp = _model(dtype), _model(dtype)
    e_opt, c_opt = _sparse_adam(eager), _sparse_adam(comp)
    stepper = compile_optimizer(c_opt)
    before = _params(eager)
    for k, batch in enumerate(BATCHES):
        _step(eager, e_opt, batch)
        _step(comp, stepper, batch)
        drift = _max_diff(_params(eager), _params(comp))
        assert drift <= TOL[dtype], f"step {k + 1}: drift {drift:.3e}"
    after = _params(comp)
    # Row 3 was never looked up: it, and the parameter without a gradient,
    # did not move.  Order: table, scale, unused, emb.weight.
    assert (after[0][3] == before[0][3]).all().item()
    assert (after[2] == before[2]).all().item()
    assert (after[3][3] == before[3][3]).all().item()


def test_untouched_rows_keep_moments_and_the_count_is_per_parameter() -> None:
    eager, comp = _model(), _model()
    e_opt, c_opt = _sparse_adam(eager), _sparse_adam(comp)
    stepper = compile_optimizer(c_opt)
    for batch in BATCHES:
        _step(eager, e_opt, batch)
        _step(comp, stepper, batch)
    want, got = e_opt.state_dict()["state"], c_opt.state_dict()["state"]
    # Order: table, scale, unused, emb.weight — no entry for ``unused``.
    assert sorted(got) == sorted(want) == [0, 1, 3]
    for idx, entry in want.items():  # type: ignore[union-attr]
        assert got[idx]["step"] == entry["step"] == len(BATCHES)  # type: ignore[index]
        assert isinstance(got[idx]["step"], int)  # type: ignore[index]
        for key in ("exp_avg", "exp_avg_sq"):
            assert (got[idx][key] == entry[key]).all(), (idx, key)  # type: ignore[index]
    # Row 3 of the embedding was never touched: its moments are still zero.
    assert not got[3]["exp_avg"][3].any()  # type: ignore[index]


def test_masked_plan_keeps_the_row_rule() -> None:
    """Past the exact-plan limit a masked plan runs; it keeps the row mask."""
    eager, comp = _model(), _model()
    e_opt, c_opt = _sparse_adam(eager), _sparse_adam(comp)
    stepper = compile_optimizer(c_opt)
    frozen_cycle = ["scale", "table", "emb", None, "scale", "emb", "table"]
    for k, name in enumerate(frozen_cycle):
        batch = BATCHES[k % len(BATCHES)]
        for model, opt in ((eager, e_opt), (comp, stepper)):
            opt.zero_grad()  # type: ignore[attr-defined]
            _loss(model, _idx(batch), _target(batch, lucid.float32)).backward()
            if name is not None:
                frozen = model.emb.weight if name == "emb" else getattr(model, name)
                frozen.grad = None
            opt.step()  # type: ignore[attr-defined]
        assert _max_diff(_params(eager), _params(comp)) == 0.0, f"step {k + 1}"
    assert any(plan.masked for plan in stepper._plans.values())  # type: ignore[attr-defined]


def test_compiling_after_eager_steps_adopts_their_state() -> None:
    """Eager warm-up, then compile: the same run as all-eager (state adopted)."""
    reference, warm = _model(), _model()
    r_opt, w_opt = _sparse_adam(reference), _sparse_adam(warm)
    for batch in BATCHES[:3]:
        _step(reference, r_opt, batch)
        _step(warm, w_opt, batch)
    stepper = compile_optimizer(w_opt)
    for k, batch in enumerate(BATCHES[3:]):
        _step(reference, r_opt, batch)
        _step(warm, stepper, batch)
        assert _max_diff(_params(reference), _params(warm)) == 0.0, f"step {k + 4}"
    assert w_opt.state_dict()["state"][0]["step"] == len(BATCHES)  # type: ignore[index]


def test_fused_step_matches_eager_with_untouched_rows() -> None:
    eager, comp = _model(), _model()
    e_opt = _sparse_adam(eager)
    fused = fused_step(comp, F.mse_loss, _sparse_adam(comp))
    for k, batch in enumerate(BATCHES):
        _step(eager, e_opt, batch)
        fused(_idx(batch), _target(batch, lucid.float32))
        drift = _max_diff(_params(eager), _params(comp))
        assert drift < FUSED_TOL, f"step {k + 1}: drift {drift:.3e}"


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda ps: optim.SparseAdam(ps, lr=0.1), id="SparseAdam"),
        pytest.param(lambda ps: optim.Adam(ps, lr=0.1), id="Adam"),
        pytest.param(lambda ps: optim.SGD(ps, lr=0.1, momentum=0.9), id="SGD"),
    ],
)
def test_adopted_state_of_a_0d_parameter_stays_0d(
    make: Callable[[list[nn.Parameter]], optim.Optimizer],
) -> None:
    """Two eager steps, compile, two more: the run of four eager steps.

    The compiled loader reads each host entry through the eager loader's
    door, so a 0-d parameter's state stays 0-d.
    """

    def run(compile_at: int | None) -> nn.Parameter:
        p = nn.Parameter(lucid.tensor(-0.5).to(COMPILE_DEVICE))
        stepper: object = make([p])
        for k in range(4):
            if k == compile_at:
                stepper = compile_optimizer(stepper)  # type: ignore[arg-type]
            stepper.zero_grad()  # type: ignore[attr-defined]
            (p * p * 3.0).backward()
            stepper.step()  # type: ignore[attr-defined]
        return p

    got, want = run(2), run(None)
    assert got.shape == ()
    assert got.item() == want.item()
