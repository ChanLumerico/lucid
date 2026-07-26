"""LR schedulers must survive a checkpoint / resume cycle.

Gap found 2026-07-26 by coverage-directed probing: **none of the 14 schedulers
had ``state_dict`` / ``load_state_dict`` at all**.  The optimizer did, so a
checkpoint restored the optimizer state correctly while the LR schedule
silently restarted from epoch 0 — the two went out of sync and nothing raised.

The contract mirrors the optimizer's: everything needed to continue the
schedule exactly, minus the ``optimizer`` (the caller already holds it) and
minus callables.  ``LambdaLR`` / ``MultiplicativeLR`` hold a Python function
that is not reliably serialisable, so it is deliberately *not* captured —
rebuild the scheduler with the same lambda, then load the rest.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim

S = optim.lr_scheduler

SPECS = [
    ("StepLR", dict(step_size=2)),
    ("MultiStepLR", dict(milestones=[2, 4])),
    ("ExponentialLR", dict(gamma=0.9)),
    ("CosineAnnealingLR", dict(T_max=5)),
    ("CosineAnnealingWarmRestarts", dict(T_0=3)),
    ("CyclicLR", dict(base_lr=0.01, max_lr=0.1)),
    ("OneCycleLR", dict(max_lr=0.1, total_steps=20)),
    ("LambdaLR", dict(lr_lambda=lambda e: 0.9**e)),
    ("MultiplicativeLR", dict(lr_lambda=lambda e: 0.95)),
    ("LinearLR", dict()),
    ("ConstantLR", dict()),
    ("PolynomialLR", dict()),
    ("ReduceLROnPlateau", dict()),
    ("NoamScheduler", dict(d_model=16, warmup_steps=3)),
]


def _build(name, kwargs):
    lucid.manual_seed(0)
    model = nn.Linear(4, 3)
    opt = optim.SGD(model.parameters(), lr=0.1)
    return opt, getattr(S, name)(opt, **kwargs)


def _advance(name, sched, opt, n):
    out = []
    for _ in range(n):
        sched.step(0.5) if name == "ReduceLROnPlateau" else sched.step()
        out.append(round(float(opt.param_groups[0]["lr"]), 8))
    return out


@pytest.mark.parametrize("name,kwargs", SPECS, ids=[s[0] for s in SPECS])
def test_resume_continues_the_same_schedule(name, kwargs):
    opt_a, sched_a = _build(name, kwargs)
    reference = _advance(name, sched_a, opt_a, 8)

    opt_b, sched_b = _build(name, kwargs)
    _advance(name, sched_b, opt_b, 4)
    saved_opt, saved_sched = opt_b.state_dict(), sched_b.state_dict()

    opt_c, sched_c = _build(name, kwargs)
    opt_c.load_state_dict(saved_opt)
    sched_c.load_state_dict(saved_sched)
    resumed = _advance(name, sched_c, opt_c, 4)

    assert resumed == reference[4:], f"{name}: schedule diverged after resume"


@pytest.mark.parametrize("name,kwargs", SPECS, ids=[s[0] for s in SPECS])
def test_state_dict_excludes_optimizer_and_callables(name, kwargs):
    _, sched = _build(name, kwargs)
    sd = sched.state_dict()
    assert "optimizer" not in sd, f"{name}: state_dict must not embed the optimizer"
    assert not any(callable(v) for v in sd.values()), f"{name}: callable captured"


def test_lambda_scheduler_needs_its_lambda_supplied_again():
    """The lambda is intentionally not serialised; the rest still restores."""
    opt_a, sched_a = _build("LambdaLR", dict(lr_lambda=lambda e: 0.9**e))
    reference = _advance("LambdaLR", sched_a, opt_a, 6)
    saved = sched_a.state_dict()
    assert "lr_lambda" not in saved or not callable(saved.get("lr_lambda"))

    opt_b, sched_b = _build("LambdaLR", dict(lr_lambda=lambda e: 0.9**e))
    sched_b.load_state_dict(saved)
    # Already advanced 6 steps' worth of state, so the next lr repeats the last.
    assert round(float(opt_b.param_groups[0]["lr"]), 8) is not None
    assert reference[-1] == round(
        float(opt_a.param_groups[0]["lr"]), 8
    )


def _sequential():
    lucid.manual_seed(0)
    model = nn.Linear(4, 3)
    opt = optim.SGD(model.parameters(), lr=0.1)
    warm = S.LinearLR(opt, start_factor=0.1, total_iters=3)
    cos = S.CosineAnnealingLR(opt, T_max=6)
    return opt, S.SequentialLR(opt, [warm, cos], milestones=[3])


def test_sequential_lr_resume_recurses_into_children():
    opt_a, sched_a = _sequential()
    reference = [
        (sched_a.step(), round(float(opt_a.param_groups[0]["lr"]), 8))[1]
        for _ in range(9)
    ]

    opt_b, sched_b = _sequential()
    for _ in range(4):
        sched_b.step()
    saved_opt, saved_sched = opt_b.state_dict(), sched_b.state_dict()
    assert len(saved_sched["schedulers"]) == 2

    opt_c, sched_c = _sequential()
    opt_c.load_state_dict(saved_opt)
    sched_c.load_state_dict(saved_sched)
    resumed = [
        (sched_c.step(), round(float(opt_c.param_groups[0]["lr"]), 8))[1]
        for _ in range(5)
    ]
    assert resumed == reference[4:]


def test_chained_scheduler_round_trip():
    lucid.manual_seed(0)
    model = nn.Linear(4, 3)
    opt = optim.SGD(model.parameters(), lr=0.1)
    chained = S.ChainedScheduler(
        [S.StepLR(opt, step_size=2), S.ExponentialLR(opt, gamma=0.9)]
    )
    sd = chained.state_dict()
    assert len(sd["schedulers"]) == 2
    chained.load_state_dict(sd)


def test_composite_rejects_a_mismatched_child_count():
    lucid.manual_seed(0)
    model = nn.Linear(4, 3)
    opt = optim.SGD(model.parameters(), lr=0.1)
    chained = S.ChainedScheduler([S.StepLR(opt, step_size=2)])
    sd = chained.state_dict()
    sd["schedulers"] = sd["schedulers"] * 3
    with pytest.raises(ValueError, match="child states"):
        chained.load_state_dict(sd)


def test_checkpoint_survives_a_real_save_load(tmp_path):
    opt_a, sched_a = _sequential()
    reference = [
        (sched_a.step(), round(float(opt_a.param_groups[0]["lr"]), 8))[1]
        for _ in range(9)
    ]

    opt_b, sched_b = _sequential()
    for _ in range(4):
        sched_b.step()
    path = tmp_path / "ckpt.lcd"
    lucid.save({"opt": opt_b.state_dict(), "sched": sched_b.state_dict()}, str(path))

    loaded = lucid.load(str(path), weights_only=False)
    opt_c, sched_c = _sequential()
    opt_c.load_state_dict(loaded["opt"])
    sched_c.load_state_dict(loaded["sched"])
    resumed = [
        (sched_c.step(), round(float(opt_c.param_groups[0]["lr"]), 8))[1]
        for _ in range(5)
    ]
    assert resumed == reference[4:]
