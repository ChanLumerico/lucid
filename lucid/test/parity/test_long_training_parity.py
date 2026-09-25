"""Three thousand training steps against the reference, and what they may differ by.

A smooth model — LayerNorm, GELU, softmax attention, AdamW — stays on the
reference's trajectory for the whole run: measured, every one of 3,072 step
losses within 1.3e-7 relative, on both devices.  It is held to 1e-5 at every
step, so a systematic error of that size anywhere in the step shows.  The
first run of this file found one: five LR schedulers ran the first optimizer
step at the base rate, a 1% gap in the first tenth of the run.

A model with ReLU and BatchNorm is chaotic: rounding moves a kink, and the
two runs part within tens of steps.  Its step losses are compared only while
they still agree; after that, its windowed losses and final accuracy are held
to a multiple of how far two Lucid runs started a float-rounding apart drift
from each other — chaos measured, not guessed.

Run in the nightly parity job (``scripts/ci_nightly_suite.py``), where the
reference is installed.
"""

from functools import lru_cache
from typing import Any

import numpy as np
import pytest

from lucid.test.parity import _long_harness as H

pytestmark = [pytest.mark.parity, pytest.mark.slow]

DEVICES = ["metal", "cpu"]

# The smooth run: every step's loss, and the held-out loss at the end.
GPT_STEP_RTOL = 1e-5

# The chaotic run.  Step losses agree to 1e-3 for the first ~20 steps
# (measured 19 and 22); a systematic difference shows before that.
RESNET_EARLY_STEPS = 10
RESNET_EARLY_RTOL = 1e-3
# How many times the chaos spread Lucid against the reference may reach.
RESNET_CHAOS_MULTIPLE = 3.0


@lru_cache(maxsize=None)
def _reference_run(recipe_name: str, ref: Any) -> H.Run:
    recipe = {"resnet_sgd_cosine": H.RESNET, "gpt_adamw_warmup": H.GPT}[recipe_name]
    runner = H.run_images if recipe is H.RESNET else H.run_text
    _, theirs = H.build_pair(recipe, "cpu", ref)
    return runner(recipe, "ref", theirs, "cpu", ref)


@pytest.mark.parametrize("device", DEVICES)
def test_the_gpt_run_stays_on_the_references_trajectory(device: str, ref: Any) -> None:
    mine, _ = H.build_pair(H.GPT, device, ref)
    got = H.run_text(H.GPT, "lucid", mine, device, ref)
    want = _reference_run(H.GPT.name, ref)

    rel = np.abs(np.array(got.losses) - want.losses) / np.abs(want.losses)
    worst = int(rel.argmax())
    assert rel.max() <= GPT_STEP_RTOL, (
        f"step {worst}: loss {got.losses[worst]:.7f} vs reference "
        f"{want.losses[worst]:.7f} (relative {rel.max():.2e})"
    )
    assert abs(got.metric - want.metric) <= GPT_STEP_RTOL * want.metric
    # Agreeing is not enough: both have to have learned the chain, whose own
    # conditional entropy is the floor.
    floor = H.markov_corpus(H.SEED).entropy
    assert got.metric < floor + 0.35, f"held-out {got.metric:.3f}, floor {floor:.3f}"


@pytest.mark.parametrize("device", DEVICES)
def test_the_resnet_run_differs_from_the_reference_only_as_chaos_does(
    device: str, ref: Any
) -> None:
    runs = {}
    for tag, seed in (("A", None), ("B", 1), ("C", 2)):
        mine, _ = H.build_pair(H.RESNET, device, ref)
        if seed is not None:
            H.nudge(mine, seed)
        runs[tag] = H.run_images(H.RESNET, "lucid", mine, device, ref)
    want = _reference_run(H.RESNET.name, ref)
    a = np.array(runs["A"].losses)

    early = np.abs(a[:RESNET_EARLY_STEPS] - want.losses[:RESNET_EARLY_STEPS])
    early /= np.abs(want.losses[:RESNET_EARLY_STEPS])
    assert early.max() <= RESNET_EARLY_RTOL, (
        f"the first {RESNET_EARLY_STEPS} steps already differ by {early.max():.2e} "
        "— too soon for chaos"
    )

    windows = {k: H.windowed_means(v.losses) for k, v in runs.items()}
    reference = H.windowed_means(want.losses)
    chaos = max(
        np.abs(windows["A"] - windows["B"]).mean(),
        np.abs(windows["A"] - windows["C"]).mean(),
        np.abs(windows["B"] - windows["C"]).mean(),
    )
    gap = np.abs(windows["A"] - reference).mean()
    assert gap <= RESNET_CHAOS_MULTIPLE * chaos, (
        f"windowed loss differs from the reference by {gap:.4f} on average; "
        f"two Lucid runs a rounding apart differ by {chaos:.4f}"
    )

    metrics = [runs[k].metric for k in ("A", "B", "C")]
    spread = max(metrics) - min(metrics)
    assert abs(runs["A"].metric - want.metric) <= max(
        RESNET_CHAOS_MULTIPLE * spread, 0.01
    ), f"accuracy {runs['A'].metric:.4f} vs reference {want.metric:.4f}"
    assert runs["A"].metric > 0.8, "the run did not learn the task"
