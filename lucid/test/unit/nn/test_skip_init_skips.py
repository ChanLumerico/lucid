"""``skip_init`` skips the initialisers instead of running and discarding them.

It used to construct the module normally — every Kaiming draw, every fill
— and then swap each parameter for fresh empty storage: no faster than
plain construction (0.36 s against 0.38 s for an 8192x8192 Linear, as
reported while porting Self-Forcing) and briefly allocating the model
twice.  The initialisers are no-ops while the module is built now.
"""

import inspect
import time

import numpy as np

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid.nn.utils import skip_init


def test_the_initialisers_do_not_run() -> None:
    lucid.manual_seed(0)
    expected_next = lucid.randn(4).tolist()
    lucid.manual_seed(0)
    layer = skip_init(nn.Linear, 64, 32)
    # No random draw happened: the generator is where it was left.
    assert lucid.randn(4).tolist() == expected_next
    assert tuple(layer.weight.shape) == (32, 64) and layer.weight.requires_grad


def test_it_is_faster_than_building_normally() -> None:
    nn.Linear(8, 8)  # warm up
    start = time.perf_counter()
    nn.Linear(2048, 2048)
    built = time.perf_counter() - start
    start = time.perf_counter()
    skip_init(nn.Linear, 2048, 2048)
    skipped = time.perf_counter() - start
    assert skipped < built / 2, (skipped, built)


def test_the_module_works_once_loaded_and_initialisers_work_again() -> None:
    reference = nn.Linear(6, 3)
    layer = skip_init(nn.Linear, 6, 3)
    layer.load_state_dict(reference.state_dict())
    x = lucid.randn(2, 6)
    np.testing.assert_allclose(layer(x).numpy(), reference(x).numpy())
    t = lucid.zeros(3)
    init.ones_(t)
    assert t.tolist() == [1.0, 1.0, 1.0]


def test_the_initialisers_keep_their_signatures() -> None:
    assert list(inspect.signature(init.kaiming_uniform_).parameters)[0] == "tensor"
