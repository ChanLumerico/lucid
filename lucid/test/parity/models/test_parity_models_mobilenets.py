"""Model parity tests — MobileNet family.

Covers:
  MobileNet v1  (Howard et al., 2017)  — 4.2 M
  MobileNet v2  (Sandler et al., 2018) — 3.5 M
  MobileNet v3  (Howard et al., 2019)  — 2.5 / 5.5 M
  MobileNet v4  (Qin et al., 2024)     — 3.4 M

All small enough to run without the slow/heavy markers.

timm names
----------
  MobileNet v2 (1.0x) → ``mobilenetv2_100``
  MobileNet v3 Large  → ``mobilenetv3_large_100``
  MobileNet v3 Small  → ``mobilenetv3_small_100``

MobileNet v1 and v4 have no exact timm equivalent with our architecture
so they use self-consistency tests only.
"""

import lucid.models as M
from lucid.test.parity.models._utils import (
    requires_timm,
    run_parity,
    run_self_consistency,
)

_INPUT = (1, 3, 224, 224)


# ── MobileNet v1 (self-consistency — no timm equivalent) ─────────────────────


class TestMobileNetV1SelfConsistency:

    def test_mobilenet_v1_deterministic(self) -> None:
        run_self_consistency(M.mobilenet_v1_cls(), input_shape=_INPUT)

    def test_mobilenet_v1_075_deterministic(self) -> None:
        run_self_consistency(M.mobilenet_v1_075_cls(), input_shape=_INPUT)

    def test_mobilenet_v1_050_deterministic(self) -> None:
        run_self_consistency(M.mobilenet_v1_050_cls(), input_shape=_INPUT)

    def test_mobilenet_v1_025_deterministic(self) -> None:
        run_self_consistency(M.mobilenet_v1_025_cls(), input_shape=_INPUT)


# ── MobileNet v2 ──────────────────────────────────────────────────────────────


@requires_timm
class TestMobileNetV2Parity:

    def test_mobilenet_v2_100_parity(self) -> None:
        run_parity(M.mobilenet_v2_cls(), "mobilenetv2_100", input_shape=_INPUT)

    def test_mobilenet_v2_075_parity(self) -> None:
        run_parity(M.mobilenet_v2_075_cls(), "mobilenetv2_075", input_shape=_INPUT)


class TestMobileNetV2SelfConsistency:

    def test_mobilenet_v2_deterministic(self) -> None:
        run_self_consistency(M.mobilenet_v2_cls(), input_shape=_INPUT)


# ── MobileNet v3 ──────────────────────────────────────────────────────────────


@requires_timm
class TestMobileNetV3Parity:

    def test_mobilenet_v3_large_parity(self) -> None:
        run_parity(
            M.mobilenet_v3_large_cls(),
            "mobilenetv3_large_100",
            input_shape=_INPUT,
        )

    def test_mobilenet_v3_small_parity(self) -> None:
        run_parity(
            M.mobilenet_v3_small_cls(),
            "mobilenetv3_small_100",
            input_shape=_INPUT,
        )


class TestMobileNetV3SelfConsistency:

    def test_mobilenet_v3_large_deterministic(self) -> None:
        run_self_consistency(M.mobilenet_v3_large_cls(), input_shape=_INPUT)

    def test_mobilenet_v3_small_deterministic(self) -> None:
        run_self_consistency(M.mobilenet_v3_small_cls(), input_shape=_INPUT)


# ── MobileNet v4 (self-consistency — architecture newer than timm stable) ─────


class TestMobileNetV4SelfConsistency:

    def test_mobilenet_v4_conv_small_deterministic(self) -> None:
        run_self_consistency(M.mobilenet_v4_conv_small_cls(), input_shape=_INPUT)
