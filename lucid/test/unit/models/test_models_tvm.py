"""Terminal Velocity Matching — the equations the method rests on.

A shape test passes on a model whose Jacobian-vector product runs along
the wrong variable, whose modulation norm is missing, or whose EMA target
is the live network by accident; all three still train to a finite loss.
So these tests assert the paper's statements instead: the terminal
velocity is the ``s``-derivative of the displacement, the semi-Lipschitz
block is invariant to the scale of its modulation, the scaled
parameterisation multiplies by ``w`` and nothing else, the EMA target
moves by Algorithm 1's rule and is not a parameter, and the sampler is
Figure 3.
"""

import math

import pytest

import lucid
import lucid.linalg
from lucid.models import DiffusionModelOutput
from lucid.models.generative.tvm import (
    TVMConfig,
    TVMForImageGeneration,
    TVMModel,
    tvm_imagenet256,
    tvm_imagenet512,
)
from lucid.nn._shadow import shadow_alloc


def _tiny(**overrides: object) -> TVMConfig:
    """A model small enough to instantiate per test."""
    base = dict(
        sample_size=8,
        patch_size=2,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
    )
    base.update(overrides)
    return TVMConfig(**base)  # type: ignore[arg-type]


def _wake(model: TVMModel) -> None:
    """Undo adaLN-Zero so the conditioning reaches the output.

    The modulation projections and the decoder start at zero, which makes
    the untrained field identically zero — any test *about* the
    conditioning has to break that first.
    """
    with lucid.no_grad():
        for block in model.blocks:
            linear = block.ada_ln[1]
            linear.weight += lucid.randn(linear.weight.shape) * 0.5
        final_linear = model.final.ada_ln[1]
        final_linear.weight += lucid.randn(final_linear.weight.shape) * 0.5
        model.final.proj.weight += lucid.randn(model.final.proj.weight.shape) * 0.5


def _gap(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


class TestNetwork:
    def test_the_untrained_field_is_exactly_zero(self) -> None:
        """adaLN-Zero and a zero decoder: the field starts at zero.

        A non-zero start means one of the zero-initialised projections was
        given the spectral initialisation instead.
        """
        model = TVMModel(_tiny()).eval()
        out = model(
            lucid.randn((2, 4, 8, 8)),
            lucid.tensor([1.0, 0.7]),
            lucid.tensor([0.0, 0.2]),
            lucid.tensor([1, 3]),
        )
        assert isinstance(out, DiffusionModelOutput) and out.loss is None
        assert float(out.sample.abs().max().item()) == 0.0

    def test_linear_layers_start_at_unit_spectral_norm(self) -> None:
        """Table 6's "Spectral" initialisation: largest singular value one."""
        model = TVMModel(_tiny())
        block = model.blocks[0]
        for linear in (block.attn.qkv, block.attn.proj, block.mlp.fc1, block.mlp.fc2):
            sigma = float(lucid.linalg.svdvals(linear.weight)[0].item())
            assert abs(sigma - 1.0) < 1e-4, sigma
        patch = model.patch_embed.weight
        sigma = float(lucid.linalg.svdvals(patch.reshape(patch.shape[0], -1))[0].item())
        assert abs(sigma - 1.0) < 1e-4, sigma

    def test_time_embedding_init_follows_the_config(self) -> None:
        """``"spectral"`` normalises the time MLPs; ``"normal"`` leaves 0.02."""
        spectral = TVMModel(_tiny(time_embedding_init="spectral"))
        normal = TVMModel(_tiny(time_embedding_init="normal"))
        sigma = float(lucid.linalg.svdvals(spectral.time_embed.fc1.weight)[0].item())
        assert abs(sigma - 1.0) < 1e-4
        std = float(normal.time_embed.fc1.weight.std().item())
        assert 0.015 < std < 0.025, std

    def test_modulation_norm_removes_the_modulation_scale(self) -> None:
        """Eq. 16: every shift, scale and gate is RMS-normalised.

        With the biases at zero, scaling a block's modulation projection
        scales every modulation vector, which the norm divides back out —
        the output must not move.  Without the norm it must, which is the
        unbounded Lipschitz constant the paper removes.
        """
        for normalized in (True, False):
            lucid.manual_seed(0)
            model = TVMModel(_tiny(modulation_norm=normalized)).eval()
            _wake(model)
            x, labels = lucid.randn((1, 4, 8, 8)), lucid.tensor([2])
            t, s = lucid.tensor([0.9]), lucid.tensor([0.3])
            before = model(x, t, s, labels).sample
            with lucid.no_grad():
                model.blocks[0].ada_ln[1].weight *= 10.0
            after = model(x, t, s, labels).sample
            if normalized:
                assert _gap(before, after) < 1e-4
            else:
                assert _gap(before, after) > 1e-3

    def test_the_field_reads_the_interval_and_the_guidance(self) -> None:
        """Holding ``t``, moving ``s`` or ``w`` must move the output."""
        model = TVMModel(_tiny(scaled_parameterization=False)).eval()
        _wake(model)
        x, labels, t = lucid.randn((1, 4, 8, 8)), lucid.tensor([2]), lucid.tensor([0.9])
        a = model(x, t, lucid.tensor([0.1]), labels, 2.0).sample
        b = model(x, t, lucid.tensor([0.5]), labels, 2.0).sample
        c = model(x, t, lucid.tensor([0.1]), labels, 1.5).sample
        assert _gap(a, b) > 1e-6, "the field ignores s"
        assert _gap(a, c) > 1e-6, "the field ignores w"

    def test_the_scaled_parameterization_is_a_factor_of_w(self) -> None:
        """Section 4: ``u = w F`` under the scaled parameterisation."""
        scaled = TVMModel(_tiny(scaled_parameterization=True)).eval()
        _wake(scaled)
        plain = TVMModel(_tiny(scaled_parameterization=False)).eval()
        plain.load_state_dict(scaled.state_dict())
        x, labels = lucid.randn((2, 4, 8, 8)), lucid.tensor([1, 4])
        t, s = lucid.tensor([0.8, 0.6]), lucid.tensor([0.2, 0.0])
        w = lucid.tensor([1.75, 2.5])
        gap = _gap(
            scaled(x, t, s, labels, w).sample,
            plain(x, t, s, labels, w).sample * w.reshape(-1, 1, 1, 1),
        )
        assert gap < 1e-5

    def test_no_label_is_the_null_label_at_unit_guidance(self) -> None:
        """``None`` is index ``num_classes`` and, by default, ``w = 1``."""
        model = TVMModel(_tiny()).eval()
        _wake(model)
        x, t, s = lucid.randn((1, 4, 8, 8)), lucid.tensor([1.0]), lucid.tensor([0.0])
        implicit = model(x, t, s).sample
        explicit = model(x, t, s, lucid.tensor([10]), 1.0).sample
        assert _gap(implicit, explicit) < 1e-6
        assert model.resolve_guidance(None, None) == 1.0
        assert model.resolve_guidance(None, lucid.tensor([3])) == 2.0


class TestObjective:
    @pytest.mark.parametrize("sampler", ["gap", "trunc", "clamp"])
    def test_s_never_exceeds_t(self, sampler: str) -> None:
        """Every sampler of Table 3 returns ``0 <= s <= t <= 1``."""
        model = TVMForImageGeneration(_tiny(time_sampler=sampler))
        t, s, s_fm = model._sample_times(256, "cpu")
        assert bool((s <= t).all().item())
        assert bool((s >= 0.0).all().item()) and bool((t <= 1.0).all().item())
        assert bool(((s_fm > 0.0) & (s_fm < 1.0)).all().item())

    def test_gap_star_draws_the_flow_matching_time_separately(self) -> None:
        """``gap*`` (the default) decouples the Flow Matching time from ``s``."""
        star = TVMForImageGeneration(_tiny(separate_flow_matching_time=True))
        _, s, s_fm = star._sample_times(64, "cpu")
        assert float((s != s_fm).float().mean().item()) > 0.95
        plain = TVMForImageGeneration(_tiny(separate_flow_matching_time=False))
        _, s, s_fm = plain._sample_times(64, "cpu")
        assert bool((s == s_fm).all().item())

    def test_guidance_range_draws_within_its_bounds(self) -> None:
        """Random guidance stays inside ``[w_min, w_max]``."""
        model = TVMForImageGeneration(_tiny(guidance_range=(1.5, 3.0)))
        w = model._sample_guidance(512, "cpu")
        assert float(w.min().item()) >= 1.5 - 1e-5
        assert float(w.max().item()) <= 3.0 + 1e-5

    def test_the_terminal_velocity_is_the_s_derivative_of_the_displacement(
        self,
    ) -> None:
        """Eq. 17: ``d/ds f = F + (s - t) ∂_s F`` along ``s`` alone.

        Checked against a central finite difference of the displacement
        ``f(s) = (s - t) u(x_t, t, s)``.  A JVP taken along ``t`` or
        ``x_t`` instead returns a finite quantity of the right shape and
        fails this.
        """
        model = TVMForImageGeneration(_tiny())
        _wake(model.tvm)
        x = lucid.randn((2, 4, 8, 8))
        t, s = lucid.tensor([0.9, 0.7]), lucid.tensor([0.4, 0.1])
        labels, w = lucid.tensor([1, 5]), lucid.tensor([2.0, 2.0])
        _, terminal = model._terminal_velocity(x, t, s, labels, w)

        def displacement(s_: lucid.Tensor) -> lucid.Tensor:
            u = model.tvm(x, t, s_, labels, w).sample
            return (s_ - t).reshape(-1, 1, 1, 1) * u

        h = 1e-2
        with lucid.no_grad():
            numeric = (displacement(s + h) - displacement(s - h)) / (2 * h)
        scale = float(numeric.abs().max().item())
        assert _gap(terminal.detach(), numeric) < 1e-2 * max(scale, 1.0)

    def test_the_gradient_runs_through_the_jvp(self) -> None:
        """Section 4: unlike MeanFlow, TVM back-propagates through ``∂_s F``.

        The same draws with :attr:`TVMConfig.detach_jvp` set must give a
        different gradient — if they agree, the JVP was detached all along.
        """
        grads = []
        for detach in (False, True):
            lucid.manual_seed(0)
            model = TVMForImageGeneration(_tiny(detach_jvp=detach, target_ema_decay=0.0))
            _wake(model.tvm)
            lucid.manual_seed(1)
            out = model(lucid.randn((2, 4, 8, 8)), lucid.tensor([1, 3]))
            out.loss.backward()
            grads.append(model.tvm.blocks[0].attn.qkv.weight.grad)
        assert grads[0] is not None and grads[1] is not None
        assert _gap(grads[0], grads[1]) > 1e-8

    def test_every_parameter_receives_a_finite_gradient(self) -> None:
        """One step reaches the whole network, the guidance MLP included."""
        model = TVMForImageGeneration(_tiny())
        out = model(lucid.randn((2, 4, 8, 8)), lucid.tensor([1, 3]))
        assert out.loss.ndim == 0 and bool(out.loss.isfinite().item())
        out.loss.backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, missing
        broken = [
            n
            for n, p in model.named_parameters()
            if p.grad is not None and not bool(p.grad.isfinite().all().item())
        ]
        assert not broken, broken

    def test_s_equals_t_leaves_only_flow_matching(self) -> None:
        """The indicator in Eq. 15: at ``s = t`` the terminal term is zero."""
        model = TVMForImageGeneration(_tiny(ratio_t_equals_s=1.0))
        _wake(model.tvm)
        out = model(lucid.randn((2, 4, 8, 8)), lucid.tensor([1, 3]))
        assert float(out.terminal_loss.item()) == 0.0
        assert float(out.flow_matching_loss.item()) > 0.0


class TestTarget:
    def test_the_target_is_not_part_of_the_model(self) -> None:
        """θ* is training state: not a parameter, not in the state dict."""
        model = TVMForImageGeneration(_tiny())
        before = sum(1 for _ in model.parameters())
        keys = set(model.state_dict())
        target = model.target_network
        assert target is not model.tvm
        assert sum(1 for _ in model.parameters()) == before
        assert set(model.state_dict()) == keys
        assert not any(p.requires_grad for p in target.parameters())

    def test_the_target_starts_at_the_live_weights(self) -> None:
        """Algorithm 1 initialises θ* ← θ."""
        model = TVMForImageGeneration(_tiny())
        live = dict(model.tvm.named_parameters())
        for name, param in model.target_network.named_parameters():
            assert _gap(param, live[name]) == 0.0, name

    def test_update_ema_steps_both_averages(self) -> None:
        """Algorithm 1: θ* and θ** each move by their own rate, per step."""
        model = TVMForImageGeneration(_tiny(target_ema_decay=0.9, eval_ema_decay=0.5))
        target = model.target_network.blocks[0].attn.qkv.weight.detach().clone()
        evaluated = model.ema_network.blocks[0].attn.qkv.weight.detach().clone()
        with lucid.no_grad():
            model.tvm.blocks[0].attn.qkv.weight += 1.0
        live = model.tvm.blocks[0].attn.qkv.weight.detach().clone()
        model.update_ema()
        assert (
            _gap(model.target_network.blocks[0].attn.qkv.weight, 0.9 * target + 0.1 * live)
            < 1e-6
        )
        assert (
            _gap(model.ema_network.blocks[0].attn.qkv.weight, 0.5 * evaluated + 0.5 * live)
            < 1e-6
        )

    def test_no_eval_average_is_kept_when_disabled(self) -> None:
        """``eval_ema_decay=None`` keeps no θ** and says so when asked."""
        model = TVMForImageGeneration(_tiny(eval_ema_decay=None))
        with pytest.raises(ValueError):
            _ = model.ema_network

    def test_zero_decay_targets_the_live_network(self) -> None:
        """Table 4b's γ = 0: no copy, the live weights under a stop-gradient."""
        model = TVMForImageGeneration(_tiny(target_ema_decay=0.0))
        assert model.target_network is model.tvm


class TestSampling:
    def test_one_step_is_one_jump(self) -> None:
        """Figure 3 at one step: ``x_0 = x_1 + (0 - 1) u``."""
        config = _tiny()

        class _Ones(TVMModel):
            def forward(
                self,
                x: lucid.Tensor,
                t: lucid.Tensor,
                s: lucid.Tensor,
                labels: lucid.Tensor | None = None,
                guidance_scale: lucid.Tensor | float | None = None,
            ) -> DiffusionModelOutput:
                return DiffusionModelOutput(sample=lucid.ones_like(x))

        model = TVMForImageGeneration(config)
        model.tvm = _Ones(config)
        noise = lucid.randn((2, 4, 8, 8))
        out = model.generate(2, noise=noise).samples
        assert _gap(out, noise - 1.0) < 1e-6

    def test_steps_walk_an_even_grid_from_one_to_zero(self) -> None:
        """``linspace(1, 0, n + 1)`` — each step jumps between neighbours."""
        config = _tiny()
        seen: list[tuple[float, float]] = []

        class _Recorder(TVMModel):
            def forward(
                self,
                x: lucid.Tensor,
                t: lucid.Tensor,
                s: lucid.Tensor,
                labels: lucid.Tensor | None = None,
                guidance_scale: lucid.Tensor | float | None = None,
            ) -> DiffusionModelOutput:
                seen.append((float(t[0].item()), float(s[0].item())))
                return DiffusionModelOutput(sample=lucid.zeros_like(x))

        model = TVMForImageGeneration(config)
        model.tvm = _Recorder(config)
        model.generate(1, steps=4)
        expected = [(1.0, 0.75), (0.75, 0.5), (0.5, 0.25), (0.25, 0.0)]
        assert all(
            math.isclose(a, c, abs_tol=1e-6) and math.isclose(b, d, abs_tol=1e-6)
            for (a, b), (c, d) in zip(seen, expected, strict=True)
        ), seen

    def test_to_vae_latent_inverts_the_whitening(self) -> None:
        """``ℓ = z σ + μ`` per channel, and ``from_vae_latent`` undoes it."""
        model = TVMForImageGeneration(_tiny())
        raw = model.to_vae_latent(lucid.zeros((1, 4, 8, 8)))
        mean = lucid.tensor(list(model.config.latent_mean)).reshape(1, -1, 1, 1)
        assert _gap(raw, mean + lucid.zeros((1, 4, 8, 8))) < 1e-6
        latent = lucid.randn((2, 4, 8, 8)) * 5.0
        assert _gap(model.to_vae_latent(model.from_vae_latent(latent)), latent) < 1e-4

    def test_use_ema_samples_from_the_evaluation_average(self) -> None:
        """``generate(use_ema=True)`` reads θ**, not the live weights."""
        model = TVMForImageGeneration(_tiny(eval_ema_decay=0.5)).eval()
        _wake(model.tvm)
        _ = model.ema_network  # θ** ← θ
        with lucid.no_grad():
            model.tvm.final.proj.weight += 1.0
        noise = lucid.randn((1, 4, 8, 8))
        live = model.generate(1, noise=noise).samples
        averaged = model.generate(1, noise=noise, use_ema=True).samples
        assert _gap(live, averaged) > 1e-6


class TestFactories:
    @pytest.mark.parametrize("factory", [tvm_imagenet256, tvm_imagenet512])
    def test_the_network_is_tables_678m(self, factory: object) -> None:
        """Table 6: 678M parameters at both resolutions.

        Counted under ``shadow_alloc`` so no gigabytes are touched.  The
        count is the paper's DiT-XL/2 with 18 heads, three scalar
        embedders and the null label row: 678,068,496.
        """
        with shadow_alloc():
            model = factory()  # type: ignore[operator]
        count = sum(math.prod(p.shape) for p in model.parameters())
        assert count == 678_068_496, count

    def test_pretrained_is_refused_until_converted(self) -> None:
        """No hosted checkpoint yet, so ``pretrained=True`` must raise."""
        with pytest.raises(NotImplementedError):
            tvm_imagenet256(pretrained=True)
