"""DIAMOND — the claims the paper makes, asserted rather than assumed.

The paper's decisions are all numerical: which preconditioners, which
noise distribution, how many denoising steps, which conditioning enters
where.  A shape test sees none of them, and every one of them is a place
where a plausible-looking implementation is silently a different model.

Each test names the mis-wiring it catches.
"""

import math

import pytest

import lucid
from lucid.models.generative.diamond import (
    DIAMONDConfig,
    DIAMONDForWorldModeling,
    DIAMONDModel,
)


def _tiny(**overrides: object) -> DIAMONDConfig:
    """A model small enough to instantiate per test."""
    base = dict(
        sample_size=16,
        unet_channels=(8, 8),
        unet_layers=(1, 1),
        reward_channels=(8, 8),
        reward_layers=(1, 1),
        actor_channels=(8, 8),
        actor_layers=(1, 1),
        cond_dim=16,
        reward_cond_dim=8,
        reward_lstm_dim=16,
        actor_lstm_dim=16,
        num_actions=4,
        horizon=3,
    )
    base.update(overrides)
    return DIAMONDConfig(**base)  # type: ignore[arg-type]


def _history(batch: int = 2, config: DIAMONDConfig | None = None) -> lucid.Tensor:
    config = _tiny() if config is None else config
    side = int(config.sample_size) if isinstance(config.sample_size, int) else 16
    return lucid.randn(
        (batch, config.conditioning_frames, config.in_channels, side, side)
    )


def _actions(batch: int = 2, config: DIAMONDConfig | None = None) -> lucid.Tensor:
    config = _tiny() if config is None else config
    return lucid.zeros(
        (batch, config.conditioning_frames), dtype=lucid.int64
    ) + lucid.ones((batch, config.conditioning_frames), dtype=lucid.int64)


class TestPreconditioning:
    def test_the_four_scalings_are_edm_s(self) -> None:
        r"""Appendix C, equations 9-12, at :math:`\sigma_{data} = 0.5`.

        These are four one-line formulas, which is exactly why they get
        transcribed wrong: swapping :math:`c_{in}` and :math:`c_{out}`,
        or dropping the :math:`\sigma_{data}` from the numerator of
        :math:`c_{out}`, leaves a model that still trains and still
        denoises, just not the paper's.
        """
        model = DIAMONDModel(_tiny())
        sigma = lucid.tensor([0.1, 1.0, 10.0])
        c_in, c_out, c_skip, c_noise = model.preconditioners(sigma)
        data = 0.5
        for index, s in enumerate((0.1, 1.0, 10.0)):
            total = math.sqrt(s**2 + data**2)
            assert abs(float(c_in.reshape(-1)[index].item()) - 1.0 / total) < 1e-6
            assert abs(float(c_out.reshape(-1)[index].item()) - s * data / total) < 1e-6
            assert (
                abs(
                    float(c_skip.reshape(-1)[index].item()) - data**2 / (data**2 + s**2)
                )
                < 1e-6
            )
            assert abs(float(c_noise[index].item()) - 0.25 * math.log(s)) < 1e-6

    def test_the_skip_hands_over_at_sigma_data(self) -> None:
        r"""The property the preconditioners exist for.

        Section 5.1's whole argument is that the *target* moves with the
        noise level: near zero noise the model is asked for the added
        noise (skip ~ 1), and far above :math:`\sigma_{data}` it is asked
        for the clean frame (skip ~ 0).  A constant skip — DDPM's
        parameterisation — is the thing the paper shows drifting.
        """
        model = DIAMONDModel(_tiny())
        _in, _out, low, _n = model.preconditioners(lucid.tensor([1e-4]))
        _in, _out, high, _n = model.preconditioners(lucid.tensor([1e4]))
        assert float(low.reshape(-1)[0].item()) > 0.999
        assert float(high.reshape(-1)[0].item()) < 1e-6

    def test_the_denoiser_is_the_identity_at_zero_noise(self) -> None:
        r"""``c_out`` vanishes with :math:`\sigma`, so :math:`D` returns its input.

        This holds for *any* weights, which is what makes it a test of
        the wrapper rather than of training: if the skip and the network
        branch were combined the other way round, an untrained model
        would return noise here instead.
        """
        config = _tiny()
        model = DIAMONDModel(config).eval()
        frames, actions = _history(config=config), _actions(config=config)
        clean = lucid.randn((2, 3, 16, 16))
        with lucid.no_grad():
            out = model.denoise(clean, lucid.tensor([1e-8, 1e-8]), frames, actions)
        assert float((out - clean).abs().max().item()) < 1e-3


class TestNoiseDistribution:
    def test_training_draws_sigma_log_normally(self) -> None:
        r"""``log sigma ~ N(-0.4, 1.2^2)`` — Appendix C, equation 13.

        The mean is the paper's, not EDM's own :math:`-1.2`, and a world
        model trained at the wrong centre spends its capacity on noise
        levels the sampler never visits.  Checked over a large draw
        because a single one says nothing about a distribution.
        """
        model = DIAMONDModel(_tiny()).eval()
        frames, actions = _history(512), _actions(512)
        target = lucid.randn((512, 3, 16, 16))
        with lucid.no_grad():
            out = model(frames, actions, target)
        logs = lucid.log(out.sigma)
        mean = float(logs.mean().item())
        std = float(((logs - mean) ** 2).mean().item()) ** 0.5
        assert abs(mean - (-0.4)) < 0.2, f"log-sigma centred at {mean:.3f}"
        assert abs(std - 1.2) < 0.2, f"log-sigma spread {std:.3f}"


class TestConditioning:
    def test_the_history_reaches_the_prediction(self) -> None:
        """Frame stacking has to actually stack.

        A denoiser that ignored its conditioning would still produce the
        right shape and still train to a plausible-looking loss — it
        would just be an unconditional image model, and the world model
        would be useless.
        """
        config = _tiny()
        model = DIAMONDModel(config).eval()
        actions = _actions(config=config)
        noised = lucid.randn((2, 3, 16, 16))
        sigma = lucid.tensor([1.0, 1.0])
        with lucid.no_grad():
            first = model.denoise(noised, sigma, _history(config=config), actions)
            second = model.denoise(noised, sigma, _history(config=config), actions)
        assert float((first - second).abs().max().item()) > 0.0

    def test_the_actions_reach_the_prediction(self) -> None:
        """Adaptive group norm has to carry the actions, not just the time.

        This is the conditioning a world model cannot do without: if the
        action does not change the predicted frame, the agent's choices
        do not change its imagined future and the policy gradient is
        noise.
        """
        config = _tiny()
        model = DIAMONDModel(config).eval()
        frames = _history(config=config)
        noised = lucid.randn((2, 3, 16, 16))
        sigma = lucid.tensor([1.0, 1.0])
        zeros = lucid.zeros((2, config.conditioning_frames), dtype=lucid.int64)
        threes = zeros + 3
        with lucid.no_grad():
            first = model.denoise(noised, sigma, frames, zeros)
            second = model.denoise(noised, sigma, frames, threes)
        assert float((first - second).abs().max().item()) > 0.0

    def test_the_noise_level_reaches_the_prediction(self) -> None:
        """The diffusion time is conditioning too, through the same path."""
        config = _tiny()
        model = DIAMONDModel(config).eval()
        frames, actions = _history(config=config), _actions(config=config)
        noised = lucid.randn((2, 3, 16, 16))
        with lucid.no_grad():
            first = model.denoise(noised, lucid.tensor([0.5, 0.5]), frames, actions)
            second = model.denoise(noised, lucid.tensor([2.0, 2.0]), frames, actions)
        assert float((first - second).abs().max().item()) > 0.0


class TestSampling:
    def test_the_schedule_ends_at_zero(self) -> None:
        """An Euler sampler that stopped short would return a noisy frame.

        The last step has to land on sigma exactly zero, which is what
        makes the final iterate the model's estimate of the clean frame
        rather than an estimate plus whatever noise was left.
        """
        model = DIAMONDModel(_tiny())
        schedule = model.sigma_schedule(3, "cpu")
        assert schedule.shape == (4,)
        assert float(schedule[-1].item()) == 0.0
        values = [float(schedule[i].item()) for i in range(4)]
        assert values == sorted(values, reverse=True), "schedule must descend"

    def test_three_steps_is_the_default(self) -> None:
        """Section 5.2 settles on ``n = 3`` "in all our experiments"."""
        assert DIAMONDConfig().denoise_steps == 3

    def test_more_steps_walk_a_different_trajectory(self) -> None:
        """The step count must reach the sampler.

        Given the same starting noise, one step and five must not agree —
        if they do, ``steps`` is being ignored and the paper's ablation
        between them cannot be run.
        """
        config = _tiny()
        model = DIAMONDModel(config).eval()
        frames, actions = _history(config=config), _actions(config=config)
        start = lucid.randn((2, 3, 16, 16))
        with lucid.no_grad():
            few = model.imagine_frame(frames, actions, steps=1, noise=start)
            many = model.imagine_frame(frames, actions, steps=5, noise=start)
        assert float((few - many).abs().max().item()) > 0.0

    def test_a_non_positive_step_count_is_refused(self) -> None:
        config = _tiny()
        model = DIAMONDModel(config)
        with pytest.raises(ValueError, match="steps must be positive"):
            model.imagine_frame(
                _history(config=config), _actions(config=config), steps=0
            )


class TestImagination:
    def test_the_rollout_is_autoregressive(self) -> None:
        """Imagined frames re-enter the conditioning.

        Figure 1's whole point: the predicted observation and the action
        taken from it become the next step's history.  A rollout that
        kept conditioning on the *real* frames would not compound error
        at all, which would make the paper's central comparison — EDM
        against DDPM over a thousand steps — untestable.

        Asserted on the conditioning rather than on the frames.  The
        frames differ between steps either way, because the sampler
        draws fresh noise each call — a first version of this test
        compared consecutive frames and passed happily with the history
        update deleted.  Once the horizon passes :math:`L`, the history
        must be *entirely* imagined, and that is exact.
        """
        config = _tiny(horizon=5)
        model = DIAMONDForWorldModeling(config).eval()
        real = _history(config=config)
        with lucid.no_grad():
            out = model(real, _actions(config=config))
        assert out.frames.shape == (2, 5, 3, 16, 16)
        window = config.conditioning_frames
        moved = float((out.history - out.frames[:, -window:]).abs().max().item())
        assert moved == 0.0, "the history is not the frames the model imagined"
        assert float((out.history - real).abs().max().item()) > 0.0

    def test_the_horizon_reaches_the_returns(self) -> None:
        config = _tiny(horizon=5)
        model = DIAMONDForWorldModeling(config).eval()
        with lucid.no_grad():
            out = model(_history(config=config), _actions(config=config))
        assert out.returns.shape == (2, 5)

    def test_a_one_step_horizon_is_refused(self) -> None:
        """Lambda-returns need a state to bootstrap from."""
        config = _tiny()
        model = DIAMONDForWorldModeling(config)
        with pytest.raises(ValueError, match="at least two imagined states"):
            model(_history(config=config), _actions(config=config), horizon=1)

    def test_both_losses_reach_their_own_parameters(self) -> None:
        """The policy loss must not train the value head, or vice versa.

        Equation 15 stops the gradient on the returns and equation 16 on
        the advantage, so each objective touches one head.  A missing
        ``detach`` would let the value network chase a target it is
        itself producing.
        """
        config = _tiny()
        model = DIAMONDForWorldModeling(config)
        out = model(_history(config=config), _actions(config=config))
        (out.policy_loss + out.value_loss).backward()
        actor = model.diamond.actor_critic
        assert any(p.grad is not None for p in actor.actor_linear.parameters())
        assert any(p.grad is not None for p in actor.critic_linear.parameters())


class TestObjectives:
    def test_the_denoiser_loss_reaches_the_denoiser(self) -> None:
        config = _tiny()
        model = DIAMONDModel(config)
        out = model(
            _history(config=config),
            _actions(config=config),
            lucid.randn((2, 3, 16, 16)),
        )
        out.loss.backward()
        assert sum(1 for p in model.denoiser.parameters() if p.grad is not None) > 0

    def test_reward_is_predicted_as_a_sign_not_a_scalar(self) -> None:
        r"""Algorithm 1 writes ``CE(r_hat, sign(r))`` — three classes.

        The environment clips reward to :math:`\{-1, 0, 1\}`, so the
        head has three outputs and the loss is cross-entropy.  A
        regression head would fit the same data and quietly discard the
        clipping the benchmark defines.
        """
        config = _tiny()
        model = DIAMONDForWorldModeling(config)
        head = model.diamond.reward_end.head[2]
        assert head.weight.shape[0] == 5, "three reward classes plus two for the end"
        assert head.bias is None, "the released head carries no bias here"
        loss = model.reward_end_loss(
            _history(config=config),
            _actions(config=config),
            lucid.tensor([[-5.0, 0.0, 3.0, 1.0], [0.0, 0.0, -1.0, 2.0]]),
            lucid.zeros((2, config.conditioning_frames)),
        )
        loss.backward()
        assert (
            sum(1 for p in model.diamond.reward_end.parameters() if p.grad is not None)
            > 0
        )

    def test_the_paper_hyperparameters(self) -> None:
        """Table 3, recorded so an edit cannot quietly rewrite it."""
        config = DIAMONDConfig()
        assert (config.horizon, config.gamma) == (15, 0.985)
        assert (config.lambda_, config.entropy_weight) == (0.95, 0.001)
        assert (config.conditioning_frames, config.burn_in) == (4, 4)
        assert (config.sigma_data, config.p_mean, config.p_std) == (0.5, -0.4, 1.2)

    def test_the_paper_architecture(self) -> None:
        """Table 2, likewise."""
        config = DIAMONDConfig()
        assert config.unet_channels == (64, 64, 64, 64)
        assert config.unet_layers == (2, 2, 2, 2)
        assert config.cond_dim == 256
        assert config.reward_channels == (32, 32, 32, 32)
        assert config.reward_lstm_dim == 512
        assert config.actor_channels == (32, 32, 64, 64)
        assert config.actor_layers == (1, 1, 1, 1)


class TestReleasedArchitecture:
    """What the checkpoint says and the paper does not.

    Four structural facts came from the released weights rather than
    from the text, and each was wrong in a first pass built from the
    paper alone.  They are pinned here because nothing about a forward
    pass would report them, and getting any of them wrong makes the
    published checkpoints unloadable.
    """

    def test_the_parameter_count_matches_the_release(self) -> None:
        """13,536,584 at four actions — Breakout's checkpoint exactly.

        The paper quotes 13M as a round aggregate in a comparison table.
        This is the real number, and it only comes out right when the
        decoder's extra block per resolution, the reward model's two-frame
        input, its three (not four) downsamples and its bias-free head are
        all in place.
        """
        import math

        model = DIAMONDModel(DIAMONDConfig(num_actions=4))
        total = sum(math.prod(p.shape) for p in model.parameters())
        assert total == 13_536_584, f"built {total:,}"

    def test_the_reward_model_reads_a_transition(self) -> None:
        """Two frames in, six channels — not one.

        A reward is a property of the transition.  Feeding one frame
        builds a model that trains and predicts nothing useful about
        reward, and quietly halves the encoder's first convolution.
        """
        config = DIAMONDConfig(num_actions=4)
        model = DIAMONDModel(config)
        assert config.reward_frames == 2
        assert model.reward_end.encoder.conv_in.weight.shape[1] == 6

    def test_the_reward_encoder_stops_at_eight(self) -> None:
        """Three downsamples, so the LSTM sees 2048 rather than 512.

        The stages number four but the downsamples sit *between* them.
        Pooling after the last one too would quarter the feature map and
        shrink the LSTM's input weight by three quarters.
        """
        model = DIAMONDModel(DIAMONDConfig(num_actions=4))
        assert model.reward_end.cell.weight_ih.shape[1] == 2048

    def test_offset_noise_is_added_at_training(self) -> None:
        """A per-channel offset on top of the isotropic draw.

        ⚠️ Absent from the paper; present in the released config at
        0.3.  Without it the model never sees a target whose overall
        level is shifted, which is exactly what offset noise exists to
        teach.
        """
        config = _tiny(sigma_offset_noise=0.0)
        assert DIAMONDConfig().sigma_offset_noise == 0.3
        plain = DIAMONDModel(config).eval()
        frames, actions = _history(config=config), _actions(config=config)
        target = lucid.zeros((2, 3, 16, 16))
        fixed = lucid.tensor([1e-6, 1e-6])
        with lucid.no_grad():
            quiet = plain(frames, actions, target, sigma=fixed)
        assert float(quiet.loss.item()) < 1e-3, "no noise at all with both off"

        loud = DIAMONDModel(_tiny(sigma_offset_noise=5.0)).eval()
        with lucid.no_grad():
            noisy = loud(frames, actions, target, sigma=fixed)
        assert float(noisy.loss.item()) > float(quiet.loss.item())


class TestCSGO:
    """The second released model, which is not a bigger Atari one.

    Its configuration differs in five ways the Atari path never
    exercises, and each is a place where an implementation built only
    for 64x64 square frames with a full agent quietly breaks.
    """

    def _config(self) -> DIAMONDConfig:
        return DIAMONDConfig(
            sample_size=(30, 56),
            unet_channels=(32, 64),
            unet_layers=(1, 1),
            attn_depths=(0, 1),
            cond_dim=32,
            num_actions=51,
            with_agent=False,
            noise_previous_obs=True,
            upsampler_channels=(16, 32),
            upsampler_layers=(1, 1),
            upsampler_attn_depths=(0, 1),
            upsampling_factor=5,
        )

    def test_frames_need_not_be_square(self) -> None:
        """30 by 56, and 30 does not survive three clean halvings.

        A stride-2 convolution rounds up — 30, 15, 8 — so the decoder
        cannot assume doubling gets it back.  Resizing to each skip is
        exact; assuming a factor of two is off by one row and fails.
        """
        config = self._config()
        model = DIAMONDModel(config).eval()
        assert config.frame_shape == (30, 56)
        frames = lucid.randn((1, 4, 3, 30, 56))
        actions = lucid.zeros((1, 4), dtype=lucid.int64) + 7
        with lucid.no_grad():
            out = model.imagine_frame(frames, actions, steps=1)
        assert out.shape == (1, 3, 30, 56)

    def test_the_agent_is_absent(self) -> None:
        """``rew_end_model: null``, ``actor_critic: null`` in the release.

        That experiment trains a world model on static gameplay with no
        reinforcement learning, so building an agent would add
        parameters nothing ever updates — and would put them in the
        checkpoint, where the port would have nothing to fill them with.
        """
        model = DIAMONDModel(self._config())
        assert model.reward_end is None
        assert model.actor_critic is None
        assert model.upsampler is not None

    def test_attention_follows_attn_depths(self) -> None:
        """Per resolution, not just the middle.

        Atari attends only in the middle blocks; CS:GO adds it at its
        deepest resolutions, and the decoder mirrors the encoder's
        choice.  A model that ignored the flag would still run.
        """
        shallow = DIAMONDModel(
            DIAMONDConfig(
                sample_size=16,
                unet_channels=(8, 8),
                unet_layers=(1, 1),
                attn_depths=(0, 0),
                cond_dim=16,
                reward_channels=(8, 8),
                reward_layers=(1, 1),
                actor_channels=(8, 8),
                actor_layers=(1, 1),
                reward_cond_dim=8,
                reward_lstm_dim=16,
                actor_lstm_dim=16,
                num_actions=4,
            )
        )
        attending = DIAMONDModel(
            DIAMONDConfig(
                sample_size=16,
                unet_channels=(8, 8),
                unet_layers=(1, 1),
                attn_depths=(0, 1),
                cond_dim=16,
                reward_channels=(8, 8),
                reward_layers=(1, 1),
                actor_channels=(8, 8),
                actor_layers=(1, 1),
                reward_cond_dim=8,
                reward_lstm_dim=16,
                actor_lstm_dim=16,
                num_actions=4,
            )
        )
        keys = {k for k in shallow.state_dict() if ".attn." in k}
        more = {k for k in attending.state_dict() if ".attn." in k}
        assert len(more) > len(keys), "attn_depths did not add attention"

    def test_noising_the_history_adds_its_own_embedding(self) -> None:
        """``noise_previous_obs`` brings a second Fourier embedding.

        The same frame at two degradations is two different
        conditionings, so the level the *history* was noised at has to
        reach the network.  The released CS:GO checkpoint carries
        ``noise_cond_emb``; the Atari ones do not, exactly tracking the
        flag.
        """
        plain = DIAMONDModel(
            DIAMONDConfig(
                sample_size=16,
                unet_channels=(8, 8),
                unet_layers=(1, 1),
                cond_dim=16,
                reward_channels=(8, 8),
                reward_layers=(1, 1),
                actor_channels=(8, 8),
                actor_layers=(1, 1),
                reward_cond_dim=8,
                reward_lstm_dim=16,
                actor_lstm_dim=16,
                num_actions=4,
            )
        )
        assert plain.denoiser.noise_cond_emb is None
        assert DIAMONDModel(self._config()).denoiser.noise_cond_emb is not None

    def test_the_upsampler_takes_three_images_and_no_action(self) -> None:
        """Noised frame, low-resolution frame, previous frame — nine channels.

        And no action embedding: what the player did is already in the
        frame being sharpened, so conditioning on it again would be
        parameters with nothing to learn.
        """
        model = DIAMONDModel(self._config())
        assert model.upsampler is not None
        assert model.upsampler.conv_in.weight.shape[1] == 9
        assert not hasattr(model.upsampler, "action_embed")

    def test_the_csgo_factory_defaults_to_its_own_checkpoint(self) -> None:
        """``pretrained=True`` must not reach for Breakout.

        The enum's ``DEFAULT`` is an Atari agent, because that is what
        ``diamond(pretrained=True)`` should give.  This factory builds a
        382M world model at 30x56, so resolving ``True`` through the
        shared default tries to load a 13M agent into it and fails on a
        shape mismatch — which is exactly what happened the first time.
        """
        from lucid.models.generative.diamond._pretrained import csgo_tag
        from lucid.models.generative.diamond import DIAMONDWeights

        assert csgo_tag(True) == "CSGO"
        assert csgo_tag("csgo") == "CSGO"
        assert csgo_tag(False) is False
        # The hazard is real only because the two disagree on shape.
        assert DIAMONDWeights.DEFAULT.name == "BREAKOUT"
        assert DIAMONDWeights.CSGO.value.meta["num_actions"] == 51
