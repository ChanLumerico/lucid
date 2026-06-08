"""Shared next-token selection primitives for autoregressive generation.

These are the cache- and family-agnostic building blocks of every
:meth:`generate` loop in the model zoo — :class:`CausalLMMixin` (decoder-only)
and ``TransformerForSeq2SeqLM`` (encoder-decoder) both drive their token
selection through :func:`_select_and_append_next`, so greedy / temperature /
top-k / top-p / repetition-penalty sampling behaves identically regardless of
model family or cache implementation.

Every primitive is fully vectorised (no per-element CPU round-trip): the logit
filters and the multinomial draw are single fused tensor expressions, so
``do_sample`` generation costs a few tensor ops per step rather than an
``O(B x vocab)`` Python loop.
"""

from dataclasses import dataclass

import lucid
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor


@dataclass(frozen=True, slots=True)
class _SamplingParams:
    r"""Bundle of next-token selection knobs shared by the eager and compiled
    decode loops (so both paths sample identically)."""

    do_sample: bool
    temperature: float
    top_k: int | None
    top_p: float | None
    repetition_penalty: float
    eos_token_id: int | None
    pad_token_id: int
    dev: str


def _select_and_append_next(
    next_logits: Tensor,
    out_tokens: list[Tensor],
    finished: list[bool],
    params: _SamplingParams,
) -> bool:
    r"""Choose the next token row, append it, and report whether decoding is done.

    Applies the repetition penalty (over the running prefix), then greedy
    ``argmax`` or temperature / top-k / top-p sampling; substitutes
    ``pad_token_id`` for already-finished rows and flips ``finished`` on EOS.
    Mutates ``out_tokens`` (appends the new ``(B,)`` row) and ``finished`` in
    place.

    Parameters
    ----------
    next_logits : Tensor
        ``(B, vocab)`` logits at the current step.
    out_tokens : list[Tensor]
        Running list of ``(B,)`` token rows; the chosen row is appended.
    finished : list[bool]
        Per-row completion flags, updated in place.
    params : _SamplingParams
        Shared sampling configuration.

    Returns
    -------
    bool
        ``True`` once every row has emitted EOS (caller breaks the loop).
    """
    B = int(next_logits.shape[0])
    if params.repetition_penalty != 1.0:
        prefix = lucid.stack(out_tokens, dim=1)
        next_logits = _apply_repetition_penalty(
            next_logits, prefix, params.repetition_penalty
        )

    if not params.do_sample:
        next_tok = lucid.argmax(next_logits, dim=-1)  # (B,)
    else:
        if params.temperature != 1.0:
            next_logits = next_logits / params.temperature
        if params.top_k is not None:
            next_logits = _top_k_filter(next_logits, params.top_k)
        if params.top_p is not None:
            next_logits = _top_p_filter(next_logits, params.top_p)
        probs = F.softmax(next_logits, dim=-1)  # (B, vocab)
        next_tok = _multinomial_one(probs, device=params.dev)

    next_list: list[int] = [int(next_tok[b].item()) for b in range(B)]
    for b in range(B):
        if finished[b]:
            next_list[b] = params.pad_token_id
        elif params.eos_token_id is not None and next_list[b] == params.eos_token_id:
            finished[b] = True
    out_tokens.append(lucid.tensor(next_list, device=params.dev).long())
    return all(finished)


def _apply_repetition_penalty(logits: Tensor, prefix: Tensor, penalty: float) -> Tensor:
    r"""Multiply logits of tokens already in ``prefix`` by ``1 / penalty``
    (or ``penalty`` if the logit is negative).

    Parameters
    ----------
    logits : Tensor
        ``(B, vocab)`` next-token logits.
    prefix : Tensor
        ``(B, T)`` int tokens generated so far.
    penalty : float
        Strictly positive float; ``> 1`` discourages repetition.

    Returns
    -------
    Tensor
        Adjusted ``(B, vocab)`` logits.

    Notes
    -----
    Convention popularised by the wider ML ecosystem: positive logits
    shrink toward 0, negative logits grow more negative — both directions
    push the affected token's probability down.

    Fully vectorised (no per-element CPU round-trip): the seen-token mask is
    built by scattering the prefix ids into a ``(B, vocab)`` indicator, then the
    penalty is applied with two element-wise ``where``s.
    """
    B = int(logits.shape[0])
    vocab = int(logits.shape[1])
    dev = logits.device.type
    # (B, vocab) indicator of tokens already present in the prefix.
    seen = lucid.scatter(
        lucid.zeros((B, vocab), device=dev),
        -1,
        prefix,
        lucid.ones_like(prefix).float(),
    )
    penalized = lucid.where(logits > 0, logits / penalty, logits * penalty)
    return lucid.where(seen > 0, penalized, logits)


def _top_k_filter(logits: Tensor, k: int) -> Tensor:
    r"""Set every logit outside the per-row top-K to ``-inf``.

    Parameters
    ----------
    logits : Tensor
        ``(B, vocab)`` logits.
    k : int
        Number of tokens to retain per row.  If ``k >= vocab``, ``logits``
        is returned unchanged.

    Returns
    -------
    Tensor
        Masked logits where non-top-K entries are replaced with a very
        large negative number (``-1e9``).
    """
    B = int(logits.shape[0])
    vocab = int(logits.shape[1])
    if k >= vocab:
        return logits
    # The k-th largest logit per row is the keep threshold; mask everything
    # strictly below it.  Vectorised — one ``topk`` + one ``where``.
    values, _ = lucid.topk(logits, k, dim=-1)
    threshold = values[:, k - 1 : k].broadcast_to((B, vocab))
    return lucid.where(logits >= threshold, logits, lucid.full_like(logits, -1e9))


def _top_p_filter(logits: Tensor, p: float) -> Tensor:
    r"""Nucleus (top-p) filtering — keep the smallest token set with
    cumulative softmax probability ≥ ``p``.

    Parameters
    ----------
    logits : Tensor
        ``(B, vocab)`` logits.
    p : float
        Cumulative probability threshold in ``(0, 1]``.

    Returns
    -------
    Tensor
        Masked logits where tokens outside the nucleus are set to
        ``-1e9``.

    Notes
    -----
    Vectorised: sort descending, take the cumulative softmax, keep every token
    whose *exclusive-prefix* cumulative probability is ``< p`` (so the token
    that crosses the threshold is itself kept, and the highest-probability token
    is always kept), then scatter the mask back to the original token order via
    the inverse permutation.
    """
    sorted_vals = lucid.flip(lucid.sort(logits, dim=-1), -1)
    sorted_idx = lucid.flip(lucid.argsort(logits, dim=-1), -1)
    sorted_probs = F.softmax(sorted_vals, dim=-1)
    exclusive_cdf = lucid.cumsum(sorted_probs, dim=-1) - sorted_probs
    keep = exclusive_cdf < p
    masked = lucid.where(keep, sorted_vals, lucid.full_like(sorted_vals, -1e9))
    # Undo the sort: gather by the inverse permutation back to token order.
    inverse = lucid.argsort(sorted_idx, dim=-1)
    return lucid.gather(masked, inverse, dim=-1)


def _multinomial_one(probs: Tensor, *, device: str) -> Tensor:
    r"""Draw exactly one token per row from a categorical distribution.

    Parameters
    ----------
    probs : Tensor
        ``(B, vocab)`` non-negative probability tensor — rows should sum
        to 1.
    device : str, keyword-only
        Device string to allocate the random draws on.

    Returns
    -------
    Tensor
        ``(B,)`` long tensor of sampled token ids.

    Notes
    -----
    Vectorised inverse-CDF on a single uniform draw per row: the sampled index
    is the count of CDF entries below the draw — ``(cumsum(probs) < u).sum()`` —
    computed entirely on device.  Same one-draw-per-row distribution as the
    scalar loop (bit-identical given the same RNG state), without the
    per-element CPU round-trip.
    """
    B = int(probs.shape[0])
    vocab = int(probs.shape[1])
    u = lucid.rand((B,), device=device).reshape(B, 1).broadcast_to((B, vocab))
    cdf = lucid.cumsum(probs, dim=-1)
    idx = (cdf < u).long().sum(dim=-1)
    # Guard the float edge case where u just exceeds cdf[-1] (≈1) → idx == vocab.
    return lucid.minimum(idx, lucid.full_like(idx, vocab - 1))
