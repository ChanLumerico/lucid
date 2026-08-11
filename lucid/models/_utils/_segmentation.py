"""Shared loss primitives for mask-classification segmentation heads.

MaskFormer and Mask2Former both frame segmentation as *mask classification*:
:math:`N` queries each predict a class and a binary mask, and training needs a
bipartite matching between those queries and the ground-truth segments plus a
per-pair mask objective.  The pieces below are the parts both families share.

The two differ in one respect that matters here — MaskFormer's mask cost is a
**focal** term, Mask2Former's is plain **sigmoid cross-entropy** — so the
pairwise helpers come in both flavours rather than one being made to stand in
for the other.
"""

import lucid
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor

__all__ = [
    "pairwise_sigmoid_ce",
    "pairwise_dice",
    "sigmoid_ce_loss",
    "dice_loss",
    "sample_point_coords",
    "point_sample",
    "uncertain_point_coords",
]


def pairwise_sigmoid_ce(mask_logits: Tensor, gt_masks: Tensor) -> Tensor:
    r"""Sigmoid cross-entropy between every query mask and every GT mask.

    Computed in closed matrix form.  Binary cross-entropy splits into a
    positive-class part weighted by the target and a negative-class part
    weighted by its complement, so contracting each against the target
    matrix yields all :math:`N \times M` pairings from two matmuls instead
    of a Python loop over pairs.

    Args:
        mask_logits: ``(N, P)`` raw (pre-sigmoid) query mask logits.
        gt_masks:    ``(M, P)`` binary ground-truth masks.

    Returns:
        ``(N, M)`` mean per-point cross-entropy for each pairing.
    """
    p = int(mask_logits.shape[1])
    # Through logsigmoid so saturated logits stay finite.
    neg_log_p = -F.logsigmoid(mask_logits)
    neg_log_1mp = -F.logsigmoid(-mask_logits)

    gt_t = gt_masks.permute(1, 0)  # (P, M)
    return (neg_log_p @ gt_t + neg_log_1mp @ (1.0 - gt_t)) / float(p)


def pairwise_dice(mask_logits: Tensor, gt_masks: Tensor) -> Tensor:
    """Dice cost between every query mask and every GT mask.

    Args:
        mask_logits: ``(N, P)`` raw query mask logits.
        gt_masks:    ``(M, P)`` binary ground-truth masks.

    Returns:
        ``(N, M)`` dice cost, ``1 - 2|A∩B| / (|A| + |B|)`` smoothed by 1 on
        both numerator and denominator — the reference's smoothing, which
        scores an empty prediction against an empty target as 0 rather than
        the 1.0 a bare epsilon guard would give.
    """
    prob = F.sigmoid(mask_logits)
    inter = prob @ gt_masks.permute(1, 0)
    denom = prob.sum(dim=-1).reshape(-1, 1) + gt_masks.sum(dim=-1).reshape(1, -1)
    return 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)


def sigmoid_ce_loss(mask_logits: Tensor, gt_masks: Tensor) -> Tensor:
    """Mean sigmoid cross-entropy for already-matched pairs.

    Args:
        mask_logits: ``(M, P)`` logits of the matched queries.
        gt_masks:    ``(M, P)`` their targets.

    Returns:
        Scalar — the mean over points, summed over pairs and divided by the
        pair count, as the reference normalises it.
    """
    if int(mask_logits.shape[0]) == 0:
        return lucid.zeros((), device=mask_logits.device.type)
    return F.binary_cross_entropy_with_logits(mask_logits, gt_masks)


def dice_loss(mask_logits: Tensor, gt_masks: Tensor) -> Tensor:
    """Dice loss for already-matched pairs.

    Args:
        mask_logits: ``(M, P)`` logits of the matched queries.
        gt_masks:    ``(M, P)`` their targets.

    Returns:
        Scalar mean over the matched pairs.
    """
    m = int(mask_logits.shape[0])
    if m == 0:
        return lucid.zeros((), device=mask_logits.device.type)
    prob = F.sigmoid(mask_logits)
    inter = (prob * gt_masks).sum(dim=-1)
    denom = prob.sum(dim=-1) + gt_masks.sum(dim=-1)
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def sample_point_coords(
    num_points: int,
    *,
    device: str = "cpu",
) -> Tensor:
    """Draw uniformly random point coordinates in the unit square.

    Mask2Former 3.2.2 evaluates its mask terms on ``K`` sampled points
    rather than densely, which is what makes training at high resolution
    affordable — the paper reports roughly a 3x memory saving.

    Args:
        num_points: ``K``.
        device:     Where to place the result.

    Returns:
        ``(K, 2)`` coordinates in ``[0, 1]``, ordered ``(x, y)``.
    """
    return lucid.rand(num_points, 2, device=device)


def point_sample(masks: Tensor, coords: Tensor) -> Tensor:
    """Bilinearly sample masks at arbitrary continuous coordinates.

    Args:
        masks:  ``(N, H, W)`` mask logits.
        coords: ``(K, 2)`` points shared by every mask, or ``(N, K, 2)``
            giving each mask its own — the importance sampler produces the
            latter, since which points are informative depends on the mask.
            Ordered ``(x, y)`` in ``[0, 1]``.

    Returns:
        ``(N, K)`` sampled values.

    Notes:
        ``grid_sample`` takes coordinates in ``[-1, 1]``, so the unit-square
        points are rescaled here rather than at every call site.
    """
    n = int(masks.shape[0])
    k = int(coords.shape[-2])
    if n == 0 or k == 0:
        return lucid.zeros((n, k), device=masks.device.type)

    scaled = coords * 2.0 - 1.0
    grid = (
        scaled.reshape(1, 1, k, 2).expand(n, 1, k, 2)
        if scaled.ndim == 2
        else scaled.reshape(n, 1, k, 2)
    )
    sampled = F.grid_sample(
        masks.unsqueeze(1), grid, mode="bilinear", align_corners=False
    )
    return sampled.reshape(n, k)


def uncertain_point_coords(
    mask_logits: Tensor,
    num_points: int,
    oversample_ratio: float = 3.0,
    importance_ratio: float = 0.75,
) -> Tensor:
    """Draw points biased towards the mask's own decision boundary.

    Mask2Former 3.2.2 samples ``K`` points per mask rather than evaluating
    densely, and biases most of them towards where the prediction is
    *uncertain* — a logit near zero.  Points deep inside or far outside a
    predicted mask carry almost no gradient, so spending the budget on the
    boundary is what lets ``K`` be small enough to matter.

    Args:
        mask_logits:      ``(N, H, W)`` predicted mask logits.
        num_points:       ``K``, the per-mask budget.
        oversample_ratio: How many candidates to draw before selecting.
        importance_ratio: Share of ``K`` taken from the most uncertain
            candidates; the remainder is drawn uniformly so the sampler
            never collapses onto the boundary alone.

    Returns:
        ``(N, K, 2)`` coordinates in ``[0, 1]``.
    """
    n = int(mask_logits.shape[0])
    dev = mask_logits.device.type
    if n == 0 or num_points == 0:
        return lucid.zeros((n, num_points, 2), device=dev)

    n_sampled = max(1, int(num_points * oversample_ratio))
    cand = lucid.rand(n, n_sampled, 2, device=dev)
    # Uncertainty is distance from the 0.5 probability contour, i.e. from a
    # logit of zero; negating makes "most uncertain" the largest value.
    uncertainty = -point_sample(mask_logits, cand).abs()  # (N, n_sampled)

    n_uncertain = min(int(importance_ratio * num_points), n_sampled)
    n_random = num_points - n_uncertain

    order = lucid.argsort(-uncertainty, dim=-1)[:, :n_uncertain]  # (N, n_unc)
    picked_parts: list[Tensor] = []
    for i in range(n):
        picked_parts.append(cand[i][order[i]].reshape(1, n_uncertain, 2))
    picked = lucid.cat(picked_parts, dim=0) if picked_parts else cand[:, :0]

    if n_random > 0:
        extra = lucid.rand(n, n_random, 2, device=dev)
        picked = lucid.cat([picked, extra], dim=1)
    return picked
