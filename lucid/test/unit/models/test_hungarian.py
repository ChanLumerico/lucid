"""Unit tests for the custom Kuhn-Munkres / Hungarian algorithm used by
DETR / MaskFormer / Mask2Former matchers.

Validates ``solve_assignment`` against ``scipy.optimize.linear_sum_assignment``
on a range of randomised rectangular cost matrices.  ``scipy`` is only used inside
this test module — production code never imports it.
"""

import random
import unittest

import pytest

# Skip the whole module if scipy isn't available (it's optional [test] extra)
scipy_opt = pytest.importorskip("scipy.optimize")
from lucid.models._utils._detection import solve_assignment


def _ref_assignment(cost: list[list[float]]) -> dict[int, int]:
    """Reference: scipy returns optimal row->col assignment for rectangular matrices."""
    row_ind, col_ind = scipy_opt.linear_sum_assignment(cost)
    return dict(zip(row_ind.tolist(), col_ind.tolist()))


def _our_assignment(cost: list[list[float]]) -> dict[int, int]:
    rows, cols = solve_assignment(cost)
    return dict(zip(rows, cols))


def _total_cost(cost: list[list[float]], assignment: dict[int, int]) -> float:
    return sum(cost[r][c] for r, c in assignment.items())


class TestHungarianCorrectness(unittest.TestCase):
    """Compare custom Hungarian against scipy on random + handcrafted cases."""

    def test_trivial_3x3(self) -> None:
        cost = [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 2.0, 1.0],
        ]
        ours = _our_assignment(cost)
        # Optimal: rows 0/1/2 → cols 0/1/2, total = 3.0
        self.assertAlmostEqual(_total_cost(cost, ours), 3.0, places=5)

    def test_rectangular_3x5(self) -> None:
        """3 rows × 5 cols — assign each row to a distinct col."""
        cost = [
            [10.0, 1.0, 9.0, 8.0, 7.0],
            [5.0, 6.0, 2.0, 8.0, 9.0],
            [4.0, 5.0, 6.0, 7.0, 1.0],
        ]
        ours = _our_assignment(cost)
        ref = _ref_assignment(cost)
        self.assertAlmostEqual(
            _total_cost(cost, ours),
            _total_cost(cost, ref),
            places=5,
        )

    def test_obvious_match_5x3(self) -> None:
        """5 queries × 3 GTs — rows 0/1/2 should match cols 0/1/2 (diagonal cheap)."""
        # Note: API is (n_rows ≤ n_cols), so caller should construct cost as (M, N).
        cost = [
            [0.0, 10.0, 10.0, 100.0, 100.0],
            [10.0, 0.0, 10.0, 100.0, 100.0],
            [10.0, 10.0, 0.0, 100.0, 100.0],
        ]
        rows, cols = solve_assignment(cost)
        self.assertEqual(rows, [0, 1, 2])
        self.assertEqual(cols, [0, 1, 2])

    def test_negative_costs(self) -> None:
        """DETR uses negative costs (since cost = -log_prob - GIoU)."""
        cost = [
            [-5.0, -1.0, -3.0],
            [-2.0, -8.0, -4.0],
            [-3.0, -2.0, -7.0],
        ]
        ours = _our_assignment(cost)
        ref = _ref_assignment(cost)
        self.assertAlmostEqual(
            _total_cost(cost, ours),
            _total_cost(cost, ref),
            places=5,
        )

    def test_random_rectangular(self) -> None:
        """100 randomised M×N matrices (M ≤ N) vs scipy."""
        rng = random.Random(42)
        for trial in range(100):
            M = rng.randint(1, 8)
            N = rng.randint(M, 12)
            cost = [[rng.uniform(-5.0, 5.0) for _ in range(N)] for _ in range(M)]
            ours = _our_assignment(cost)
            ref = _ref_assignment(cost)
            ours_cost = _total_cost(cost, ours)
            ref_cost = _total_cost(cost, ref)
            self.assertAlmostEqual(
                ours_cost,
                ref_cost,
                places=4,
                msg=f"trial {trial} M={M} N={N}: ours={ours_cost} ref={ref_cost}",
            )

    def test_detr_scale(self) -> None:
        """DETR-scale random matrix: 5 GTs × 100 queries."""
        rng = random.Random(0)
        M, N = 5, 100
        cost = [[rng.uniform(-2.0, 2.0) for _ in range(N)] for _ in range(M)]
        ours = _our_assignment(cost)
        ref = _ref_assignment(cost)
        self.assertAlmostEqual(
            _total_cost(cost, ours),
            _total_cost(cost, ref),
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
