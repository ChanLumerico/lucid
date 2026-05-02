"""Specs for linalg ops. Forward only — autograd path for linalg is
intentionally not exercised here yet (the engine doesn't wire backward for
inv/qr/etc. in this phase)."""

from __future__ import annotations

import numpy as np
import torch

from lucid._C import engine as E

from ._specs import OpSpec


def _spd(rng, n=4):
    A = rng.standard_normal(size=(n, n)).astype("float32")
    return [A.T @ A + n * np.eye(n, dtype="float32")]


def _general(rng, n=4):
    return [rng.standard_normal(size=(n, n)).astype("float32")]


def _general_rect(rng, m=4, n=3):
    return [rng.standard_normal(size=(m, n)).astype("float32")]


SPECS: list[OpSpec] = [
    OpSpec(
        name="linalg_inv",
        engine_fn=lambda ts: E.linalg.inv(ts[0]),
        torch_fn=lambda ts: torch.linalg.inv(ts[0]),
        input_gen=_general,
        atol=1e-2, rtol=1e-2,  # f32 inv is lossy
    ),
    OpSpec(
        name="linalg_det",
        engine_fn=lambda ts: E.linalg.det(ts[0]),
        torch_fn=lambda ts: torch.linalg.det(ts[0]),
        input_gen=_general,
        atol=1e-2, rtol=1e-2,
    ),
    OpSpec(
        name="linalg_cholesky",
        engine_fn=lambda ts: E.linalg.cholesky(ts[0]),
        torch_fn=lambda ts: torch.linalg.cholesky(ts[0]),
        input_gen=_spd,
        atol=1e-3, rtol=1e-3,
        skip_grad=True,
    ),
    OpSpec(
        name="linalg_norm_l2_vector",
        engine_fn=lambda ts: E.linalg.norm(ts[0], 2.0, [], False),
        torch_fn=lambda ts: torch.linalg.norm(ts[0], ord=2),
        input_shapes=[(20,)],
        atol=1e-3, rtol=1e-3,
        notes="ord=2 on a vector = L2; matrix induced 2-norm needs SVD.",
    ),
    OpSpec(
        name="linalg_norm_l1_vector",
        engine_fn=lambda ts: E.linalg.norm(ts[0], 1.0, [], False),
        torch_fn=lambda ts: torch.linalg.norm(ts[0], ord=1),
        input_shapes=[(20,)],
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="linalg_norm_l2_axis",
        engine_fn=lambda ts: E.linalg.norm(ts[0], 2.0, [-1], False),
        torch_fn=lambda ts: torch.linalg.norm(ts[0], ord=2, dim=-1),
        input_shapes=[(4, 5)],
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="linalg_qr",
        engine_fn=lambda ts: E.linalg.qr(ts[0])[0],
        torch_fn=lambda ts: torch.linalg.qr(ts[0]).Q,
        input_gen=_general_rect,
        atol=1e-2, rtol=1e-2,
        skip_grad=True,
        notes="QR sign convention can flip across backends; check Q only.",
    ),
    OpSpec(
        name="linalg_matrix_power",
        engine_fn=lambda ts: E.linalg.matrix_power(ts[0], 3),
        torch_fn=lambda ts: torch.linalg.matrix_power(ts[0], 3),
        input_gen=_general,
        atol=1e-2, rtol=1e-2,
        skip_grad=True,
    ),
    OpSpec(
        name="linalg_solve",
        engine_fn=lambda ts: E.linalg.solve(ts[0], ts[1]),
        torch_fn=lambda ts: torch.linalg.solve(ts[0], ts[1]),
        input_gen=lambda rng: [
            (rng.standard_normal((4, 4)).astype("float32")
              + 4 * np.eye(4, dtype="float32")),
            rng.standard_normal((4, 3)).astype("float32"),
        ],
        atol=1e-3, rtol=1e-3,
    ),
    OpSpec(
        name="linalg_pinv",
        engine_fn=lambda ts: E.linalg.pinv(ts[0]),
        torch_fn=lambda ts: torch.linalg.pinv(ts[0]),
        input_gen=_general_rect,
        atol=1e-2, rtol=1e-2,
        skip_grad=True,
    ),
    OpSpec(
        name="linalg_svd_S",
        engine_fn=lambda ts: E.linalg.svd(ts[0], False)[0],
        torch_fn=lambda ts: torch.linalg.svd(ts[0]).S,
        input_gen=_general_rect,
        atol=1e-2, rtol=1e-2,
        skip_grad=True,
        notes="compute_uv=False — singular values only.",
    ),
    OpSpec(
        name="linalg_eigh_w",
        engine_fn=lambda ts: E.linalg.eig(ts[0])[0],
        torch_fn=lambda ts: torch.linalg.eig(ts[0]).eigenvalues.real,
        input_gen=_spd,
        atol=1e-2, rtol=1e-2,
        skip_grad=True,
        skip_gpu=True,
        notes="General eig: LAPACK geev order vs MLX may differ; CPU-only.",
    ),
    OpSpec(
        name="linalg_eigh_sym_w",
        engine_fn=lambda ts: E.linalg.eigh(ts[0])[0],
        torch_fn=lambda ts: torch.linalg.eigh(ts[0]).eigenvalues,
        input_gen=_spd,
        atol=1e-2, rtol=1e-2,
        skip_grad=True,
        notes="Symmetric eigh: eigenvalues ascending, CPU+GPU.",
    ),
    OpSpec(
        name="linalg_eigh_reconstruct",
        engine_fn=lambda ts: (lambda r: E.matmul(E.matmul(r[1], E.diag(r[0])), E.transpose(r[1])))(E.linalg.eigh(ts[0])),
        torch_fn=lambda ts: (lambda r: r.eigenvectors @ torch.diag(r.eigenvalues) @ r.eigenvectors.T)(torch.linalg.eigh(ts[0])),
        input_gen=_spd,
        atol=1e-3, rtol=1e-3,
        skip_grad=True,
        notes="eigh reconstruction: V @ diag(w) @ V^T ≈ A (sidesteps sign ambiguity).",
    ),
]
