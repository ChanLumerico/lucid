# Known parity divergences

This file is the authoritative list of every `xfail(strict=True)` in
`lucid/test/parity/`. When the underlying bug is fixed, the corresponding
test must be flipped from `xfail` to expected-pass — `strict=True` causes
the xpass to fail loudly so the flag has to be removed deliberately.

> **Rule:** the size of this file should monotonically decrease over time.

---

## Active xfails

*(none — every active xfail is currently closed)*

---

## Closed (regression-guarded by tests that now pass)

| Bug | Closed by | Where the fix landed |
|---|---|---|
| `matmul` 1D·2D / 2D·1D backward (`swapaxes` on 1-D) | `parity/ops/test_bfunc_parity.py::matmul_1d_2d`, `matmul_2d_1d` | `lucid/_func/bfunc.py` — promote 1D to 2D inside backward |
| `.T` backward returns un-transposed grad | `parity/ops/test_ufunc_parity.py::T_2d` | `lucid/_func/ufunc.py` — `_T.__grad__` returns `grad.T` |
| `stack` axis≥1 backward scales by `shape[0]` | `parity/ops/test_shape_parity.py::stack_axis1` | `lucid/_utils/func.py` — squeeze stacked axis |
| `masked_fill` broadcast backward IndexError | `parity/ops/test_indexing_parity.py::masked_fill_broadcast_row` | `lucid/_utils/func.py` — use `np.where` |
| `trace` ndim>2 backward | `parity/ops/test_ufunc_parity.py::trace_highdim` | `lucid/_func/ufunc.py` — broadcast-shape eye + grad |
| NAdam math diverges from torch / paper | `parity/optim/test_optim_parity.py::NAdam_default` | `lucid/optim/adam.py` — `mu_product` schedule |
| RAdam ρ_t crossover differs from torch | `parity/optim/test_optim_parity.py::RAdam_default` | `lucid/optim/adam.py` — threshold `>5`, `r_t / ρ_t`, eps placement |
| MultiStepLR / CosineAnnealingLR / LambdaLR off-by-one | `parity/optim/test_scheduler_parity.py::*` | `lucid/optim/lr_scheduler/_schedulers.py` — `last_epoch + 1` |
| LSTM/GRU forward gates miscount | `parity/nn/test_rnn_parity.py::LSTM_1layer`, `GRU_1layer` | `lucid/nn/modules/rnn.py` — `lucid.split` → `lucid.chunk` |
| ConvTranspose2d backward — `BackwardOperation` double-wraps tuple grads on poly-input ops with `num_inputs == 1` | `parity/nn/test_modules_parity.py::ConvTranspose2d_*` | `lucid/_backend/core.py` — drop `num_inputs == 1` from grad-tuple wrapping |
| Rprop state["step_size"] never updated (lib_.where rebound local) | `parity/optim/test_optim_parity.py::Rprop_default` | `lucid/optim/prop.py` — write back to state, use lib_.clip + lib_.sign |
| JIT compiled fn backward — `training_mode=False` blocked grad graph | `parity/jit_tests/test_jit_correctness.py::test_jit_backward_match` | `lucid/_jit/api.py` — infer `training_mode` from inputs' `requires_grad`; `_attach_compiled_backward` flips output's `requires_grad` |
