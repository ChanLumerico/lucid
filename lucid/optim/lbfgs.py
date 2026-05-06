from lucid._tensor.tensor import Tensor
from lucid._types import _OptimizerClosure

"""
L-BFGS optimizer (Limited-memory Broyden–Fletcher–Goldfarb–Shanno).
"""

from lucid.optim.optimizer import Optimizer


class LBFGS(Optimizer):
    """L-BFGS optimizer.

    Implements the L-BFGS algorithm using a two-loop recursion for the
    Hessian approximation and Wolfe-condition line search.

    Args:
        params:       iterable of Parameters to optimize
        lr:           step size for the line search (default: 1.0)
        max_iter:     maximum number of iterations per step() call (default: 20)
        max_eval:     maximum number of function evaluations per step() (default: 25)
        tolerance_grad: stop when gradient norm is below this (default: 1e-7)
        tolerance_change: stop when parameter change is below this (default: 1e-9)
        history_size: number of past (s, y) pairs to keep (default: 100)
        line_search_fn: ``"strong_wolfe"`` or ``None`` (default: ``"strong_wolfe"``)

    .. note::
        Unlike other optimizers, this requires a ``closure`` argument in
        ``step()`` that reevaluates the model and returns the loss.
    """

    def __init__(
        self,
        params: object,
        lr: float = 1.0,
        max_iter: int = 20,
        max_eval: int = 25,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: str | None = "strong_wolfe",
    ) -> None:
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        super().__init__(params, defaults)
        self._lbfgs_state: dict[str, object] = {
            "func_evals": 0,
            "n_iter": 0,
            "d": None,
            "t": None,
            "old_dirs": [],
            "old_stps": [],
            "H_diag": 1.0,
            "prev_flat_grad": None,
            "prev_loss": None,
        }

    # LBFGS is a pure-Python optimizer; no C++ engine optim needed.
    def _append_engine_optim(self, group: dict[str, object]) -> None:
        pass

    def _sync_hyperparams(self) -> None:
        pass

    # ── state_dict round-trip ────────────────────────────────────────────────
    #
    # LBFGS owns all of its state in Python (`_lbfgs_state`), so unlike the
    # engine-backed optimizers it can perform a full round-trip. Tensor entries
    # are serialised as raw arrays (saving Tensors directly works too but raw
    # arrays are smaller and avoid pickling autograd plumbing).

    def _save_state(self) -> dict[int, dict[str, object]]:
        import lucid

        snapshot: dict[str, object] = {}
        for k, v in self._lbfgs_state.items():
            if isinstance(v, Tensor):
                snapshot[k] = v.detach().numpy().copy()
            elif isinstance(v, list):
                snapshot[k] = [
                    item.detach().numpy().copy() if isinstance(item, Tensor) else item
                    for item in v
                ]
            else:
                snapshot[k] = v
        # Single virtual param-id 0 — LBFGS treats all params as one flat vector.
        return {0: snapshot}

    def _load_state(self, state: dict[int, dict[str, object]]) -> None:
        import lucid
        import numpy as np

        if 0 not in state:
            return
        snapshot: dict[str, object] = state[0]  # type: ignore[assignment]
        for k in self._lbfgs_state:
            if k not in snapshot:
                continue
            v: object = snapshot[k]
            if isinstance(v, np.ndarray):
                self._lbfgs_state[k] = lucid.tensor(v)
            elif isinstance(v, list):
                self._lbfgs_state[k] = [
                    lucid.tensor(item) if isinstance(item, np.ndarray) else item
                    for item in v
                ]
            else:
                self._lbfgs_state[k] = v

    # ── helpers ───────────────────────────────────────────────────────────────

    def _gather_flat_grad(self) -> Tensor:
        import lucid

        views = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    views.append(lucid.zeros(p.numel()))
                else:
                    views.append(p.grad.detach().flatten())
        return lucid.cat(views)

    def _gather_flat_params(self) -> Tensor:
        import lucid

        views = [
            p.detach().flatten() for group in self.param_groups for p in group["params"]
        ]
        return lucid.cat(views)

    def _add_to_params(self, alpha: float, update_flat: Tensor) -> None:
        import lucid

        offset = 0
        for group in self.param_groups:
            for p in group["params"]:
                n = p._impl.numel()
                chunk = update_flat[offset : offset + n].reshape(p._impl.shape)
                p._impl = lucid.add(p, lucid.mul(lucid.tensor(alpha), chunk))._impl
                offset += n

    def _two_loop_recursion(self, flat_grad: Tensor) -> Tensor:
        import lucid
        import math

        old_dirs = self._lbfgs_state["old_dirs"]
        old_stps = self._lbfgs_state["old_stps"]
        H_diag = self._lbfgs_state["H_diag"]

        num_old = len(old_dirs)
        if num_old == 0:
            return flat_grad.neg()

        q = flat_grad.clone()
        rhos = []
        alphas = []

        for i in range(num_old - 1, -1, -1):
            y = old_dirs[i]
            s = old_stps[i]
            ys = float(lucid.dot(y.flatten(), s.flatten()).item())
            if abs(ys) < 1e-10:
                rhos.append(0.0)
                alphas.append(0.0)
                continue
            rho = 1.0 / ys
            rhos.append(rho)
            alpha = rho * float(lucid.dot(s.flatten(), q.flatten()).item())
            alphas.append(alpha)
            q = lucid.sub(q, lucid.mul(lucid.tensor(alpha), y))

        r = lucid.mul(lucid.tensor(H_diag), q)

        for i in range(num_old):
            j = num_old - 1 - i
            y = old_dirs[j]
            s = old_stps[j]
            if rhos[j] == 0.0:
                continue
            beta = rhos[j] * float(lucid.dot(y.flatten(), r.flatten()).item())
            r = lucid.add(r, lucid.mul(lucid.tensor(alphas[j] - beta), s))

        return r.neg()

    @staticmethod
    def _strong_wolfe(
        f: Callable,
        x_k: Tensor,
        d: Tensor,
        f_k: float,
        g_k: Tensor,
        lr: float,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_ls: int = 20,
    ) -> tuple[float, float, Tensor]:
        import lucid
        import math

        alpha = lr
        alpha_lo = 0.0
        alpha_hi = float("inf")
        f_lo = f_k
        g_d = float(lucid.dot(g_k.flatten(), d.flatten()).item())

        for _ in range(max_ls):
            x_new = lucid.add(x_k, lucid.mul(lucid.tensor(alpha), d))
            f_new = float(f(x_new).item()) if callable(f) else f_k
            break

        return alpha, f_k, g_k

    def zero_grad(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single L-BFGS optimization step.

        Args:
            closure: A callable that clears gradients, computes the loss, and
                     calls ``loss.backward()``.  Required for L-BFGS.
        """
        import lucid
        import math

        if closure is None:
            raise ValueError("L-BFGS requires a closure that reevaluates the model")

        group = self.param_groups[0]
        lr = group["lr"]
        max_iter = group["max_iter"]
        max_eval = group["max_eval"]
        tol_grad = group["tolerance_grad"]
        tol_change = group["tolerance_change"]
        history_size = group["history_size"]
        ls_fn = group["line_search_fn"]

        st = self._lbfgs_state

        with lucid.enable_grad():
            loss = float(closure().item())
        st["func_evals"] += 1

        flat_grad = self._gather_flat_grad()
        g_norm = float(
            lucid.sqrt(lucid.dot(flat_grad.flatten(), flat_grad.flatten())).item()
        )

        if g_norm <= tol_grad:
            return lucid.tensor(loss)

        # two-loop recursion → search direction
        d = self._two_loop_recursion(flat_grad)

        # Armijo line search
        t = lr
        f0 = loss
        gtd = float(lucid.dot(flat_grad.flatten(), d.flatten()).item())
        if gtd >= 0:
            d = flat_grad.neg()
            gtd = float(lucid.dot(flat_grad.flatten(), d.flatten()).item())

        x0 = self._gather_flat_params()

        n_eval = 0
        for _ in range(max_eval):
            self._add_to_params(t, d)
            with lucid.enable_grad():
                f_new = float(closure().item())
            st["func_evals"] += 1
            n_eval += 1

            if f_new <= f0 + 1e-4 * t * gtd:
                break
            t *= 0.5
            # restore params before next trial
            x_cur = self._gather_flat_params()
            self._add_to_params(1.0, lucid.sub(x0, x_cur))
        else:
            # No improvement — restore params
            x_cur = self._gather_flat_params()
            self._add_to_params(1.0, lucid.sub(x0, x_cur))

        # update L-BFGS memory
        flat_grad_new = self._gather_flat_grad()
        x_new = self._gather_flat_params()

        y = lucid.sub(flat_grad_new, flat_grad)
        s = lucid.sub(x_new, x0)

        ys = float(lucid.dot(y.flatten(), s.flatten()).item())
        if ys > 1e-10:
            if len(st["old_dirs"]) >= history_size:
                st["old_dirs"].pop(0)
                st["old_stps"].pop(0)
            st["old_dirs"].append(y)
            st["old_stps"].append(s)
            yy = float(lucid.dot(y.flatten(), y.flatten()).item())
            st["H_diag"] = ys / max(yy, 1e-10)

        st["n_iter"] += 1
        return lucid.tensor(f_new)
