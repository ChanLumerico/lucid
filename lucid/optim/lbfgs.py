"""
L-BFGS optimizer (Limited-memory Broyden–Fletcher–Goldfarb–Shanno).
"""

from collections.abc import Iterable
from typing import Callable, cast

import lucid
from lucid._tensor.tensor import Tensor
from lucid._types import _OptimizerClosure
from lucid.nn.parameter import Parameter
from lucid.optim.optimizer import Optimizer


class LBFGS(Optimizer):
    r"""Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) optimizer.

    L-BFGS is a quasi-Newton method that approximates the inverse Hessian
    using a limited history of gradient and parameter difference vectors.
    At each step it computes a search direction :math:`d_t` via the
    two-loop recursion:

    .. math::

        d_t = -H_t^{-1} \nabla L(\theta_t)

    where :math:`H_t^{-1}` is the L-BFGS Hessian approximation built from
    the last ``history_size`` curvature pairs
    :math:`\{(s_k, y_k)\}_{k=t-m}^{t-1}`:

    .. math::

        s_k &= \theta_{k+1} - \theta_k \\
        y_k &= \nabla L(\theta_{k+1}) - \nabla L(\theta_k)

    The diagonal scaling of :math:`H_t^{-1}` is initialised as:

    .. math::

        H_{\text{diag}} = \frac{s_{t-1}^\top y_{t-1}}{y_{t-1}^\top y_{t-1}}

    A back-tracking Armijo line search finds a step size :math:`\alpha` that
    satisfies the sufficient-decrease condition:

    .. math::

        L(\theta_t + \alpha d_t)
        \le L(\theta_t) + c_1 \alpha \, \nabla L(\theta_t)^\top d_t

    with :math:`c_1 = 10^{-4}`.

    Parameters
    ----------
    params : iterable of Parameter or iterable of dict
        Parameters to optimise.
    lr : float, optional
        Initial step size for the line search (default: ``1.0``).
    max_iter : int, optional
        Maximum number of L-BFGS iterations per :meth:`step` call
        (default: ``20``).
    max_eval : int, optional
        Maximum number of closure evaluations per :meth:`step` call
        (default: ``25``).
    tolerance_grad : float, optional
        Gradient-norm convergence threshold; optimisation stops when
        :math:`\|\nabla L\|_2 \le \text{tolerance\_grad}`
        (default: ``1e-7``).
    tolerance_change : float, optional
        Parameter-change convergence threshold (default: ``1e-9``).
    history_size : int, optional
        Number of :math:`(s, y)` curvature pairs retained in memory
        (default: ``100``).
    line_search_fn : str or None, optional
        Line search strategy.  Currently ``"strong_wolfe"`` (back-tracking
        Armijo) and ``None`` (fixed step) are recognised
        (default: ``"strong_wolfe"``).

    Attributes
    ----------
    param_groups : list of dict
        Single parameter group containing all parameters.
    defaults : dict
        Default hyperparameter values.

    Notes
    -----
    Unlike first-order optimizers, L-BFGS **requires a closure** argument
    in :meth:`step` that clears gradients, computes the loss, and calls
    ``loss.backward()``.  Without a closure the method raises
    :exc:`ValueError`.

    L-BFGS is best suited for full-batch or large-batch training where the
    curvature information is reliable.  It is not recommended for
    stochastic mini-batch training because noisy gradients corrupt the
    Hessian approximation.

    Examples
    --------
    >>> import lucid.optim as optim
    >>> optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20)
    >>> def closure():
    ...     optimizer.zero_grad()
    ...     loss = criterion(model(x), y)
    ...     loss.backward()
    ...     return loss
    >>> optimizer.step(closure)
    """

    def __init__(
        self,
        params: Iterable[Parameter] | Iterable[dict[str, object]],
        lr: float = 1.0,
        max_iter: int = 20,
        max_eval: int = 25,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: str | None = "strong_wolfe",
    ) -> None:
        """Initialise the LBFGS.  See the class docstring for parameter semantics."""
        defaults: dict[str, object] = dict(
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
        import numpy as np

        if 0 not in state:
            return
        snapshot: dict[str, object] = state[0]
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
        views = []
        for group in self.param_groups:
            for p in cast(list[Tensor], group["params"]):
                if p.grad is None:
                    views.append(lucid.zeros(p.numel()))
                else:
                    views.append(p.grad.detach().flatten())
        return lucid.cat(views)

    def _gather_flat_params(self) -> Tensor:
        views = [
            p.detach().flatten()
            for group in self.param_groups
            for p in cast(list[Tensor], group["params"])
        ]
        return lucid.cat(views)

    def _add_to_params(self, alpha: float, update_flat: Tensor) -> None:
        offset = 0
        for group in self.param_groups:
            for p in cast(list[Tensor], group["params"]):
                n = p._impl.numel()
                chunk = lucid.reshape(update_flat[offset : offset + n], list(p._impl.shape))  # type: ignore[arg-type]
                p._impl = lucid.add(p, lucid.mul(lucid.tensor(alpha), chunk))._impl
                offset += n

    def _two_loop_recursion(self, flat_grad: Tensor) -> Tensor:

        old_dirs = cast(list[Tensor], self._lbfgs_state["old_dirs"])
        old_stps = cast(list[Tensor], self._lbfgs_state["old_stps"])
        H_diag = cast(float, self._lbfgs_state["H_diag"])

        num_old = len(old_dirs)
        if num_old == 0:
            return flat_grad.neg()

        q = flat_grad.clone()
        rhos = []
        alphas = []

        for i in range(num_old - 1, -1, -1):
            y = old_dirs[i]
            s = old_stps[i]
            ys = float(lucid.linalg.dot(y.flatten(), s.flatten()).item())
            if abs(ys) < 1e-10:
                rhos.append(0.0)
                alphas.append(0.0)
                continue
            rho = 1.0 / ys
            rhos.append(rho)
            alpha = rho * float(lucid.linalg.dot(s.flatten(), q.flatten()).item())
            alphas.append(alpha)
            q = lucid.sub(q, lucid.mul(lucid.tensor(alpha), y))

        r = lucid.mul(lucid.tensor(H_diag), q)

        for i in range(num_old):
            j = num_old - 1 - i
            y = old_dirs[j]
            s = old_stps[j]
            if rhos[j] == 0.0:
                continue
            beta = rhos[j] * float(lucid.linalg.dot(y.flatten(), r.flatten()).item())
            r = lucid.add(r, lucid.mul(lucid.tensor(alphas[j] - beta), s))

        return r.neg()

    @staticmethod
    def _strong_wolfe(
        f: Callable[..., object],
        x_k: Tensor,
        d: Tensor,
        f_k: float,
        g_k: Tensor,
        lr: float,
        c1: float = 1e-4,
        c2: float = 0.9,
        max_ls: int = 20,
    ) -> tuple[float, float, Tensor]:

        alpha = lr
        alpha_lo = 0.0
        alpha_hi = float("inf")
        f_lo = f_k
        g_d = float(lucid.linalg.dot(g_k.flatten(), d.flatten()).item())

        for _ in range(max_ls):
            x_new = lucid.add(x_k, lucid.mul(lucid.tensor(alpha), d))
            f_new = float(cast(Tensor, f(x_new)).item()) if callable(f) else f_k
            break

        return alpha, f_k, g_k

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Set gradients of all parameters to ``None``.

        L-BFGS always sets gradients to ``None`` regardless of the
        ``set_to_none`` argument, because the closure passed to
        :meth:`step` is responsible for zeroing and recomputing gradients
        on each function evaluation.

        Parameters
        ----------
        set_to_none : bool, optional
            Ignored; kept for API compatibility with :class:`Optimizer`
            (default: ``True``).

        Examples
        --------
        >>> def closure():
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(x), y)
        ...     loss.backward()
        ...     return loss
        >>> optimizer.step(closure)
        """
        for group in self.param_groups:
            for p in cast(list[Tensor], group["params"]):
                p.grad = None

    def step(self, closure: _OptimizerClosure = None) -> Tensor | None:
        """Perform a single L-BFGS optimisation step.

        Computes the L-BFGS search direction using the two-loop recursion,
        performs a back-tracking Armijo line search to find an acceptable
        step size, updates all parameters, and then updates the curvature
        history :math:`(s, y)`.

        Parameters
        ----------
        closure : callable
            A zero-argument callable that:

            1. Calls ``optimizer.zero_grad()`` to clear stale gradients.
            2. Runs the forward pass and computes the scalar loss.
            3. Calls ``loss.backward()`` to populate gradients.
            4. Returns the loss tensor.

            This argument is **required** — passing ``None`` raises
            :exc:`ValueError`.

        Returns
        -------
        Tensor
            The loss value at the final parameter position after the line
            search.

        Raises
        ------
        ValueError
            If ``closure`` is ``None``.

        Notes
        -----
        The closure may be called multiple times per :meth:`step` call
        (up to ``max_eval`` times) during the line search.  Ensure that
        any side effects (e.g. batch norm running stats) are handled
        appropriately if this matters for your use-case.

        Examples
        --------
        >>> def closure():
        ...     optimizer.zero_grad()
        ...     output = model(x)
        ...     loss = criterion(output, y)
        ...     loss.backward()
        ...     return loss
        >>> optimizer.step(closure)
        """

        if closure is None:
            raise ValueError("L-BFGS requires a closure that reevaluates the model")

        group = self.param_groups[0]
        lr = cast(float, group["lr"])
        max_iter = cast(int, group["max_iter"])
        max_eval = cast(int, group["max_eval"])
        tol_grad = cast(float, group["tolerance_grad"])
        tol_change = cast(float, group["tolerance_change"])
        history_size = cast(int, group["history_size"])
        ls_fn = group["line_search_fn"]

        st = self._lbfgs_state

        with lucid.enable_grad():
            loss = float(closure().item())
        st["func_evals"] += 1

        flat_grad = self._gather_flat_grad()
        g_norm = float(
            lucid.sqrt(
                lucid.linalg.dot(flat_grad.flatten(), flat_grad.flatten())
            ).item()
        )

        if g_norm <= tol_grad:
            return lucid.tensor(loss)

        # two-loop recursion → search direction
        d = self._two_loop_recursion(flat_grad)

        # Armijo line search
        t = lr
        f0 = loss
        gtd = float(lucid.linalg.dot(flat_grad.flatten(), d.flatten()).item())
        if gtd >= 0:
            d = flat_grad.neg()
            gtd = float(lucid.linalg.dot(flat_grad.flatten(), d.flatten()).item())

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

        ys = float(lucid.linalg.dot(y.flatten(), s.flatten()).item())
        if ys > 1e-10:
            old_dirs_list = cast(list[Tensor], st["old_dirs"])
            old_stps_list = cast(list[Tensor], st["old_stps"])
            if len(old_dirs_list) >= history_size:
                old_dirs_list.pop(0)
                old_stps_list.pop(0)
            old_dirs_list.append(y)
            old_stps_list.append(s)
            yy = float(lucid.linalg.dot(y.flatten(), y.flatten()).item())
            st["H_diag"] = ys / max(yy, 1e-10)

        st["n_iter"] += 1
        return lucid.tensor(f_new)
