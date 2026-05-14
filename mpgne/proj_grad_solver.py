"""
proj_grad_solver.py — Jacobi Best-Response (Projected Gradient) iterative
                       GNE baseline solver.

The simplest distributed iterative GNE method: at every iteration k each
agent *simultaneously* computes its best-response against the others' current
strategies (Jacobi update rule), which is equivalent to solving a projected
gradient step for a quadratic game.

Jacobi Best-Response Iteration
───────────────────────────────
    x_i^{k+1} = argmin  ½ x_i^T Q_i x_i  +  (F_cross_i U_{-i}^k + c_i + F_i p)^T x_i
                s.t.    A_loc_i x_i ≤ b_loc_i + S_loc_i p          (input bounds)
                        ±Γ_i x_i    ≤ rhs_state_i(U_{-i}^k, p)     (state bounds)
                        C_i x_i     ≤ rhs_coup_i(U_{-i}^k, p)      (coupling, if any)

All agents use x^k (not x^{k+1}) on the right-hand side — the Jacobi rule.

Stopping criterion
──────────────────
    δ^k = ‖x^{k+1} − x^k‖  <  tol

QP Solver selection
───────────────────
Pass  qp_solver="osqp"   (default) for OSQP — fast, cold-start only.
Pass  qp_solver="slsqp"  to fall back to scipy SLSQP.

Notes
─────
* Cold-start only (x^0 = 0) — no warm-starting, matching the benchmark
  protocol for a fair comparison with FACET-GNE.
* Reference: Facchinei & Pang (2003), "Finite-Dimensional Variational
  Inequalities", Chapter 12 (best-response dynamics).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

try:
    import osqp as _osqp_lib
    _OSQP_AVAILABLE = True
except ImportError:
    _OSQP_AVAILABLE = False

from .game import GNEGame


# ─────────────────────────────────────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PGResult:
    """
    Outcome of one Jacobi Best-Response run.

    Attributes
    ----------
    x_sol      : list[ndarray]   x_i* per agent
    n_iter     : int             iterations performed
    converged  : bool            True if δ < tol before max_iter
    conv_hist  : list[float]     ‖x^{k+1} − x^k‖ per iteration
    solve_time : float           wall-clock seconds
    """
    x_sol:      list[np.ndarray]
    n_iter:     int
    converged:  bool
    conv_hist:  list[float] = field(default_factory=list)
    solve_time: float = 0.0

    @property
    def x_stacked(self) -> np.ndarray:
        """Stacked x* = [x_0*; …; x_{N-1}*]."""
        return np.concatenate(self.x_sol)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-agent best-response QP
# ─────────────────────────────────────────────────────────────────────────────

def _solve_agent_br_osqp(
    ai,
    lin: np.ndarray,
    A_stack: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve agent best-response QP via OSQP (cold-start, no warm-starting)."""
    m = A_stack.shape[0]

    P = sp.csc_matrix(np.triu(ai.Q))   # upper-triangular sparse
    A = sp.csc_matrix(A_stack)
    l_stack = -np.inf * np.ones(m)

    prob = _osqp_lib.OSQP()
    prob.setup(
        P, lin, A, l_stack, rhs,
        warm_starting=False,            # cold start — no reuse
        verbose=False,
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=10_000,
        adaptive_rho=True,
        polish=False,
    )
    res = prob.solve()

    if res.info.status in ("solved", "solved_inaccurate") and res.x is not None:
        return res.x
    # OSQP failed — silently fall back to SLSQP
    return _solve_agent_br_slsqp(ai, lin, A_stack, rhs)


def _solve_agent_br_slsqp(
    ai,
    lin: np.ndarray,
    A_stack: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve agent best-response QP via scipy SLSQP."""
    def obj(x):
        return 0.5 * x @ ai.Q @ x + lin @ x

    def jac(x):
        return ai.Q @ x + lin

    r = rhs.copy()
    constraints = [{
        'type': 'ineq',
        'fun':  lambda x, r=r:  r - A_stack @ x,
        'jac':  lambda x:          -A_stack,
    }]

    res = minimize(
        obj, np.zeros(ai.n_x), jac=jac,
        method='SLSQP',
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 300, 'disp': False},
    )
    return res.x


def _build_state_rhs_br(ai, p, x_list, game, i):
    """Build RHS for state constraints on agent i given fixed U_{-i}."""
    x_lb_rep = ai.x_lb_rep
    x_ub_rep = ai.x_ub_rep
    nx = game.n_p

    others = [j for j in range(game.N) if j != i]
    U_neg = np.concatenate([x_list[j] for j in others])

    M_theta_i = ai.M_theta  # dimpc ordering: [Phi_x | Gamma_others]
    # Reorder to GNE: [Gamma_others | Phi_x]
    M_theta_gne = np.hstack([M_theta_i[:, nx:], M_theta_i[:, :nx]])
    theta_i = np.concatenate([U_neg, p])
    X_coupling = M_theta_gne @ theta_i

    rhs_upper = x_ub_rep - X_coupling
    rhs_lower = -x_lb_rep + X_coupling
    return np.concatenate([rhs_upper, rhs_lower])


def _solve_agent_br(
    game: GNEGame,
    i: int,
    p: np.ndarray,
    x_list: list[np.ndarray],
    qp_solver: str = "osqp",
) -> np.ndarray:
    """
    Solve agent i's best-response QP treating x_{-i} = x_list as fixed.

    min  ½ x_i^T Q_i x_i + (F_cross_i U_{-i} + c_i + F_i p)^T x_i
    s.t. A_loc_i x_i ≤ b_loc_i + S_loc_i p          (input bounds)
         ±Γ_i x_i ≤ rhs_state_i                      (state bounds, if present)
         C_i x_i  ≤ rhs_coup_i                       (coupling, if present)

    Parameters
    ----------
    qp_solver : "osqp" (default) or "slsqp"
    """
    ai = game.agents[i]

    # Linear cost term
    if ai.F_cross is not None:
        others = [j for j in range(game.N) if j != i]
        U_neg = np.concatenate([x_list[j] for j in others])
        lin = ai.c + ai.F_cross @ U_neg + ai.F @ p
    else:
        lin = ai.c + ai.F @ p

    # Build constraint stack
    A_parts = [ai.A_loc]
    rhs_parts = [ai.b_loc + ai.S_loc @ p]

    # State constraints
    if ai.has_state_constraints:
        rhs_state = _build_state_rhs_br(ai, p, x_list, game, i)
        n_half = len(rhs_state) // 2
        A_parts.append(ai.Gamma_self)
        A_parts.append(-ai.Gamma_self)
        rhs_parts.append(rhs_state[:n_half])
        rhs_parts.append(rhs_state[n_half:])

    # Coupling constraint
    if game.n_coupling > 0 and ai.C is not None:
        rhs_coup_i = game.d + game.S_coup @ p
        for j in range(game.N):
            if j != i and game.agents[j].C is not None:
                rhs_coup_i = rhs_coup_i - game.agents[j].C @ x_list[j]
        A_parts.append(ai.C)
        rhs_parts.append(rhs_coup_i)

    A_stack = np.vstack(A_parts)
    rhs     = np.concatenate(rhs_parts)

    if qp_solver == "osqp" and _OSQP_AVAILABLE:
        return _solve_agent_br_osqp(ai, lin, A_stack, rhs)
    else:
        return _solve_agent_br_slsqp(ai, lin, A_stack, rhs)


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def pg_solve(
    game: GNEGame,
    p: np.ndarray,
    max_iter: int = 5000,
    tol: float = 1e-6,
    verbose: bool = False,
    qp_solver: str = "osqp",
    x_init: list[np.ndarray] | None = None,
) -> PGResult:
    """
    Solve the GNE for a specific parameter p using Jacobi Best-Response.

    Parameters
    ----------
    game      : GNEGame
    p         : parameter vector (n_p,)
    max_iter  : maximum iterations
    tol       : stopping threshold for ‖x^{k+1} − x^k‖
    verbose   : print per-iteration summary
    qp_solver : "osqp" (default) or "slsqp" — inner QP solver for BR step.
    x_init    : warm-start list of x_i^0, one per agent.  If None, starts at zeros.

    Returns
    -------
    PGResult
    """
    if qp_solver == "osqp" and not _OSQP_AVAILABLE:
        import warnings
        warnings.warn("OSQP not installed; falling back to SLSQP. "
                      "Install with: pip install osqp", stacklevel=2)
        qp_solver = "slsqp"

    p = np.asarray(p, dtype=float).ravel()
    N = game.N

    # Warm-start from previous solution or cold-start at zeros
    if x_init is not None and len(x_init) == N:
        x_list: list[np.ndarray] = [np.asarray(x, dtype=float).copy() for x in x_init]
    else:
        x_list: list[np.ndarray] = [np.zeros(game.agents[i].n_x) for i in range(N)]

    conv_hist: list[float] = []
    converged = False
    delta     = np.inf

    t0 = time.perf_counter()

    for k in range(max_iter):
        # Jacobi: all agents solve BR simultaneously using x^k
        x_new = [
            _solve_agent_br(game, i, p, x_list, qp_solver=qp_solver)
            for i in range(N)
        ]

        # Convergence metric: ‖x^{k+1} − x^k‖
        delta = float(
            np.linalg.norm(
                np.concatenate(x_new) - np.concatenate(x_list)
            )
        )
        conv_hist.append(delta)
        x_list = x_new

        if verbose and (k % 50 == 0 or k == max_iter - 1):
            print(f"  [BR] iter {k:4d}  delta={delta:.2e}")

        if delta < tol:
            converged = True
            break

    return PGResult(
        x_sol=x_list,
        n_iter=k + 1,
        converged=converged,
        conv_hist=conv_hist,
        solve_time=time.perf_counter() - t0,
    )
