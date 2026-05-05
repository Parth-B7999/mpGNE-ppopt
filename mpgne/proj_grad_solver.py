"""
proj_grad_solver.py — Jacobi Best-Response (Projected Gradient) iterative
                       GNE baseline solver.

The simplest distributed iterative GNE method: at every iteration k each
agent *simultaneously* computes its best-response against the others' current
strategies (Jacobi update rule), which is equivalent to solving a projected
gradient step for a quadratic game.

Jacobi Best-Response Iteration
───────────────────────────────
    x_i^{k+1} = argmin  ½ x_i^T Q_i x_i  +  (c_i + F_i p)^T x_i
                s.t.    A_loc_i x_i ≤ b_loc_i + S_loc_i p          (local)
                        C_i x_i     ≤ d + S_coup p - Σ_{j≠i} C_j x_j^k
                                                                   (coupling)

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
    rhs_loc: np.ndarray,
    rhs_coup_i: np.ndarray,
) -> np.ndarray:
    """Solve agent best-response QP via OSQP (cold-start, no warm-starting)."""
    n      = ai.n_x
    m_loc  = len(rhs_loc)
    m_coup = len(rhs_coup_i)

    # Stack constraints: [A_loc; C_i] x <= [rhs_loc; rhs_coup_i]
    A_stack = sp.csc_matrix(np.vstack([ai.A_loc, ai.C]))
    u_stack = np.concatenate([rhs_loc, rhs_coup_i])
    l_stack = -np.inf * np.ones(m_loc + m_coup)

    P = sp.csc_matrix(np.triu(ai.Q))   # upper-triangular sparse

    prob = _osqp_lib.OSQP()
    prob.setup(
        P, lin, A_stack, l_stack, u_stack,
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
    return _solve_agent_br_slsqp(ai, lin, rhs_loc, rhs_coup_i)


def _solve_agent_br_slsqp(
    ai,
    lin: np.ndarray,
    rhs_loc: np.ndarray,
    rhs_coup_i: np.ndarray,
) -> np.ndarray:
    """Solve agent best-response QP via scipy SLSQP."""
    def obj(x):
        return 0.5 * x @ ai.Q @ x + lin @ x

    def jac(x):
        return ai.Q @ x + lin

    r_loc  = rhs_loc.copy()
    r_coup = rhs_coup_i.copy()
    constraints = [
        {
            'type': 'ineq',
            'fun':  lambda x, r=r_loc:  r - ai.A_loc @ x,
            'jac':  lambda x:          -ai.A_loc,
        },
        {
            'type': 'ineq',
            'fun':  lambda x, r=r_coup: r - ai.C @ x,
            'jac':  lambda x:          -ai.C,
        },
    ]

    res = minimize(
        obj, np.zeros(ai.n_x), jac=jac,
        method='SLSQP',
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 300, 'disp': False},
    )
    return res.x


def _solve_agent_br(
    game: GNEGame,
    i: int,
    p: np.ndarray,
    x_list: list[np.ndarray],
    qp_solver: str = "osqp",
) -> np.ndarray:
    """
    Solve agent i's best-response QP treating x_{-i} = x_list as fixed.

    min  ½ x_i^T Q_i x_i + (c_i + F_i p)^T x_i
    s.t. A_loc_i x_i ≤ b_loc_i + S_loc_i p          (local constraints)
         C_i x_i     ≤ rhs_coup_i                    (agent-i coupling slack)

    where  rhs_coup_i = d + S_coup p - Σ_{j≠i} C_j x_j

    Parameters
    ----------
    qp_solver : "osqp" (default) or "slsqp"
    """
    ai = game.agents[i]

    lin       = ai.c + ai.F @ p                          # (n_x_i,)
    rhs_loc   = ai.b_loc + ai.S_loc @ p                  # (n_loc,)
    rhs_coup_i = game.d + game.S_coup @ p                # (n_coupling,)
    for j in range(game.N):
        if j != i:
            rhs_coup_i = rhs_coup_i - game.agents[j].C @ x_list[j]

    if qp_solver == "osqp" and _OSQP_AVAILABLE:
        return _solve_agent_br_osqp(ai, lin, rhs_loc, rhs_coup_i)
    else:
        return _solve_agent_br_slsqp(ai, lin, rhs_loc, rhs_coup_i)


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
) -> PGResult:
    """
    Solve the GNE for a specific parameter p using Jacobi Best-Response.

    Always cold-starts at x = 0 (no warm-starting).

    Parameters
    ----------
    game      : GNEGame
    p         : parameter vector (n_p,)
    max_iter  : maximum iterations
    tol       : stopping threshold for ‖x^{k+1} − x^k‖
    verbose   : print per-iteration summary
    qp_solver : "osqp" (default) or "slsqp" — inner QP solver for BR step.
                Both use cold-start (no warm-starting within the QP solver).

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

    # Cold-start: all zeros
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
