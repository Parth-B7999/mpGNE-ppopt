"""
admm_solver.py — Iterative ADMM baseline solver for GNE problems.

For a given parameter p, finds x*(p) iteratively using ADMM (Alternating
Direction Method of Multipliers).  This is the online iterative baseline
that the explicit mpGNE and FACET-mpGNE methods replace.

Math reference: operator splitting approach for GNE (Section I, paper).

Problem
───────
Find x* = (x_1*, ..., x_N*) such that for each agent i:
    x_i* ∈ argmin_{x_i} J_i(x_i, p)
           s.t.  A_loc_i x_i ≤ b_loc_i + S_loc_i p         (local)
                 sum_j C_j x_j ≤ d + S_coup p               (coupling, shared)

ADMM Splitting
──────────────
Introduce local coupling copies  z_i = C_i x_i  for each agent i.

Augmented Lagrangian (λ_i: dual for equality C_i x_i = z_i):
    L_ρ = Σ_i [J_i(x_i) - λ_i^T C_i x_i + (ρ/2)‖z_i - C_i x_i‖²]
    s.t.  Σ_i z_i ≤ d + S_coup p

Iterations
──────────
x-update  (each agent i, parallelisable):
    min  ½ x_i^T (Q_i + ρ C_i^T C_i) x_i + l_i^T x_i
    s.t. A_loc_i x_i ≤ b_loc_i + S_loc_i p
    where  l_i = c_i + F_i p - C_i^T λ_i^k - ρ C_i^T z_i^k
    → solved with scipy SLSQP

z-update  (global projection onto coupling set):
    z_i^{unc} = C_i x_i^{k+1} + λ_i^k / ρ
    Project {z_i} onto {Σ_i z_i ≤ d + S_coup p} element-wise:
      excess_r = max(0, Σ_i z_{i,r}^{unc} - rhs_r)
      z_{i,r} = z_{i,r}^{unc} - excess_r / N          (uniform shift)

λ-update  (dual for coupling equality):
    λ_i^{k+1} = λ_i^k + ρ (C_i x_i^{k+1} - z_i^{k+1})

Stopping criterion
──────────────────
Primal residual:  r = ‖concat_i(C_i x_i - z_i)‖
Dual residual:    s = ρ ‖concat_i C_i^T (z_i^{k+1} - z_i^k)‖
Stop when max(r, s) < tol.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .game import GNEGame


# ─────────────────────────────────────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ADMMResult:
    """
    Outcome of one ADMM run for a specific parameter p.

    Attributes
    ----------
    x_sol : list[ndarray]      x_i* for each agent i, shape (n_x_i,)
    z_sol : list[ndarray]      z_i* = C_i x_i* at convergence, shape (n_coupling,)
    lambda_sol : list[ndarray] final dual variables λ_i, shape (n_coupling,)
    n_iter : int               iterations performed
    converged : bool           True if stopping criterion met before max_iter
    primal_res : float         ‖r‖ at final iteration
    dual_res : float           ‖s‖ at final iteration
    coupling_violation : float max(0, Σ_i C_i x_i* - d - S_coup p)  (scalar)
    primal_hist : list[float]  ‖r^k‖ per iteration (for convergence plots)
    dual_hist   : list[float]  ‖s^k‖ per iteration
    solve_time  : float        wall-clock seconds
    """
    x_sol:              list[np.ndarray]
    z_sol:              list[np.ndarray]
    lambda_sol:         list[np.ndarray]
    n_iter:             int
    converged:          bool
    primal_res:         float
    dual_res:           float
    coupling_violation: float
    primal_hist:        list[float] = field(default_factory=list)
    dual_hist:          list[float] = field(default_factory=list)
    solve_time:         float = 0.0

    @property
    def x_stacked(self) -> np.ndarray:
        """Stacked x* = [x_0*; ...; x_{N-1}*]."""
        return np.concatenate(self.x_sol)


# ─────────────────────────────────────────────────────────────────────────────
#  x-update: each agent's augmented QP
# ─────────────────────────────────────────────────────────────────────────────

def _solve_agent_xupdate(
    game: GNEGame,
    i: int,
    p: np.ndarray,
    z_i: np.ndarray,
    lambda_i: np.ndarray,
    rho: float,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve agent i's x-update QP.

    min  ½ x_i^T (Q_i + ρ C_i^T C_i) x_i + l_i^T x_i
    s.t. A_loc_i x_i ≤ b_loc_i + S_loc_i p

    where  l_i = c_i + F_i p + C_i^T λ_i - ρ C_i^T z_i

    Parameters
    ----------
    game     : GNEGame
    i        : agent index
    p        : parameter vector (n_p,)
    z_i      : current z_i^k  (n_coupling,)
    lambda_i : current λ_i^k  (n_coupling,)
    rho      : ADMM penalty parameter
    x0       : warm-start (n_x_i,), zeros if None

    Returns
    -------
    x_i^{k+1}  (n_x_i,)
    """
    ai = game.agents[i]

    # Augmented Hessian: Q_i + ρ C_i^T C_i
    Q_aug = ai.Q + rho * ai.C.T @ ai.C            # (n_x_i, n_x_i)

    # Augmented linear term: c_i + F_i p + C_i^T λ_i - ρ C_i^T z_i
    # Boyd et al. 2010 ADMM: L_ρ = f(x) + λ^T(Ax - z) + ρ/2||Ax - z||^2
    # ∂/∂x_i: c_i + F_i p + C_i^T λ_i - ρ C_i^T z_i + ρ C_i^T C_i x_i
    l_i = ai.c + ai.F @ p + ai.C.T @ lambda_i - rho * ai.C.T @ z_i  # (n_x_i,)

    # Local constraint RHS: b_loc + S_loc p
    rhs = ai.b_loc + ai.S_loc @ p   # (n_loc,)

    def obj(x):
        return 0.5 * x @ Q_aug @ x + l_i @ x

    def jac(x):
        return Q_aug @ x + l_i

    # scipy SLSQP: inequality constraints as g(x) >= 0  →  rhs - A_loc x >= 0
    constraints = [{
        'type': 'ineq',
        'fun':  lambda x: rhs - ai.A_loc @ x,
        'jac':  lambda x: -ai.A_loc,
    }]

    x_init = np.zeros(ai.n_x) if x0 is None else x0.copy()
    res = minimize(
        obj, x_init, jac=jac,
        method='SLSQP',
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 200, 'disp': False},
    )
    return res.x


# ─────────────────────────────────────────────────────────────────────────────
#  z-update: projection onto coupling set
# ─────────────────────────────────────────────────────────────────────────────

def _z_update(
    game: GNEGame,
    p: np.ndarray,
    x_list: list[np.ndarray],
    lambdas: list[np.ndarray],
    rho: float,
) -> list[np.ndarray]:
    """
    z-update: project unconstrained minimisers onto coupling set.

    Unconstrained minimiser for each agent:
        z_i^{unc} = C_i x_i^{k+1} + λ_i^k / ρ

    Project {z_i} onto {Σ_i z_i ≤ d + S_coup p}  row-wise uniform shift:
        excess_r = max(0,  Σ_i z_{i,r}^{unc}  -  rhs_r)
        z_{i,r}  = z_{i,r}^{unc}  -  excess_r / N

    Returns
    -------
    list of z_i^{k+1}  (n_coupling,) for each agent
    """
    N   = game.N
    rhs = game.d + game.S_coup @ p   # (n_coupling,)

    # Boyd ADMM z-update: min -λ_i^T z_i + ρ/2||C_i x_i - z_i||^2
    # stationarity: ρ(z_i - C_i x_i) - λ_i = 0  →  z_i^{unc} = C_i x_i + λ_i/ρ
    z_unc = [
        game.agents[i].C @ x_list[i] + lambdas[i] / rho
        for i in range(N)
    ]

    agg = sum(z_unc)                   # Σ_i z_i^{unc}  (n_coupling,)
    excess = np.maximum(0.0, agg - rhs)# element-wise excess  (n_coupling,)
    shift  = excess / N                # uniform shift per agent

    return [z_i - shift for z_i in z_unc]


# ─────────────────────────────────────────────────────────────────────────────
#  λ-update: dual variable for equality C_i x_i = z_i
# ─────────────────────────────────────────────────────────────────────────────

def _lambda_update(
    game: GNEGame,
    x_list: list[np.ndarray],
    z_list: list[np.ndarray],
    lambdas: list[np.ndarray],
    rho: float,
) -> list[np.ndarray]:
    """
    λ_i^{k+1} = λ_i^k + ρ (C_i x_i^{k+1} - z_i^{k+1})
    """
    return [
        lambdas[i] + rho * (game.agents[i].C @ x_list[i] - z_list[i])
        for i in range(game.N)
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Residuals
# ─────────────────────────────────────────────────────────────────────────────

def _compute_residuals(
    game: GNEGame,
    x_list: list[np.ndarray],
    z_list: list[np.ndarray],
    z_prev: list[np.ndarray],
    rho: float,
) -> tuple[float, float]:
    """
    Primal residual:  r = ‖concat_i(C_i x_i - z_i)‖
    Dual residual:    s = ρ ‖concat_i C_i^T (z_i^{k+1} - z_i^k)‖
    """
    r_parts = [game.agents[i].C @ x_list[i] - z_list[i]
               for i in range(game.N)]
    s_parts = [rho * game.agents[i].C.T @ (z_list[i] - z_prev[i])
               for i in range(game.N)]

    primal = float(np.linalg.norm(np.concatenate(r_parts)))
    dual   = float(np.linalg.norm(np.concatenate(s_parts)))
    return primal, dual


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def admm_solve(
    game: GNEGame,
    p: np.ndarray,
    rho: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    x_init: list[np.ndarray] | None = None,
) -> ADMMResult:
    """
    Solve the GNE for a specific parameter p using ADMM.

    Parameters
    ----------
    game     : GNEGame
    p        : parameter vector (n_p,)
    rho      : ADMM penalty parameter (> 0).
               Too small → slow convergence; too large → oscillation.
               Rule of thumb: rho ≈ sqrt(λ_max(Q_i)).
    max_iter : maximum ADMM iterations
    tol      : stopping threshold for max(primal_res, dual_res)
    verbose  : print per-iteration summary
    x_init   : warm-start list of x_i^0  (if None, initialise at zeros)

    Returns
    -------
    ADMMResult
    """
    p   = np.asarray(p, dtype=float).ravel()
    N   = game.N
    rhs = game.d + game.S_coup @ p   # coupling RHS for this p

    # ── initialise ────────────────────────────────────────────────────────────
    if x_init is None:
        x_list = [np.zeros(game.agents[i].n_x) for i in range(N)]
    elif isinstance(x_init, np.ndarray) and x_init.ndim == 1:
        # Split stacked array
        x_list = []
        offset = 0
        for i in range(N):
            nx_i = game.agents[i].n_x
            x_list.append(x_init[offset:offset+nx_i])
            offset += nx_i
    else:
        # Assume it's a list of arrays
        x_list = [x.copy() for x in x_init]

    z_list     = [game.agents[i].C @ x_list[i] for i in range(N)]
    lambdas    = [np.zeros(game.n_coupling) for _ in range(N)]

    primal_hist: list[float] = []
    dual_hist:   list[float] = []
    converged    = False
    primal_res   = np.inf
    dual_res     = np.inf

    t0 = time.perf_counter()

    for k in range(max_iter):
        z_prev = [z.copy() for z in z_list]

        # ── x-update (each agent independently) ──────────────────────────────
        x_new = [
            _solve_agent_xupdate(game, i, p, z_list[i], lambdas[i], rho,
                                  x0=x_list[i])
            for i in range(N)
        ]

        # ── z-update (global projection) ─────────────────────────────────────
        z_new = _z_update(game, p, x_new, lambdas, rho)

        # ── λ-update ──────────────────────────────────────────────────────────
        lambdas = _lambda_update(game, x_new, z_new, lambdas, rho)

        x_list = x_new
        z_list = z_new

        # ── residuals ─────────────────────────────────────────────────────────
        primal_res, dual_res = _compute_residuals(game, x_list, z_list, z_prev, rho)
        primal_hist.append(primal_res)
        dual_hist.append(dual_res)

        if verbose and (k % 50 == 0 or k == max_iter - 1):
            agg = sum(game.agents[i].C @ x_list[i] for i in range(N))
            viol = float(np.max(np.maximum(0.0, agg - rhs)))
            print(f"  [ADMM] iter {k:4d}  r={primal_res:.2e}  s={dual_res:.2e}"
                  f"  coupling_viol={viol:.2e}")

        if max(primal_res, dual_res) < tol:
            converged = True
            break

    solve_time = time.perf_counter() - t0

    # ── coupling violation at termination ────────────────────────────────────
    agg = sum(game.agents[i].C @ x_list[i] for i in range(N))
    coupling_viol = float(np.max(np.maximum(0.0, agg - rhs)))

    return ADMMResult(
        x_sol=x_list,
        z_sol=z_list,
        lambda_sol=lambdas,
        n_iter=k + 1,
        converged=converged,
        primal_res=primal_res,
        dual_res=dual_res,
        coupling_violation=coupling_viol,
        primal_hist=primal_hist,
        dual_hist=dual_hist,
        solve_time=solve_time,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep over parameter grid
# ─────────────────────────────────────────────────────────────────────────────

def admm_solve_grid(
    game: GNEGame,
    p_grid: np.ndarray,
    rho: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
) -> list[ADMMResult]:
    """
    Solve GNE for each row of p_grid using ADMM.

    Warm-starts each solve from the previous solution.

    Parameters
    ----------
    game   : GNEGame
    p_grid : (n_samples, n_p) array of parameter vectors
    rho, max_iter, tol : passed to admm_solve

    Returns
    -------
    list of ADMMResult, one per row of p_grid
    """
    results = []
    x_warm  = None

    for idx, p in enumerate(p_grid):
        res = admm_solve(game, p, rho=rho, max_iter=max_iter,
                         tol=tol, verbose=False, x_init=x_warm)
        results.append(res)
        x_warm = res.x_sol     # warm-start next solve

        if verbose:
            status = "OK" if res.converged else "MAX_ITER"
            print(f"  [grid {idx:4d}/{len(p_grid)}]  {status}  "
                  f"iter={res.n_iter}  viol={res.coupling_violation:.2e}")

    return results
