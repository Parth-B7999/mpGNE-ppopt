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
    → solved with OSQP (default) or scipy SLSQP (fallback)

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

QP Solver selection
───────────────────
Pass  qp_solver="osqp"   (default) for OSQP — fast, no warm-starting.
Pass  qp_solver="slsqp"  to fall back to scipy SLSQP.
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

def _solve_agent_xupdate_osqp(
    ai,
    l_i: np.ndarray,
    Q_aug: np.ndarray,
    A_stack: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve agent x-update QP via OSQP (cold-start, no warm-starting)."""
    n = ai.n_x
    m = A_stack.shape[0]

    P = sp.csc_matrix(np.triu(Q_aug))          # upper-triangular sparse
    A = sp.csc_matrix(A_stack)                 # (m, n)
    lb = -np.inf * np.ones(m)                  # one-sided: A x <= rhs

    prob = _osqp_lib.OSQP()
    prob.setup(
        P, l_i, A, lb, rhs,
        warm_starting=False,    # cold start — no reuse between calls
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
    return _solve_agent_xupdate_slsqp(ai, l_i, Q_aug, A_stack, rhs)


def _solve_agent_xupdate_slsqp(
    ai,
    l_i: np.ndarray,
    Q_aug: np.ndarray,
    A_stack: np.ndarray,
    rhs: np.ndarray,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Solve agent x-update QP via scipy SLSQP."""
    def obj(x):
        return 0.5 * x @ Q_aug @ x + l_i @ x

    def jac(x):
        return Q_aug @ x + l_i

    r = rhs.copy()
    constraints = [{
        'type': 'ineq',
        'fun':  lambda x, r=r:  r - A_stack @ x,
        'jac':  lambda x:          -A_stack,
    }]

    x_init = np.zeros(ai.n_x) if x0 is None else x0.copy()
    res = minimize(
        obj, x_init, jac=jac,
        method='SLSQP',
        constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 200, 'disp': False},
    )
    return res.x


def _build_state_constraint_rhs(ai, p, x_list, game, i):
    """Build RHS for state constraints on agent i given fixed U_{-i}."""
    x_lb_rep  = ai.x_lb_rep        # (Np*nx,)
    x_ub_rep  = ai.x_ub_rep        # (Np*nx,)
    nx = game.n_p

    # Assemble U_{-i} in the correct order
    others = [j for j in range(game.N) if j != i]
    U_neg = np.concatenate([x_list[j] for j in others])  # (n_x_neg,)

    # M_theta = [Phi_x | Gamma_{j1} | ...] in dimpc ordering
    # Reorder to GNE ordering [Gamma_others | Phi_x] to match θ_i = [U_{-i}; p]
    M_theta_i = ai.M_theta          # (Np*nx, nx + n_x_neg)
    M_theta_gne = np.hstack([M_theta_i[:, nx:], M_theta_i[:, :nx]])

    theta_i = np.concatenate([U_neg, p])                    # (n_x_neg + n_p,)
    X_coupling = M_theta_gne @ theta_i                      # (Np*nx,)
    # State constraints: x_lb_rep <= Gamma_i U_i + X_coupling <= x_ub_rep
    # → ±Gamma_i U_i <= ±(x_{ub/lb}_rep - X_coupling)
    rhs_upper = x_ub_rep - X_coupling
    rhs_lower = -x_lb_rep + X_coupling
    return np.concatenate([rhs_upper, rhs_lower])


def _solve_agent_xupdate(
    game: GNEGame,
    i: int,
    p: np.ndarray,
    z_i: np.ndarray | None,
    lambda_i: np.ndarray | None,
    rho: float,
    x_list: list[np.ndarray] | None = None,
    x0: np.ndarray | None = None,
    qp_solver: str = "osqp",
) -> np.ndarray:
    """
    Solve agent i's x-update QP.

    With coupling:  min  ½ x_i^T (Q_i + ρ C_i^T C_i) x_i + l_i^T x_i
                    s.t. A_loc_i x_i ≤ b_loc_i + S_loc_i p

    Without coupling: min ½ x_i^T Q_i x_i + (c_i + F_i p)^T x_i
                      s.t. A_loc_i x_i ≤ b_loc_i + S_loc_i p
                           ±Γ_i x_i ≤ ±(x_{ub/lb}_rep - X_coupling)

    Parameters
    ----------
    qp_solver : "osqp" (default, fast, cold-start) or "slsqp"
    """
    ai = game.agents[i]

    # Augmented Hessian
    if ai.C is not None and game.n_coupling > 0:
        Q_aug = ai.Q + rho * ai.C.T @ ai.C
        l_i = ai.c + ai.F @ p + ai.C.T @ lambda_i - rho * ai.C.T @ z_i
    else:
        Q_aug = ai.Q.copy()
        # Cost: ½ x_i^T Q_i x_i + (F_cross_i @ U_{-i} + F_i @ p)^T x_i
        # The F_cross term depends on U_{-i} which is in x_list
        if ai.F_cross is not None and x_list is not None:
            others = [j for j in range(game.N) if j != i]
            U_neg = np.concatenate([x_list[j] for j in others])
            l_i = ai.c + ai.F_cross @ U_neg + ai.F @ p
        else:
            l_i = ai.c + ai.F @ p

    # Build constraint stack
    A_parts = [ai.A_loc]
    rhs_parts = [ai.b_loc + ai.S_loc @ p]

    # State constraints
    if ai.has_state_constraints and x_list is not None:
        rhs_state = _build_state_constraint_rhs(ai, p, x_list, game, i)
        n_half = len(rhs_state) // 2
        A_parts.append(ai.Gamma_self)
        A_parts.append(-ai.Gamma_self)
        rhs_parts.append(rhs_state[:n_half])    # upper bound
        rhs_parts.append(rhs_state[n_half:])    # lower bound

    # Coupling constraint (if present)
    if ai.C is not None and game.n_coupling > 0 and x_list is not None:
        rhs_coup = game.d + game.S_coup @ p
        for j in range(game.N):
            if j != i and game.agents[j].C is not None:
                rhs_coup = rhs_coup - game.agents[j].C @ x_list[j]
        A_parts.append(ai.C)
        rhs_parts.append(rhs_coup)

    A_stack = np.vstack(A_parts)
    rhs     = np.concatenate(rhs_parts)

    if qp_solver == "osqp" and _OSQP_AVAILABLE:
        return _solve_agent_xupdate_osqp(ai, l_i, Q_aug, A_stack, rhs)
    else:
        return _solve_agent_xupdate_slsqp(ai, l_i, Q_aug, A_stack, rhs, x0=x0)


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
    """
    N   = game.N
    rhs = game.d + game.S_coup @ p

    z_unc = [
        game.agents[i].C @ x_list[i] + lambdas[i] / rho
        for i in range(N)
    ]

    agg    = sum(z_unc)
    excess = np.maximum(0.0, agg - rhs)
    shift  = excess / N

    return [z_i - shift for z_i in z_unc]


# ─────────────────────────────────────────────────────────────────────────────
#  λ-update
# ─────────────────────────────────────────────────────────────────────────────

def _lambda_update(
    game: GNEGame,
    x_list: list[np.ndarray],
    z_list: list[np.ndarray],
    lambdas: list[np.ndarray],
    rho: float,
) -> list[np.ndarray]:
    """λ_i^{k+1} = λ_i^k + ρ (C_i x_i^{k+1} - z_i^{k+1})"""
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
    rho: float = 0.5,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    x_init: list[np.ndarray] | None = None,
    qp_solver: str = "osqp",
) -> ADMMResult:
    """
    Solve the GNE for a specific parameter p using ADMM.

    With coupling constraint: standard ADMM (x/z/lambda updates).
    Without coupling: Jacobi best-response iteration (state constraints
    couple agents through dynamics).

    Parameters
    ----------
    game      : GNEGame
    p         : parameter vector (n_p,)
    rho       : ADMM penalty parameter (> 0).
    max_iter  : maximum iterations
    tol       : stopping threshold for max(primal_res, dual_res) or ||x_new - x||
    verbose   : print per-iteration summary
    x_init    : warm-start list of x_i^0  (if None, initialise at zeros)
    qp_solver : "osqp" (default) or "slsqp" — inner QP solver for x-update.

    Returns
    -------
    ADMMResult
    """
    if qp_solver == "osqp" and not _OSQP_AVAILABLE:
        import warnings
        warnings.warn("OSQP not installed; falling back to SLSQP. "
                      "Install with: pip install osqp", stacklevel=2)
        qp_solver = "slsqp"

    p   = np.asarray(p, dtype=float).ravel()
    N   = game.N
    has_coupling = (game.n_coupling > 0)

    # ── initialise ────────────────────────────────────────────────────────────
    if x_init is None:
        x_list = [np.zeros(game.agents[i].n_x) for i in range(N)]
    elif isinstance(x_init, np.ndarray) and x_init.ndim == 1:
        x_list = []
        offset = 0
        for i in range(N):
            nx_i = game.agents[i].n_x
            x_list.append(x_init[offset:offset+nx_i])
            offset += nx_i
    else:
        x_list = [x.copy() for x in x_init]

    if has_coupling:
        rhs = game.d + game.S_coup @ p
        z_list  = [game.agents[i].C @ x_list[i] for i in range(N)]
        lambdas = [np.zeros(game.n_coupling) for _ in range(N)]
    else:
        rhs = None
        z_list = []
        lambdas = []

    primal_hist: list[float] = []
    dual_hist:   list[float] = []
    converged    = False
    primal_res   = np.inf
    dual_res     = np.inf

    t0 = time.perf_counter()

    for k in range(max_iter):
        if has_coupling:
            z_prev = [z.copy() for z in z_list]

        # ── x-update (each agent independently) ──────────────────────────────
        if has_coupling:
            x_new = [
                _solve_agent_xupdate(game, i, p, z_list[i], lambdas[i], rho,
                                     x_list=x_list, x0=x_list[i], qp_solver=qp_solver)
                for i in range(N)
            ]
        else:
            x_new = [
                _solve_agent_xupdate(game, i, p, None, None, rho,
                                     x_list=x_list, x0=x_list[i], qp_solver=qp_solver)
                for i in range(N)
            ]

        if has_coupling:
            # ── z-update (global projection) ─────────────────────────────────
            z_new = _z_update(game, p, x_new, lambdas, rho)
            # ── λ-update ─────────────────────────────────────────────────────
            lambdas = _lambda_update(game, x_new, z_new, lambdas, rho)
            z_list = z_new

        x_list = x_new

        # ── residuals / convergence ──────────────────────────────────────────
        if has_coupling:
            primal_res, dual_res = _compute_residuals(game, x_list, z_list, z_prev, rho)
            primal_hist.append(primal_res)
            dual_hist.append(dual_res)
            converged = max(primal_res, dual_res) < tol
        else:
            # Best-response convergence: ||x_new - x_old||
            if k > 0:
                delta = float(np.linalg.norm(
                    np.concatenate(x_new) - np.concatenate(x_prev_best)
                ))
                primal_hist.append(delta)
                dual_hist.append(0.0)
                converged = delta < tol
            else:
                primal_hist.append(np.inf)
                dual_hist.append(0.0)
            x_prev_best = [x.copy() for x in x_new]

        if verbose and (k % 50 == 0 or k == max_iter - 1 or converged):
            if has_coupling:
                agg  = sum(game.agents[i].C @ x_list[i] for i in range(N))
                viol = float(np.max(np.maximum(0.0, agg - rhs)))
                print(f"  [ADMM] iter {k:4d}  r={primal_res:.2e}  s={dual_res:.2e}"
                      f"  coupling_viol={viol:.2e}")
            else:
                print(f"  [ADMM/BR] iter {k:4d}  delta={primal_hist[-1]:.2e}")

        if converged:
            break

    solve_time = time.perf_counter() - t0

    if has_coupling:
        agg = sum(game.agents[i].C @ x_list[i] for i in range(N))
        coupling_viol = float(np.max(np.maximum(0.0, agg - rhs)))
    else:
        coupling_viol = 0.0

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
    qp_solver: str = "osqp",
) -> list[ADMMResult]:
    """
    Solve GNE for each row of p_grid using ADMM.

    Warm-starts each solve from the previous solution.

    Parameters
    ----------
    game      : GNEGame
    p_grid    : (n_samples, n_p) array of parameter vectors
    qp_solver : "osqp" (default) or "slsqp"
    """
    results = []
    x_warm  = None

    for idx, p in enumerate(p_grid):
        res = admm_solve(game, p, rho=rho, max_iter=max_iter,
                         tol=tol, verbose=False, x_init=x_warm,
                         qp_solver=qp_solver)
        results.append(res)
        x_warm = res.x_sol

        if verbose:
            status = "OK" if res.converged else "MAX_ITER"
            print(f"  [grid {idx:4d}/{len(p_grid)}]  {status}  "
                  f"iter={res.n_iter}  viol={res.coupling_violation:.2e}")

    return results
