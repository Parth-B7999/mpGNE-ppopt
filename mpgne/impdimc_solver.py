"""
impdimc_solver.py — ImpGNE: Iterative multiparametric Distributed MPC.

Implements the core iteration loop of Algorithm 1 from Saini et al. (2025),
WITHOUT Wegstein acceleration — plain Jacobi substitution using the
precomputed explicit mpQP solution maps.

Key Distinction from Jacobi BR (proj_grad_solver.py)
─────────────────────────────────────────────────────
  Jacobi BR  : solves a QP from scratch at every inner iteration.
               Cost per iteration: O(QP solve) × M agents  ≈ slow.

  ImpGNE  : uses the PRECOMPUTED explicit parametric solution maps
               (agent_sols critical regions).  Each agent's best-response
               is evaluated as a single affine function:
                   U_i*(θ_i) = A_cr · θ_i + b_cr
               Cost per iteration: O(matrix-vector multiply) × M  ≈ sub-ms.

Algorithm (plain Jacobi, no acceleration)
─────────────────────────────────────────
Given current state x(k) = p and precomputed agent_sols:

  Initialise: U_bar = [U_1; ...; U_M] = 0   (cold-start)

  For iter = 0, 1, 2, ...

    For each agent i (simultaneously):
      1. Build parametric vector:
            θ_i = [ p;  U_{-i} ]
         where U_{-i} = all OTHER agents' blocks from U_bar.

      2. Locate the active critical region CR_i in agent_sols[i]
         that contains θ_i  (hyperplane test: E_cr θ_i ≤ f_cr).

      3. Evaluate explicit best-response (affine map):
            U_i_new = A_cr · θ_i + b_cr

         If θ_i falls outside all known CRs → ADMM fallback for this agent.

    U_bar_new = concat(U_i_new for i in 0..M-1)

    δ = ‖U_bar_new − U_bar‖
    If δ < tol → converged, stop.

    U_bar ← U_bar_new

Reference
─────────
Saini, D., Ramamoorthy, K. S., & Paulen, R. (2025).
"Implicit-function-theorem-based multiparametric DiMPC for Cooperative
Distributed MPC using Explicit MPC for Individual Agents", CCE 2025.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .game import GNEGame


# ─────────────────────────────────────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IMPDiMPCResult:
    """
    Outcome of one ImpGNE run.

    Attributes
    ----------
    x_sol      : list[ndarray]   x_i* per agent (final U_bar split into blocks)
    n_iter     : int             outer iterations performed
    converged  : bool            True if δ < tol before max_iter
    conv_hist  : list[float]     ‖U_bar^{p+1} − U_bar^p‖ per iteration
    n_fallback : int             number of agent-steps that used ADMM fallback
    solve_time : float           wall-clock seconds
    """
    x_sol:      list[np.ndarray]
    n_iter:     int
    converged:  bool
    conv_hist:  list[float] = field(default_factory=list)
    n_fallback: int = 0
    solve_time: float = 0.0

    @property
    def x_stacked(self) -> np.ndarray:
        """Stacked x* = [x_0*; …; x_{N-1}*]."""
        return np.concatenate(self.x_sol)


# ─────────────────────────────────────────────────────────────────────────────
#  Critical-region lookup (one agent)
# ─────────────────────────────────────────────────────────────────────────────

def _locate_cr(agent_sol, theta: np.ndarray) -> int | None:
    """
    Find the index of the first critical region in agent_sol whose
    polyhedron contains theta  (E · theta <= f, all rows).

    Returns None if no region is feasible.
    """
    best_idx = None
    best_viol = np.inf
    for idx, cr in enumerate(agent_sol.regions):
        viol = float(np.max(cr.E @ theta - cr.f))
        if viol <= 0.0:
            return idx                  # exact membership → return immediately
        if viol < best_viol:            # track nearest region for fallback
            best_viol, best_idx = viol, idx
    # If nothing is exactly feasible, return None (caller handles fallback)
    return None


def _evaluate_cr(cr, theta: np.ndarray) -> np.ndarray:
    """
    Evaluate the explicit affine solution at theta:
        U* = A_cr · theta + b_cr
    """
    return cr.A @ theta + cr.b


# ─────────────────────────────────────────────────────────────────────────────
#  ADMM fallback (one agent, cheap single-agent QP via scipy)
# ─────────────────────────────────────────────────────────────────────────────

def _admm_fallback_agent(
    game: GNEGame,
    i: int,
    p: np.ndarray,
    U_bar: np.ndarray,
    n_x_list: list[int],
) -> np.ndarray:
    """
    Solve agent i's best-response QP via scipy SLSQP when the explicit map
    does not cover the current operating point (fallback only).

    Coupling RHS = d + S_coup p - Σ_{j≠i} C_j U_j
    """
    from scipy.optimize import minimize
    ai = game.agents[i]

    lin      = ai.c + ai.F @ p
    rhs_loc  = ai.b_loc + ai.S_loc @ p
    rhs_coup = game.d + game.S_coup @ p

    off = 0
    for j, nx_j in enumerate(n_x_list):
        if j != i:
            rhs_coup = rhs_coup - ai.C @ U_bar[off:off+nx_j] \
                if False else rhs_coup - game.agents[j].C @ U_bar[off:off+nx_j]
        off += nx_j

    def obj(x): return 0.5 * x @ ai.Q @ x + lin @ x
    def jac(x): return ai.Q @ x + lin

    r_loc, r_coup = rhs_loc.copy(), rhs_coup.copy()
    constraints = [
        {'type': 'ineq', 'fun': lambda x, r=r_loc:  r - ai.A_loc @ x,
                          'jac': lambda x:           -ai.A_loc},
        {'type': 'ineq', 'fun': lambda x, r=r_coup: r - ai.C @ x,
                          'jac': lambda x:           -ai.C},
    ]
    res = minimize(obj, np.zeros(ai.n_x), jac=jac, method='SLSQP',
                   constraints=constraints,
                   options={'ftol': 1e-10, 'maxiter': 300, 'disp': False})
    return res.x


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def impdimc_solve(
    game: GNEGame,
    p: np.ndarray,
    agent_sols: list,
    max_iter: int = 200,
    tol: float = 1e-4,
    verbose: bool = False,
) -> IMPDiMPCResult:
    """
    Solve the GNE for parameter p using ImpGNE (plain Jacobi, no Wegstein).

    Uses precomputed explicit mpQP maps (agent_sols) for sub-ms per-agent
    best-response evaluation.  Falls back to SLSQP for agents whose
    operating point lies outside all known critical regions.

    Parameters
    ----------
    game       : GNEGame
    p          : current state / parameter vector  (n_p,)
    agent_sols : list of mpQP solution objects (one per agent),
                 as returned by solve_all_agents_mp().
                 Each entry has .regions, each region has .E, .f, .A, .b
    max_iter   : maximum Jacobi iterations
    tol        : convergence threshold  ‖U_bar^{p+1} − U_bar^p‖ < tol
    verbose    : print per-iteration info

    Returns
    -------
    IMPDiMPCResult
    """
    p = np.asarray(p, dtype=float).ravel()
    N = game.N

    n_x_list = [game.agents[i].n_x for i in range(N)]
    n_total  = sum(n_x_list)
    offsets  = np.concatenate([[0], np.cumsum(n_x_list)])

    # ── Cold-start at zeros ───────────────────────────────────────────────────
    U_bar = np.zeros(n_total)

    conv_hist:  list[float] = []
    converged  = False
    delta      = np.inf
    n_fallback = 0

    t0 = time.perf_counter()

    for it in range(max_iter):

        U_new_parts = []

        for i in range(N):
            # ── Build θ_i = [p;  U_{-i}] ─────────────────────────────────────
            U_neg_i_parts = [
                U_bar[offsets[j]:offsets[j+1]]
                for j in range(N) if j != i
            ]
            theta_i = np.concatenate([p] + U_neg_i_parts)   # (n_p + Σ_{j≠i} n_x_j,)

            # ── Locate critical region ────────────────────────────────────────
            cr_idx = _locate_cr(agent_sols[i], theta_i)

            if cr_idx is not None:
                # Explicit affine evaluation — sub-ms
                U_i_new = _evaluate_cr(agent_sols[i].regions[cr_idx], theta_i)
            else:
                # Fallback: solve QP for this agent
                U_i_new = _admm_fallback_agent(game, i, p, U_bar, n_x_list)
                n_fallback += 1

            U_new_parts.append(U_i_new)

        U_bar_new = np.concatenate(U_new_parts)

        # ── Convergence ───────────────────────────────────────────────────────
        delta = float(np.linalg.norm(U_bar_new - U_bar))
        conv_hist.append(delta)

        if verbose and (it % 20 == 0 or it == max_iter - 1):
            print(f"  [ImpGNE] iter {it:4d}  δ={delta:.2e}"
                  f"  fb={n_fallback}")

        U_bar = U_bar_new

        if delta < tol:
            converged = True
            break

    # ── Split result into per-agent blocks ────────────────────────────────────
    x_sol = [U_bar[offsets[i]:offsets[i+1]] for i in range(N)]

    return IMPDiMPCResult(
        x_sol=x_sol,
        n_iter=it + 1,
        converged=converged,
        conv_hist=conv_hist,
        n_fallback=n_fallback,
        solve_time=time.perf_counter() - t0,
    )
