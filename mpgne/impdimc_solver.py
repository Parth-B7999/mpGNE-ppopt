"""
impdimc_solver.py — ImpGNE: Iterative multiparametric GNE solver.

Plain Jacobi substitution using precomputed explicit mpQP solution maps.
No fallback needed — the mpQP solution partitions the entire parameter box
bounded by state and input limits, so θ_i = [U_{-i}; p] is always covered.

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
            θ_i = [U_{-i}; p]
         where U_{-i} = all OTHER agents' blocks from U_bar.

      2. Locate the active critical region CR_i in agent_sols[i]
         that contains θ_i  (hyperplane test: E_cr θ_i ≤ f_cr).

      3. Evaluate explicit best-response (affine map):
            U_i_new = A_cr · θ_i + b_cr

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
#  Critical-region lookup (one agent)
# ─────────────────────────────────────────────────────────────────────────────

def _locate_cr(agent_sol, theta: np.ndarray, tol: float = 1e-6) -> int:
    """
    Find the index of the critical region containing theta.

    The mpQP solution partitions the entire parameter box, so theta is
    always inside some region (up to floating-point tolerance). Returns
    the nearest region if no exact match — the affine law is continuous
    across region boundaries so the nearest region gives the correct answer.
    """
    best_idx = 0
    best_viol = np.inf
    for idx, cr in enumerate(agent_sol.regions):
        viol = float(np.max(cr.E @ theta - cr.f))
        if viol <= tol:
            return idx
        if viol < best_viol:
            best_viol, best_idx = viol, idx
    return best_idx


def _evaluate_cr(cr, theta: np.ndarray) -> np.ndarray:
    """
    Evaluate the explicit affine solution at theta:
        U* = A_cr · theta + b_cr
    """
    return cr.A @ theta + cr.b


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

    t0 = time.perf_counter()

    for it in range(max_iter):

        U_new_parts = []

        for i in range(N):
            # ── Build θ_i = [U_{-i}; p] ──────────────────────────────────────
            U_neg_i_parts = [
                U_bar[offsets[j]:offsets[j+1]]
                for j in range(N) if j != i
            ]
            theta_i = np.concatenate([p] + U_neg_i_parts)   # (n_p + Σ_{j≠i} n_x_j,)

            # ── Locate and evaluate — mpQP covers entire parameter box ───────
            cr_idx = _locate_cr(agent_sols[i], theta_i)
            U_i_new = _evaluate_cr(agent_sols[i].regions[cr_idx], theta_i)

            U_new_parts.append(U_i_new)

        U_bar_new = np.concatenate(U_new_parts)

        # ── Convergence ───────────────────────────────────────────────────────
        delta = float(np.linalg.norm(U_bar_new - U_bar))
        conv_hist.append(delta)

        if verbose and (it % 20 == 0 or it == max_iter - 1):
            print(f"  [ImpGNE] iter {it:4d}  δ={delta:.2e}")

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
        solve_time=time.perf_counter() - t0,
    )
