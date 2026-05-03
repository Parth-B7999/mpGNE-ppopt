"""
gne_combiner.py — Algorithm 1 (Steps 7-13) from Hall & Bemporad (2025).

Given each agent's explicit best-response mpQP solution (AgentSolution),
this module finds all valid GNE critical regions in p-space.

Analogy to dimpc IF-mpDiMPC
────────────────────────────
IF-mpDiMPC (_assemble_linear_system):
    Online, for a given x(k), search CR combinations → solve L U = R x(k) + d.

GNE combiner (here):
    Offline, parametric in p, for each CR combination → assemble M_x x* = M_p p + M_1
    then project the CR constraints to p-space and check for a non-empty polyhedron.

The key difference is that GNE combiner produces a PWA map  x*(p)  valid for ALL p,
not just for a single operating point.

Three-stage pipeline per combination C_k = (j_1, ..., j_N)
─────────────────────────────────────────────────────────────
1. _assemble_equilibrium_system   Build M_x, M_p, M_1  (paper Eq. 6)
2. _solve_equilibrium             Invert M_x (unique) or pseudoinverse (infinite)
3. _project_crs_to_p_space        Substitute x*(p) into each agent's CR → D p ≤ e
4. _cr_nonempty                   Chebyshev-center LP: is {p : D p ≤ e} non-empty?
5. build_gne_solution             Outer loop over all combinations → GNESolution

Notation (paper Eq. 6 → code)
──────────────────────────────
M_x x* = M_p p + M_1
  M_x ∈ R^{n_x × n_x}   : LHS matrix  (I on diagonal, -A_i_x blocks off-diagonal)
  M_p ∈ R^{n_x × n_p}   : p-coefficient from each agent's affine law
  M_1 ∈ R^{n_x}         : constant term from each agent's affine law

Unique case  rank(M_x) = n_x:
    x*(p) = H_x p + h_x   where H_x = M_x^{-1} M_p,  h_x = M_x^{-1} M_1

Infinite case  rank(M_x) < n_x  (min-norm selection):
    x*(p) = M_x^+ (M_p p + M_1)        (pseudoinverse, Eq. 9a in paper)
    solvability: U_2^T M_p = 0 and U_2^T M_1 = 0  (Eq. 10)
"""

from __future__ import annotations
import itertools
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.linalg import svd

from .game import GNEGame
from .cr_store import AgentSolution, GNECriticalRegion, GNESolution


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 — build equilibrium linear system  (paper Eq. 6)
# ─────────────────────────────────────────────────────────────────────────────

def _assemble_equilibrium_system(
    combo: tuple[int, ...],
    agent_solutions: list[AgentSolution],
    game: GNEGame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble  M_x x* = M_p p + M_1  from the combination C_k = (j_1,...,j_N).

    For each agent i with CR j_i, the affine best-response is:
        x_i* = A_i θ_i + b_i
             = A_i_x x_{-i}* + A_i_p p + b_i

    where θ_i = [x_{-i}; p], so:
        A_i_x = A_i[:, :n_x_neg_i]   (n_x_i, n_x_neg_i)
        A_i_p = A_i[:, n_x_neg_i:]   (n_x_i, n_p)

    Rearranged for all agents simultaneously:
        I x_i* - sum_{j≠i} A_i_x_j x_j* = A_i_p p + b_i

    In block matrix form → M_x x* = M_p p + M_1.

    Parameters
    ----------
    combo            : (j_0, j_1, ..., j_{N-1}) — one CR index per agent
    agent_solutions  : list of AgentSolution, one per agent
    game             : GNEGame

    Returns
    -------
    Mx : (n_x_total, n_x_total)
    Mp : (n_x_total, n_p)
    M1 : (n_x_total,)
    """
    N         = game.N
    n_x_total = game.n_x_total
    n_p       = game.n_p

    Mx = np.zeros((n_x_total, n_x_total))
    Mp = np.zeros((n_x_total, n_p))
    M1 = np.zeros(n_x_total)

    for i, j_i in enumerate(combo):
        cr      = agent_solutions[i][j_i]
        n_x_i   = game.agents[i].n_x
        n_x_neg = n_x_total - n_x_i

        # Split affine law: A_i = [A_i_x | A_i_p]
        A_i_x = cr.A[:, :n_x_neg]   # (n_x_i, n_x_neg)
        A_i_p = cr.A[:, n_x_neg:]   # (n_x_i, n_p)

        row_s = game.x_slice(i).start
        row_e = game.x_slice(i).stop

        # Diagonal block: identity (x_i* on LHS)
        Mx[row_s:row_e, row_s:row_e] = np.eye(n_x_i)

        # Off-diagonal blocks: -A_i_x_j for each j ≠ i
        # x_{-i} ordering in θ_i: agents in index order, skipping i
        col = 0
        for j in range(N):
            if j == i:
                continue
            n_x_j = game.agents[j].n_x
            col_s  = game.x_slice(j).start
            col_e  = game.x_slice(j).stop
            Mx[row_s:row_e, col_s:col_e] = -A_i_x[:, col:col + n_x_j]
            col += n_x_j

        # RHS blocks
        Mp[row_s:row_e, :] = A_i_p
        M1[row_s:row_e]    = cr.b

    return Mx, Mp, M1


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2 — solve M_x x* = M_p p + M_1
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _EquilibriumResult:
    H_x:       np.ndarray   # (n_x, n_p) — x*(p) = H_x p + h_x
    h_x:       np.ndarray   # (n_x,)
    is_unique: bool
    solvable:  bool          # False → no GNE exists for any p (rank drop + U_2 condition)


def _solve_equilibrium(
    Mx: np.ndarray,
    Mp: np.ndarray,
    M1: np.ndarray,
    tol_rank: float = 1e-8,
) -> _EquilibriumResult:
    """
    Solve M_x x* = M_p p + M_1 for x*(p) = H_x p + h_x.

    Unique case (rank(M_x) == n_x):
        H_x = M_x^{-1} M_p,  h_x = M_x^{-1} M_1   (paper Eq. 7a)

    Infinite / min-norm case (rank(M_x) < n_x):
        H_x = M_x^+ M_p,  h_x = M_x^+ M_1          (paper Eq. 9a)
        Solvability check: U_2^T M_p ≈ 0 and U_2^T M_1 ≈ 0  (paper Eq. 10)
        If not solvable → no GNE for this combination.
    """
    n_x = Mx.shape[0]
    n_p = Mp.shape[1]

    rank = np.linalg.matrix_rank(Mx, tol=tol_rank)

    if rank == n_x:
        # Full rank → unique GNE
        try:
            Mx_inv = np.linalg.inv(Mx)
        except np.linalg.LinAlgError:
            return _EquilibriumResult(
                H_x=np.zeros((n_x, n_p)), h_x=np.zeros(n_x),
                is_unique=False, solvable=False,
            )
        return _EquilibriumResult(
            H_x=Mx_inv @ Mp,
            h_x=Mx_inv @ M1,
            is_unique=True,
            solvable=True,
        )

    # Rank-deficient → SVD for min-norm / infinite solution (paper Eq. 8-9)
    U, sigma, Vt = svd(Mx, full_matrices=True)
    n_M = rank                           # number of non-zero singular values
    sigma1 = sigma[:n_M]
    U1 = U[:, :n_M]
    U2 = U[:, n_M:]                      # null-space of M_x^T

    # Solvability: U_2^T (M_p p + M_1) = 0 must hold ∀ p  (paper Eq. 10)
    # i.e. U_2^T M_p = 0 and U_2^T M_1 = 0
    if (np.linalg.norm(U2.T @ Mp) > tol_rank * 10
            or np.linalg.norm(U2.T @ M1) > tol_rank * 10):
        return _EquilibriumResult(
            H_x=np.zeros((n_x, n_p)), h_x=np.zeros(n_x),
            is_unique=False, solvable=False,
        )

    # Min-norm solution: x*(p) = V_1 diag(σ)^{-1} U_1^T (M_p p + M_1)  (Eq. 9a)
    V1 = Vt[:n_M, :].T                  # (n_x, n_M)
    Sigma1_inv = np.diag(1.0 / sigma1)
    Mx_pinv = V1 @ Sigma1_inv @ U1.T   # (n_x, n_x)

    return _EquilibriumResult(
        H_x=Mx_pinv @ Mp,
        h_x=Mx_pinv @ M1,
        is_unique=False,
        solvable=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 3 — project agent CRs to p-space  (paper Eq. 7b)
# ─────────────────────────────────────────────────────────────────────────────

def _project_crs_to_p_space(
    combo: tuple[int, ...],
    agent_solutions: list[AgentSolution],
    game: GNEGame,
    H_x: np.ndarray,
    h_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Substitute x*(p) = H_x p + h_x into each agent's CR constraint,
    yielding a polyhedron purely in p-space.

    For agent i with CR j_i:
        E_i θ_i ≤ f_i   with θ_i = [x_{-i}*(p); p]
        = [E_i_x | E_i_p] [x_{-i}*(p); p] ≤ f_i
        = E_i_x (H_x_neg p + h_x_neg) + E_i_p p ≤ f_i
        = (E_i_x H_x_neg + E_i_p) p ≤ f_i - E_i_x h_x_neg
        =: D_i p ≤ e_i

    Returns
    -------
    D : (total_ineq, n_p)  — stacked D_i for all agents
    e : (total_ineq,)      — stacked e_i for all agents
    """
    D_blocks, e_blocks = [], []

    for i, j_i in enumerate(combo):
        cr      = agent_solutions[i][j_i]
        n_x_i   = game.agents[i].n_x
        n_x_neg = game.n_x_total - n_x_i

        E_i_x = cr.E[:, :n_x_neg]   # (n_ineq, n_x_neg)
        E_i_p = cr.E[:, n_x_neg:]   # (n_ineq, n_p)

        # Extract rows of H_x and h_x corresponding to x_{-i}
        # (x_{-i} ordering: all agents except i, in index order)
        others   = [j for j in range(game.N) if j != i]
        H_x_neg  = np.vstack([H_x[game.x_slice(j)] for j in others])    # (n_x_neg, n_p)
        h_x_neg  = np.concatenate([h_x[game.x_slice(j)] for j in others])# (n_x_neg,)

        D_i = E_i_x @ H_x_neg + E_i_p   # (n_ineq, n_p)
        e_i = cr.f - E_i_x @ h_x_neg    # (n_ineq,)

        D_blocks.append(D_i)
        e_blocks.append(e_i)

    return np.vstack(D_blocks), np.concatenate(e_blocks)


# ─────────────────────────────────────────────────────────────────────────────
#  Stage 4 — Chebyshev-center LP: is the p-space CR non-empty?
# ─────────────────────────────────────────────────────────────────────────────

def _cr_nonempty(
    D: np.ndarray,
    e: np.ndarray,
    tol: float = 1e-6,
) -> bool:
    """
    Check if { p : D p ≤ e } is non-empty using the Chebyshev-center LP.

    Variables: z = [p (n_p); r (1)]
    Maximize r  s.t.  D_i p + r ||D_i||_2 ≤ e_i

    Returns True iff the optimal r* > -tol  (polyhedron has interior).

    This is stronger than mere feasibility: r* > 0 means the polyhedron
    contains a ball of positive radius (full-dimensional interior).
    We use r* > -tol to also accept boundary-only intersections.
    """
    n_p  = D.shape[1]
    n_c  = D.shape[0]
    nrms = np.linalg.norm(D, axis=1, keepdims=True)   # (n_c, 1)

    # Build augmented constraint: [D | nrms] [p; r] ≤ e
    A_aug = np.hstack([D, nrms])                       # (n_c, n_p+1)
    b_aug = e

    # Objective: min -r  (maximise r)
    c_obj = np.zeros(n_p + 1)
    c_obj[-1] = -1.0

    res = linprog(
        c_obj, A_ub=A_aug, b_ub=b_aug,
        bounds=[(None, None)] * n_p + [(None, None)],
        method='highs',
        options={'disp': False},
    )

    if res.status == 0:
        return float(res.x[-1]) > -tol
    if res.status == 3:
        return True   # unbounded → feasible (no bound on r, D is degenerate)
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Public API — build the full GNE solution
# ─────────────────────────────────────────────────────────────────────────────

def build_gne_solution(
    game: GNEGame,
    agent_solutions: list[AgentSolution],
    tol_rank: float = 1e-8,
    tol_nonempty: float = 1e-6,
    verbose: bool = True,
) -> GNESolution:
    """
    Algorithm 1 (Steps 7-13): enumerate all CR combinations and collect
    valid GNE critical regions in p-space.

    For each combination C_k = (j_1, ..., j_N):
      1. Assemble M_x, M_p, M_1  (Eq. 6)
      2. Solve for x*(p) = H_x p + h_x  (Eq. 7a or min-norm via SVD)
      3. Project CRs to p-space: D p ≤ e  (Eq. 7b)
      4. Add parameter box: p_lb ≤ p ≤ p_ub
      5. Check non-empty via Chebyshev LP → store GNECriticalRegion

    Parameters
    ----------
    game             : GNEGame
    agent_solutions  : list[AgentSolution] — one per agent (from mp_solver)
    tol_rank         : threshold for rank detection in M_x
    tol_nonempty     : Chebyshev radius threshold for non-empty CR check
    verbose          : print progress

    Returns
    -------
    GNESolution with all valid GNECriticalRegion objects
    """
    N         = game.N
    n_p       = game.n_p

    # Build iteration order over all combinations
    cr_index_lists = [list(range(agent_solutions[i].n_cr)) for i in range(N)]
    n_total = 1
    for idx in cr_index_lists:
        n_total *= len(idx)

    if verbose:
        print(f"\n[gne_combiner] N={N} agents, n_p={n_p}")
        cr_counts = [agent_solutions[i].n_cr for i in range(N)]
        print(f"  CRs per agent: {cr_counts}  →  {n_total} combinations to check")

    # Parameter space box rows (added to every CR in p-space)
    D_box = np.vstack([ np.eye(n_p), -np.eye(n_p)])   # (2*n_p, n_p)
    e_box = np.concatenate([game.p_ub, -game.p_lb])   # (2*n_p,)

    gne_regions: list[GNECriticalRegion] = []
    n_singular  = 0   # M_x rank-deficient combos
    n_insolvable = 0  # infinite case but U_2^T condition fails
    n_empty     = 0   # non-empty check failed

    t0 = time.perf_counter()

    for combo in itertools.product(*cr_index_lists):

        # ── Stage 1: equilibrium system ───────────────────────────────────────
        Mx, Mp, M1 = _assemble_equilibrium_system(combo, agent_solutions, game)

        # ── Stage 2: solve for x*(p) ──────────────────────────────────────────
        eq = _solve_equilibrium(Mx, Mp, M1, tol_rank=tol_rank)

        if not eq.solvable:
            n_insolvable += 1
            continue

        if not eq.is_unique:
            n_singular += 1

        # ── Stage 3: project CRs to p-space ──────────────────────────────────
        D_crs, e_crs = _project_crs_to_p_space(
            combo, agent_solutions, game, eq.H_x, eq.h_x
        )

        # Merge with parameter box
        D_full = np.vstack([D_crs, D_box])
        e_full = np.concatenate([e_crs, e_box])

        # ── Stage 4: non-empty check ──────────────────────────────────────────
        if not _cr_nonempty(D_full, e_full, tol=tol_nonempty):
            n_empty += 1
            continue

        # ── Store valid GNECriticalRegion ─────────────────────────────────────
        gne_regions.append(GNECriticalRegion(
            combination=tuple(combo),
            D=D_full,
            e=e_full,
            H_x=eq.H_x,
            h_x=eq.h_x,
            Mx=Mx,
            Mp=Mp,
            M1=M1,
            is_unique=eq.is_unique,
        ))

    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"  Combinations: {n_total} total  |  "
              f"{len(gne_regions)} valid GNE CRs  |  "
              f"{n_insolvable} insolvable  |  "
              f"{n_singular} rank-deficient  |  "
              f"{n_empty} empty")
        print(f"  Elapsed: {elapsed:.2f}s")

    sol = GNESolution(regions=gne_regions, n_p=n_p, N=N)
    if verbose:
        print(f"  {sol.summary()}")
    return sol


# ─────────────────────────────────────────────────────────────────────────────
#  Diagnostic: verify a GNE solution at a specific p
# ─────────────────────────────────────────────────────────────────────────────

def verify_gne_at_p(
    p: np.ndarray,
    gne_sol: GNESolution,
    game: GNEGame,
    agent_solutions: list[AgentSolution],
    tol: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    For a given p, find the matching GNE CR and verify x*(p) is a valid GNE.

    Checks:
    1. p is inside a GNE CR
    2. Equilibrium residual ||M_x x* - M_p p - M_1|| is small
    3. x*(p) satisfies all game constraints
    4. Each agent's x_i* is inside its own AgentCR (in θ_i = [x_{-i}*; p] space)

    Returns dict with keys: 'found', 'residual', 'feasible', 'cr_valid', 'k'
    """
    p = np.asarray(p).ravel()
    k = gne_sol.locate(p)

    if k is None:
        if verbose:
            print(f"[verify] p not in any GNE CR")
        return {'found': False}

    cr_k = gne_sol[k]
    x_star = cr_k.evaluate(p)

    # 1. Equilibrium residual
    residual = cr_k.residual(p)

    # 2. Global feasibility
    feasible = game.all_feasible(x_star, p, tol=tol)

    # 3. Each agent's x_i* inside its CR (in θ_i space)
    cr_combo = cr_k.combination
    cr_valid = True
    for i, j_i in enumerate(cr_combo):
        n_x_neg = game.n_x_total - game.agents[i].n_x
        others = [j for j in range(game.N) if j != i]
        x_neg = np.concatenate([x_star[game.x_slice(j)] for j in others])
        theta_i = np.concatenate([x_neg, p])
        cr = agent_solutions[i][j_i]
        if not cr.contains(theta_i, tol=tol):
            cr_valid = False
            break

    if verbose:
        print(f"[verify] p={p}  →  GNE CR k={k}  combo={cr_combo}")
        print(f"  x*(p) = {x_star}")
        print(f"  equilibrium residual = {residual:.2e}")
        print(f"  feasible = {feasible}  |  CR valid = {cr_valid}")

    return {
        'found': True,
        'k': k,
        'x_star': x_star,
        'residual': residual,
        'feasible': feasible,
        'cr_valid': cr_valid,
    }
