"""
gne_selector.py — GNE solution selection for infinite-equilibria CRs.

Paper: Hall & Bemporad (2025), Section II-C, Eqs. 9-17.

For CRs where rank(M_x) < n_x (infinitely many GNEs), the general solution is:

    x*(p, y_2) = V_1 diag(σ)^{-1} U_1^T (M_p p + M_1)  +  V_2 y_2    (Eq. 9a)
                 ┗━━━━━━━━━━━━━━━ x_min(p) ━━━━━━━━━━━━━━━━┛   ┗━━free━┛

where y_2 ∈ R^{n_x - rank(M_x)} is a free parameter and V_2 spans the null
space of M_x.

Three selection criteria choose a specific y_2:

  min_norm  (Eq. 12) : y_2 = 0  →  minimum ||x*||_2
                       closed form, NO additional solve, fastest
  welfare   (Eq. 14) : minimise f^SW = Σ_i J_i(x*)  →  quadratic in y_2
                       closed form QP with y_2 = -(V_2^T Q_SW V_2)^+ g_y
  v_gne     (Eq. 17) : equal coupling Lagrange multipliers across agents
                       linear system in y_2 from KKT stationarity conditions

For UNIQUE CRs (rank(M_x) = n_x), all three return the same x*(p).

Usage
─────
    from mpgne.gne_selector import select_gne, evaluate_all_types

    # Single selection
    x_star = select_gne(cr, game, p, gne_type="welfare")

    # All three at once (for comparison / paper tables)
    results = evaluate_all_types(cr, game, p)
    # results["min_norm"], results["welfare"], results["v_gne"]

Implementation note
───────────────────
The paper's full offline approach sub-partitions each infinite CR into polyhedral
sub-regions via mpQP (Eq. 12 / 17) in the (p, y_2)-space, yielding a nested PWA
map.  Here we compute the selection AT QUERY TIME given a specific p — correct
for any given p, and sufficient for closed-loop simulation.  The distinction only
matters if you need the full parametric map over the infinite CR's interior.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.linalg import svd

from .game import GNEGame
from .cr_store import GNECriticalRegion


# ─────────────────────────────────────────────────────────────────────────────
#  Null-space decomposition (cached per call — cheap for small n_x)
# ─────────────────────────────────────────────────────────────────────────────

def _svd_decompose(Mx: np.ndarray, tol: float = 1e-8):
    """
    SVD of M_x, splitting into range (U1, σ1, V1) and null space (V2).

    Returns
    -------
    V2     : (n_x, n_null)  null-space basis  (columns are null vectors)
    x_min  : callable  p → V_1 diag(σ_1)^{-1} U_1^T (M_p p + M_1) — min-norm map
    rank   : int
    U1, sigma1, V1, U2 : SVD components for further use
    """
    U, sigma, Vt = svd(Mx, full_matrices=True)
    rank  = int(np.sum(sigma > tol))

    U1     = U[:, :rank]
    sigma1 = sigma[:rank]
    V1     = Vt[:rank, :].T        # (n_x, rank)
    U2     = U[:, rank:]           # (n_x, n_null)  — left null space
    V2     = Vt[rank:, :].T       # (n_x, n_null)  — right null space of M_x

    # Min-norm solution map: x_min(p) = Mx^+ (Mp p + M1)
    Mx_pinv = V1 @ np.diag(1.0 / sigma1) @ U1.T   # (n_x, n_x)

    return V2, Mx_pinv, rank, U1, sigma1, V1, U2


def _x_min(cr: GNECriticalRegion, p: np.ndarray) -> np.ndarray:
    """Min-norm solution for any CR (unique or infinite)."""
    return cr.H_x @ p + cr.h_x   # already stored as pseudoinverse result


# ─────────────────────────────────────────────────────────────────────────────
#  Welfare GNE selection  (paper Eq. 14-15)
# ─────────────────────────────────────────────────────────────────────────────

def _build_welfare_matrices(game: GNEGame, p: np.ndarray):
    """
    Build block-diagonal social welfare cost  f^SW = 1/2 x*^T Q_SW x* + l_SW^T x*

    Q_SW = block_diag(Q_0, ..., Q_{N-1})    (n_x_total × n_x_total)
    l_SW = [c_0 + F_0 p; ...; c_{N-1} + F_{N-1} p]  (n_x_total,)

    Paper Eq. 15: f^SW(x*) = Σ_i 1/2 (x*)^T Q_i x* + (c_i + F_i p)^T x*
    """
    n_x = game.n_x_total
    Q_SW = np.zeros((n_x, n_x))
    l_SW = np.zeros(n_x)
    for a in game.agents:
        sl = game.x_slice(a.index)
        Q_SW[sl, sl] = a.Q
        l_SW[sl]     = a.c + a.F @ p
    return Q_SW, l_SW


def _welfare_y2(
    x_min_p: np.ndarray,
    V2: np.ndarray,
    Q_SW: np.ndarray,
    l_SW: np.ndarray,
) -> np.ndarray:
    """
    Minimise f^SW over y_2 (unconstrained, closed form).

    Substituting x* = x_min + V_2 y_2:
        f^SW(y_2) = const + 1/2 y_2^T (V_2^T Q_SW V_2) y_2
                           + (V_2^T Q_SW x_min + V_2^T l_SW)^T y_2

    Stationarity: (V_2^T Q_SW V_2) y_2* = -(V_2^T Q_SW x_min + V_2^T l_SW)
    """
    H_y = V2.T @ Q_SW @ V2                          # (n_null, n_null) — PSD
    g_y = V2.T @ Q_SW @ x_min_p + V2.T @ l_SW       # (n_null,)

    # Use pseudoinverse for robustness when H_y is singular
    y2, *_ = np.linalg.lstsq(H_y, -g_y, rcond=None)
    return y2


def select_welfare_gne(
    cr: GNECriticalRegion,
    game: GNEGame,
    p: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Return welfare GNE x*(p) for given CR and p.

    For unique CRs: returns the unique solution (same as min-norm).
    For infinite CRs: selects y_2 that minimises social welfare f^SW.
    """
    x_minn = _x_min(cr, p)
    if cr.is_unique:
        return x_minn

    V2, *_ = _svd_decompose(cr.Mx, tol=tol)
    if V2.shape[1] == 0:
        return x_minn

    Q_SW, l_SW = _build_welfare_matrices(game, p)
    y2 = _welfare_y2(x_minn, V2, Q_SW, l_SW)
    return x_minn + V2 @ y2


# ─────────────────────────────────────────────────────────────────────────────
#  v-GNE selection  (paper Eq. 16-17)
# ─────────────────────────────────────────────────────────────────────────────

def _coupling_multiplier_affine(
    game: GNEGame,
    i: int,
    x_i: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """
    Coupling Lagrange multiplier for agent i via KKT stationarity.

    KKT (coupling active):  Q_i x_i* + c_i + F_i p + C_i^T λ_i* = 0
    → λ_i* = -(C_i C_i^T)^+ C_i (Q_i x_i* + c_i + F_i p)

    Returns λ_i*  shape (n_coupling,).
    """
    a   = game.agents[i]
    grad = a.Q @ x_i + a.c + a.F @ p          # KKT stationarity gradient
    C    = a.C
    CCT  = C @ C.T                             # (n_coupling, n_coupling)
    # C (C^T C)^{-1}: pseudoinverse of C^T = (C C^T)^{-1} C
    if np.linalg.matrix_rank(CCT) == CCT.shape[0]:
        lam = -np.linalg.solve(CCT, C @ grad)
    else:
        lam, *_ = np.linalg.lstsq(C.T, -grad, rcond=None)
    return lam


def _v_gne_y2(
    x_min_p: np.ndarray,
    V2: np.ndarray,
    game: GNEGame,
    p: np.ndarray,
) -> np.ndarray:
    """
    Select y_2 satisfying the v-GNE condition: equal Lagrange multipliers.

    Condition (Eq. 16):  λ*_0(p,y_2) = λ*_i(p,y_2)  for i = 1,...,N-1

    Each λ*_i is affine in y_2:
        λ_i*(p, y_2) = -M_i Q_i (x_i_min + V2_i y_2) + M_i (c_i + F_i p)

    where M_i = (C_i C_i^T)^{-1} C_i  (or pseudoinverse).

    Setting λ*_0 = λ*_i gives a linear system in y_2 with (N-1)*n_coupling rows.
    Solved in least-squares sense (may be over/underdetermined).
    """
    n_null = V2.shape[1]

    def _M_i(i):
        C   = game.agents[i].C
        CCT = C @ C.T
        if np.linalg.matrix_rank(CCT) == CCT.shape[0]:
            return np.linalg.solve(CCT, C)   # (C C^T)^{-1} C
        M, *_ = np.linalg.lstsq(C.T, np.eye(C.shape[0]), rcond=None)
        return M.T

    def _V2_i(i):
        sl = game.x_slice(i)
        return V2[sl, :]                     # (n_x_i, n_null)

    def _x_i_min(i):
        return x_min_p[game.x_slice(i)]

    def _lam_const(i):
        """Constant part of λ_i*(p) (independent of y_2) at x_i = x_i_min."""
        a    = game.agents[i]
        xi   = _x_i_min(i)
        grad = a.Q @ xi + a.c + a.F @ p
        return -_M_i(i) @ grad              # (n_coupling,)

    def _lam_V2_coef(i):
        """Coefficient of y_2 in λ_i*(p, y_2)."""
        a = game.agents[i]
        return -_M_i(i) @ a.Q @ _V2_i(i)   # (n_coupling, n_null)

    # Equal-multiplier system: λ*_0(y_2) - λ*_i(y_2) = 0  for i=1..N-1
    rows, rhs = [], []
    lam0_const = _lam_const(0)
    A0 = _lam_V2_coef(0)

    for i in range(1, game.N):
        Ai = _lam_V2_coef(i)
        rows.append(A0 - Ai)                        # (n_coupling, n_null)
        rhs.append(_lam_const(i) - lam0_const)      # (n_coupling,)

    if not rows:
        return np.zeros(n_null)

    A_v = np.vstack(rows)     # ((N-1)*n_coupling, n_null)
    b_v = np.concatenate(rhs) # ((N-1)*n_coupling,)

    # Guard: if A_v is near-zero the system is degenerate (equal-multiplier
    # condition trivially satisfied or impossible to satisfy within this CR).
    # Revert to min-norm selection (y_2 = 0) to avoid numerical explosion.
    if np.linalg.norm(A_v) < 1e-8 * max(1.0, np.linalg.norm(b_v)):
        return np.zeros(n_null)

    y2, *_ = np.linalg.lstsq(A_v, b_v, rcond=None)

    # Sanity check: y_2 should be moderate.  If it exploded (ill-conditioned
    # system) fall back to y_2 = 0 so the equilibrium residual stays small.
    if np.linalg.norm(y2) > 1e6:
        return np.zeros(n_null)

    return y2


def select_v_gne(
    cr: GNECriticalRegion,
    game: GNEGame,
    p: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Return variational GNE x*(p) for given CR and p.

    For unique CRs: returns the unique solution.
    For infinite CRs: selects y_2 enforcing equal coupling Lagrange multipliers
                      across all agents (KKT stationarity approach, Eq. 16-17).
    """
    x_minn = _x_min(cr, p)
    if cr.is_unique:
        return x_minn

    V2, *_ = _svd_decompose(cr.Mx, tol=tol)
    if V2.shape[1] == 0:
        return x_minn

    y2 = _v_gne_y2(x_minn, V2, game, p)
    return x_minn + V2 @ y2


# ─────────────────────────────────────────────────────────────────────────────
#  Unified interface
# ─────────────────────────────────────────────────────────────────────────────

def select_gne(
    cr: GNECriticalRegion,
    game: GNEGame,
    p: np.ndarray,
    gne_type: str = "min_norm",
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Return x*(p) for the specified GNE selection criterion.

    Parameters
    ----------
    cr       : GNECriticalRegion  (must contain p, i.e. cr.contains(p))
    game     : GNEGame
    p        : parameter vector (n_p,)
    gne_type : "min_norm" | "welfare" | "v_gne"
    tol      : SVD rank threshold

    Returns
    -------
    x_star : (n_x_total,)  the selected GNE

    Notes
    -----
    For UNIQUE CRs (cr.is_unique=True) all three methods return the same result.
    For INFINITE CRs (cr.is_unique=False) each method selects a different point
    from the infinite set of equilibria consistent with that CR combination.
    """
    p = np.asarray(p).ravel()

    if gne_type == "min_norm":
        return _x_min(cr, p)
    elif gne_type == "welfare":
        return select_welfare_gne(cr, game, p, tol=tol)
    elif gne_type == "v_gne":
        return select_v_gne(cr, game, p, tol=tol)
    else:
        raise ValueError(f"Unknown gne_type: {gne_type!r}. "
                         f"Choose from 'min_norm', 'welfare', 'v_gne'.")


@dataclass
class AllGNETypes:
    """All three GNE selections at a specific p for one CR."""
    p:          np.ndarray
    is_unique:  bool
    min_norm:   np.ndarray
    welfare:    np.ndarray
    v_gne:      np.ndarray
    residuals:  dict          # equilibrium residual ||M_x x* - M_p p - M_1|| per type

    def print_summary(self):
        print(f"  p = {self.p}")
        print(f"  CR type: {'unique' if self.is_unique else 'infinite-many'}")
        print(f"  {'Type':<12} {'x*':>30}  {'equil. residual':>16}")
        print(f"  {'-'*60}")
        for gne_type, x in [("min_norm", self.min_norm),
                             ("welfare",  self.welfare),
                             ("v_gne",    self.v_gne)]:
            x_str = np.array2string(x, precision=4, suppress_small=True,
                                    max_line_width=30)
            print(f"  {gne_type:<12} {x_str:>30}  "
                  f"{self.residuals[gne_type]:>16.2e}")


def evaluate_all_types(
    cr: GNECriticalRegion,
    game: GNEGame,
    p: np.ndarray,
    tol: float = 1e-8,
) -> AllGNETypes:
    """
    Compute all three GNE selections at p and return them together.

    Useful for paper comparisons: shows that all three are valid GNEs (small
    residuals) but differ in the x* they select for infinite CRs.

    Parameters
    ----------
    cr   : GNECriticalRegion  (must contain p)
    game : GNEGame
    p    : parameter vector

    Returns
    -------
    AllGNETypes  with fields min_norm, welfare, v_gne, residuals
    """
    p = np.asarray(p).ravel()

    x_mn = select_gne(cr, game, p, "min_norm", tol)
    x_wf = select_gne(cr, game, p, "welfare",  tol)
    x_vg = select_gne(cr, game, p, "v_gne",    tol)

    def _res(x):
        return float(np.linalg.norm(cr.Mx @ x - cr.Mp @ p - cr.M1))

    return AllGNETypes(
        p=p,
        is_unique=cr.is_unique,
        min_norm=x_mn,
        welfare=x_wf,
        v_gne=x_vg,
        residuals={
            "min_norm": _res(x_mn),
            "welfare":  _res(x_wf),
            "v_gne":    _res(x_vg),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Scan all CRs in a GNESolution for a given p
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_crs(
    gne_sol,           # GNESolution
    game: GNEGame,
    p: np.ndarray,
    tol: float = 1e-8,
    verbose: bool = True,
) -> list[AllGNETypes]:
    """
    For all CRs that contain p, compute all three GNE selections.

    Returns a list (one AllGNETypes per matching CR).  In the non-overlapping
    unique case this has length 1; in the overlapping infinite case it may be
    longer (different equilibria from the same p).
    """
    p   = np.asarray(p).ravel()
    ks  = gne_sol.locate_all(p)
    out = []

    for k in ks:
        cr  = gne_sol[k]
        res = evaluate_all_types(cr, game, p, tol=tol)
        if verbose:
            print(f"\n  CR k={k}  combination={cr.combination}  "
                  f"{'unique' if cr.is_unique else 'infinite'}:")
            res.print_summary()
        out.append(res)

    if verbose and not ks:
        print(f"  p={p} is outside all GNE CRs")

    return out
