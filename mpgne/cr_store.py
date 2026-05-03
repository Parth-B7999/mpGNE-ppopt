"""
cr_store.py — Critical region storage for explicit mpGNE.

Two-level CR hierarchy (Hall & Bemporad 2025):

  Level 1 — per-agent (in θ_i = (x_{-i}, p) space)
  ────────────────────────────────────────────────────
  AgentCR           one CR for agent i:  { θ_i : E_i θ_i ≤ f_i }
                    best response:        x_i* = A_i θ_i + b_i
  AgentSolution     all CRs for agent i  (mirrors dimpc ControllerSolution)

  Level 2 — combined GNE (in p-space only, paper Eq. 7)
  ────────────────────────────────────────────────────────
  GNECriticalRegion one equilibrium CR:  { p : D_k p ≤ e_k }
                    GNE solution:         x*(p) = H_x p + h_x   (unique case)
                    combination:          (j_1, ..., j_N) — one CR index per agent
  GNESolution       all combined GNE CRs — the full PWA explicit GNE map

Notation map (paper → code)
───────────────────────────
Agent CR in θ_i-space     E_i θ_i ≤ f_i                      AgentCR.{E, f}
Best response affine law  x_i* = A_i θ_i + b_i               AgentCR.{A, b}
Equilibrium linear sys    M_x x* = M_p p + M_1      (Eq. 6)  GNECriticalRegion.{Mx, Mp, M1}
Unique GNE affine law     x* = H_x p + h_x          (Eq. 7a) GNECriticalRegion.{Hx, hx}
Equilibrium CR in p-space D_k p ≤ e_k               (Eq. 7b) GNECriticalRegion.{D, e}
"""

from __future__ import annotations
from dataclasses import dataclass, field
import pickle
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Level 1 — per-agent CR (mirrors dimpc CriticalRegion / ControllerSolution)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentCR:
    """
    One critical region for agent i's best-response mpQP.

    Region (in θ_i = (x_{-i}, p) space):  E θ_i ≤ f
    Best response:                          x_i* = A θ_i + b

    Attributes
    ----------
    E : ndarray (n_ineq, n_theta_i)   CR half-space normals
    f : ndarray (n_ineq,)             CR half-space RHS  (stored 1-D)
    A : ndarray (n_x_i, n_theta_i)   affine solution coefficient
    b : ndarray (n_x_i,)             affine solution offset
    index : int                       position in parent AgentSolution (0-based)
    facet_neighbors : list[int]       CR indices sharing a facet (filled offline)
    """

    E: np.ndarray
    f: np.ndarray
    A: np.ndarray
    b: np.ndarray
    index: int = 0
    facet_neighbors: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.E = np.atleast_2d(np.asarray(self.E, dtype=float))
        self.f = np.asarray(self.f, dtype=float).ravel()
        self.A = np.atleast_2d(np.asarray(self.A, dtype=float))
        self.b = np.asarray(self.b, dtype=float).ravel()

    def contains(self, theta: np.ndarray, tol: float = 1e-8) -> bool:
        """True if θ_i ∈ CR:  E θ_i ≤ f + tol."""
        return bool(np.all(self.E @ np.asarray(theta).ravel() <= self.f + tol))

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        """Return best response x_i* = A θ_i + b,  shape (n_x_i,)."""
        return self.A @ np.asarray(theta).ravel() + self.b

    @property
    def n_theta(self) -> int:
        return self.E.shape[1]

    @property
    def n_x(self) -> int:
        return self.A.shape[0]

    @property
    def n_ineq(self) -> int:
        return self.E.shape[0]


@dataclass
class AgentSolution:
    """
    All critical regions for agent i's best-response mpQP.

    Mirrors dimpc ControllerSolution.  Indexed as sol[v] → AgentCR v.

    Attributes
    ----------
    agent_index : int        zero-based agent index i
    n_x_i : int             decision dimension of agent i
    n_theta_i : int         parameter dimension  (= sum_{j≠i} n_x_j  +  n_p)
    regions : list[AgentCR]
    """

    agent_index: int
    n_x_i: int
    n_theta_i: int
    regions: list[AgentCR] = field(default_factory=list)

    def __getitem__(self, v: int) -> AgentCR:
        return self.regions[v]

    def __len__(self) -> int:
        return len(self.regions)

    @property
    def n_cr(self) -> int:
        return len(self.regions)

    def locate(self, theta: np.ndarray, tol: float = 1e-8) -> int | None:
        """Return index of first CR containing θ_i, or None if not found."""
        for cr in self.regions:
            if cr.contains(theta, tol=tol):
                return cr.index
        return None

    def evaluate(self, theta: np.ndarray, tol: float = 1e-8) -> np.ndarray | None:
        """Return best response x_i*(θ_i), or None if θ_i outside all CRs."""
        v = self.locate(theta, tol=tol)
        return None if v is None else self.regions[v].evaluate(theta)


# ─────────────────────────────────────────────────────────────────────────────
#  Level 2 — combined GNE CR (in p-space, paper Eq. 6-7)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GNECriticalRegion:
    """
    One equilibrium critical region in p-space (paper Eq. 7).

    For a combination C_k = (j_1, ..., j_N) of per-agent CRs, the stacked
    best-response affine laws give the linear equilibrium system (Eq. 6):

        M_x x* = M_p p + M_1

    Unique case (rank(M_x) = n_x_total):
        x*(p) = H_x p + h_x   where  H_x = M_x^{-1} M_p,  h_x = M_x^{-1} M_1
        CR in p-space:  D p ≤ e   (intersection of all agents' CR constraints
                                    projected to p, per Eq. 7b)

    Infinite case (rank(M_x) < n_x_total):
        Parametric family stored via SVD; min-norm selection gives a specific
        affine law.  H_x, h_x hold the min-norm solution coefficients.

    Attributes
    ----------
    combination : tuple[int, ...]   per-agent CR indices (j_1, ..., j_N)
    D : ndarray (n_ineq_p, n_p)    CR half-space normals in p-space
    e : ndarray (n_ineq_p,)        CR half-space RHS
    H_x : ndarray (n_x_total, n_p) affine solution p-coefficient
    h_x : ndarray (n_x_total,)     affine solution constant
    Mx : ndarray (n_x_total, n_x_total)  equilibrium system LHS
    Mp : ndarray (n_x_total, n_p)        equilibrium system p-coeff
    M1 : ndarray (n_x_total,)            equilibrium system constant
    is_unique : bool                      True if M_x is full rank
    """

    combination: tuple[int, ...]
    D: np.ndarray
    e: np.ndarray
    H_x: np.ndarray
    h_x: np.ndarray
    Mx: np.ndarray
    Mp: np.ndarray
    M1: np.ndarray
    is_unique: bool = True

    def __post_init__(self):
        self.D  = np.atleast_2d(np.asarray(self.D, dtype=float))
        self.e  = np.asarray(self.e, dtype=float).ravel()
        self.H_x = np.atleast_2d(np.asarray(self.H_x, dtype=float))
        self.h_x = np.asarray(self.h_x, dtype=float).ravel()
        self.Mx  = np.atleast_2d(np.asarray(self.Mx, dtype=float))
        self.Mp  = np.atleast_2d(np.asarray(self.Mp, dtype=float))
        self.M1  = np.asarray(self.M1, dtype=float).ravel()

    def contains(self, p: np.ndarray, tol: float = 1e-8) -> bool:
        """True if p ∈ CR_k:  D p ≤ e + tol."""
        return bool(np.all(self.D @ np.asarray(p).ravel() <= self.e + tol))

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        """Return stacked GNE x*(p) = H_x p + h_x,  shape (n_x_total,)."""
        return self.H_x @ np.asarray(p).ravel() + self.h_x

    def residual(self, p: np.ndarray) -> float:
        """Equilibrium residual ||M_x x*(p) - M_p p - M_1||  (should be ~0)."""
        x_star = self.evaluate(p)
        return float(np.linalg.norm(self.Mx @ x_star - self.Mp @ p - self.M1))

    @property
    def n_p(self) -> int:
        return self.D.shape[1]

    @property
    def n_x_total(self) -> int:
        return self.H_x.shape[0]

@dataclass
class GNESolution:
    """
    Full PWA explicit GNE solution — all equilibrium CRs in p-space.
    """
    regions: list[GNECriticalRegion] = field(default_factory=list)
    n_p: int = 0
    N: int = 0

    def __getitem__(self, k: int) -> GNECriticalRegion:
        return self.regions[k]

    def __len__(self) -> int:
        return len(self.regions)

    @property
    def n_cr(self) -> int:
        return len(self.regions)

    @property
    def n_unique(self) -> int:
        return sum(1 for r in self.regions if r.is_unique)

    @property
    def n_infinite(self) -> int:
        return sum(1 for r in self.regions if not r.is_unique)

    def locate(self, p: np.ndarray, tol: float = 1e-8) -> int | None:
        """Return index of first GNE CR containing p, or None."""
        for k, cr in enumerate(self.regions):
            if cr.contains(p, tol=tol):
                return k
        return None

    def locate_all(self, p: np.ndarray, tol: float = 1e-8) -> list[int]:
        """Return indices of ALL GNE CRs containing p (CRs may overlap)."""
        return [k for k, cr in enumerate(self.regions) if cr.contains(p, tol=tol)]

    def evaluate(self, p: np.ndarray, tol: float = 1e-8) -> np.ndarray | None:
        """
        Return stacked GNE x*(p) for the first matching CR, or None.
        """
        k = self.locate(p, tol=tol)
        return None if k is None else self.regions[k].evaluate(p)

    def summary(self) -> str:
        lines = [
            f"GNESolution: N={self.N} agents, n_p={self.n_p}",
            f"  total CRs: {self.n_cr}  "
            f"(unique: {self.n_unique}, infinite: {self.n_infinite})",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  PPOPT → AgentSolution converter
# ─────────────────────────────────────────────────────────────────────────────

def agent_solution_from_ppopt(
    ppopt_solution,
    agent_index: int,
    n_x_i: int,
    n_theta_i: int,
) -> AgentSolution:
    """
    Convert a PPOPT Solution object to an AgentSolution.

    PPOPT CriticalRegion fields used:
        cr.E  (n_ineq, n_theta_i)   CR half-space matrix
        cr.f  (n_ineq, 1)           CR half-space RHS
        cr.A  (n_x_i,  n_theta_i)  affine solution coefficient
        cr.b  (n_x_i,  1)          affine solution offset
    """
    regions = []
    for v, pcr in enumerate(ppopt_solution.critical_regions):
        regions.append(AgentCR(
            E=pcr.E,
            f=pcr.f.ravel(),
            A=pcr.A,
            b=pcr.b.ravel(),
            index=v,
        ))
    return AgentSolution(
        agent_index=agent_index,
        n_x_i=n_x_i,
        n_theta_i=n_theta_i,
        regions=regions,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_agent_solutions(solutions: list[AgentSolution], path: str) -> None:
    """Pickle list of AgentSolution to disk."""
    with open(path, "wb") as fh:
        pickle.dump(solutions, fh)
    total = sum(s.n_cr for s in solutions)
    print(f"[save] {path}  ({total} CRs across {len(solutions)} agents)")


def load_agent_solutions(path: str) -> list[AgentSolution]:
    """Load pickled list of AgentSolution from disk."""
    with open(path, "rb") as fh:
        sols = pickle.load(fh)
    total = sum(s.n_cr for s in sols)
    print(f"[load] {path}  ({total} CRs across {len(sols)} agents)")
    return sols


def save_gne_solution(gne_sol: GNESolution, path: str) -> None:
    """Pickle GNESolution to disk."""
    with open(path, "wb") as fh:
        pickle.dump(gne_sol, fh)
    print(f"[save] {path}  ({gne_sol.summary()})")


def load_gne_solution(path: str) -> GNESolution:
    """Load pickled GNESolution from disk."""
    with open(path, "rb") as fh:
        sol = pickle.load(fh)
    print(f"[load] {path}  ({sol.summary()})")
    return sol
