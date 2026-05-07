"""
game.py — Agent and GNEGame data structures.

Mathematical problem (Hall & Bemporad 2025, Eq. 1):

  Agent i solves:
      min_{x_i}  1/2 x_i^T Q_i x_i  +  (c_i + F_i p)^T x_i
      s.t.  A_loc_i x_i  <=  b_loc_i + S_loc_i p     (local)
            sum_j C_j x_j  <=  d + S_coup p           (shared coupling)

  p in R^{n_p} is a shared external parameter vector (prices, references,
  initial conditions, capacity limits, ...).

  x_{-i} = other agents' decisions — treated as parameters in agent i's mpQP.

GNEGame collects all agents and the shared coupling constraint.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Agent:
    """
    One agent in a Generalized Nash Equilibrium Problem.

    Decision variable:  x_i in R^{n_x}

    Cost:
        J_i(x_i, p) = 1/2 x_i^T Q_i x_i  +  (c_i + F_i p)^T x_i

    Local constraints (parametric in p):
        A_loc @ x_i  <=  b_loc + S_loc @ p

    Coupling matrix block:
        C_i  (n_coupling x n_x)  — agent i's contribution to the shared constraint
        Full coupling:  sum_j C_j x_j <= d + S_coup @ p

    Attributes
    ----------
    index : int
        Zero-based agent index.
    n_x : int
        Dimension of agent i's decision variable x_i.
    Q : ndarray (n_x, n_x)
        Local cost Hessian — must be positive definite.
    c : ndarray (n_x,)
        Constant linear cost term.
    F : ndarray (n_x, n_p)
        Parametric linear cost term from external parameter p.
        For game-theoretic MPC: F = Γ_i^T Q_i_full Φ_x (cost from x_0).
    F_cross : ndarray (n_x, n_x_neg) or None
        Optional cost coupling from other agents' decisions x_{-i}.
        For game-theoretic MPC: F_cross = Γ_i^T Q_i_full [Γ_{j1}|Γ_{j2}|...].
        None (default) → no cost coupling through x_{-i} (generic GNE games).
    C : ndarray (n_coupling, n_x)
        Agent i's block of the global coupling constraint matrix.
    A_loc : ndarray (n_loc, n_x)
        Local constraint LHS.
    b_loc : ndarray (n_loc,)
        Local constraint RHS constant part.
    S_loc : ndarray (n_loc, n_p)
        Local constraint RHS parametric part.
    """

    index: int
    n_x: int
    Q: np.ndarray
    c: np.ndarray
    F: np.ndarray
    A_loc: np.ndarray
    b_loc: np.ndarray
    S_loc: np.ndarray
    C: np.ndarray | None = field(default=None)
    F_cross: np.ndarray | None = field(default=None)
    Gamma_self: np.ndarray | None = field(default=None)
    M_theta: np.ndarray | None = field(default=None)
    x_lb_rep: np.ndarray | None = field(default=None)
    x_ub_rep: np.ndarray | None = field(default=None)

    @property
    def n_p(self) -> int:
        return self.F.shape[1]

    @property
    def n_coupling(self) -> int:
        return self.C.shape[0] if self.C is not None else 0

    @property
    def n_loc(self) -> int:
        return self.A_loc.shape[0]

    @property
    def has_state_constraints(self) -> bool:
        return self.Gamma_self is not None

    def local_cost(self, x_i: np.ndarray, p: np.ndarray) -> float:
        """Evaluate J_i for given x_i and p."""
        lin = self.c + self.F @ p
        return float(0.5 * x_i @ self.Q @ x_i + lin @ x_i)

    def local_feasible(self, x_i: np.ndarray, p: np.ndarray, tol: float = 1e-8) -> bool:
        """True if x_i satisfies local constraints for the given p."""
        return bool(np.all(self.A_loc @ x_i <= self.b_loc + self.S_loc @ p + tol))


@dataclass
class GNEGame:
    """
    A parametric Generalized Nash Equilibrium Problem with N agents.

    Shared coupling constraint (across all agents):
        sum_i  C_i x_i  <=  d + S_coup @ p

    Parameter space (box):
        p_lb <= p <= p_ub

    Attributes
    ----------
    agents : list[Agent]
        Length N, indexed 0 to N-1.
    d : ndarray (n_coupling,)
        Coupling RHS constant part.
    S_coup : ndarray (n_coupling, n_p)
        Coupling RHS parametric part.
    p_lb : ndarray (n_p,)
        Lower bound on the parameter vector p.
    p_ub : ndarray (n_p,)
        Upper bound on the parameter vector p.
    """

    agents: list[Agent]
    d: np.ndarray | None = field(default=None)
    S_coup: np.ndarray | None = field(default=None)
    p_lb: np.ndarray = field(default_factory=lambda: np.array([]))
    p_ub: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def N(self) -> int:
        """Number of agents."""
        return len(self.agents)

    @property
    def n_p(self) -> int:
        """Dimension of parameter vector p."""
        return self.p_lb.shape[0]

    @property
    def n_coupling(self) -> int:
        """Number of shared coupling constraints (0 if no coupling)."""
        return self.d.shape[0] if self.d is not None else 0

    @property
    def n_x_total(self) -> int:
        """Total decision dimension across all agents."""
        return sum(a.n_x for a in self.agents)

    def x_slice(self, i: int) -> slice:
        """Index slice for agent i's decisions in the stacked vector x = [x_0;...;x_{N-1}]."""
        start = sum(self.agents[j].n_x for j in range(i))
        return slice(start, start + self.agents[i].n_x)

    def coupling_lhs(self, x: np.ndarray) -> np.ndarray | None:
        """Evaluate sum_i C_i x_i for stacked x = [x_0; ...; x_{N-1}]. Returns None if no coupling."""
        if self.d is None:
            return None
        result = np.zeros(self.n_coupling)
        for a in self.agents:
            if a.C is not None:
                result += a.C @ x[self.x_slice(a.index)]
        return result

    def coupling_feasible(self, x: np.ndarray, p: np.ndarray, tol: float = 1e-8) -> bool:
        """True if stacked x satisfies the shared coupling constraint for parameter p."""
        if self.d is None:
            return True
        lhs = self.coupling_lhs(x)
        if lhs is None:
            return True
        return bool(np.all(lhs <= self.d + self.S_coup @ p + tol))

    def all_feasible(self, x: np.ndarray, p: np.ndarray, tol: float = 1e-8) -> bool:
        """True if all local and coupling constraints are satisfied."""
        for a in self.agents:
            if not a.local_feasible(x[self.x_slice(a.index)], p, tol):
                return False
        return self.coupling_feasible(x, p, tol)

    def total_cost(self, x: np.ndarray, p: np.ndarray) -> float:
        """Sum of all agents' individual costs (social welfare metric, not optimized)."""
        return sum(
            a.local_cost(x[self.x_slice(a.index)], p)
            for a in self.agents
        )


# ---------------------------------------------------------------------------
# Random game generator
# ---------------------------------------------------------------------------

def make_random_game(
    N: int,
    n_x: int = 1,
    n_p: int = 1,
    n_coupling: int = 1,
    x_bound: float = 10.0,
    p_bound: float = 10.0,
    coupling_scale: float = 1.0,
    seed: int | None = None,
) -> GNEGame:
    """
    Generate a random GNE game with N agents.

    Each agent has:
      - n_x decision variables
      - PD cost Hessian Q_i = M^T M + eps*I  (guaranteed PD)
      - Zero constant linear cost (c_i = 0)
      - Zero parametric cost (F_i = 0)  — override after call for specific studies
      - Box local constraints: -x_bound <= x_i <= x_bound
      - No parametric local constraints (S_loc = 0)

    Shared coupling (scalar or vector):
      sum_i C_i x_i <= d + S_coup p
      where C_i = coupling_scale * ones(n_coupling, n_x) / N  (equal weights)
      and d = coupling_scale * x_bound * N / 2  (feasible for all x in box)
      S_coup = 0  (non-parametric coupling RHS by default)

    Parameters
    ----------
    N : int
        Number of agents.
    n_x : int
        Decision dimension per agent.
    n_p : int
        Parameter dimension.
    n_coupling : int
        Number of shared coupling constraints.
    x_bound : float
        Box bound on each decision variable (symmetric: [-x_bound, x_bound]).
    p_bound : float
        Box bound on each parameter (symmetric: [-p_bound, p_bound]).
    coupling_scale : float
        Scale of the coupling constraint matrices C_i.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    GNEGame
    """
    rng = np.random.default_rng(seed)

    agents = []
    for i in range(N):
        # PD Hessian: Q = M^T M + 0.1*I
        M_rand = rng.standard_normal((n_x, n_x))
        Q_i = M_rand.T @ M_rand + 0.1 * np.eye(n_x)

        c_i = np.zeros(n_x)
        F_i = np.zeros((n_x, n_p))

        # Coupling block: equal-weight sum (None if n_coupling == 0)
        C_i = (coupling_scale / N) * np.ones((n_coupling, n_x)) if n_coupling > 0 else None

        # Local box constraints: -x_bound <= x_i <= x_bound
        # Written as [I; -I] x_i <= [x_bound * 1; x_bound * 1]
        A_loc = np.vstack([np.eye(n_x), -np.eye(n_x)])         # (2*n_x, n_x)
        b_loc = x_bound * np.ones(2 * n_x)
        S_loc = np.zeros((2 * n_x, n_p))

        agents.append(Agent(
            index=i,
            n_x=n_x,
            Q=Q_i,
            c=c_i,
            F=F_i,
            C=C_i,
            A_loc=A_loc,
            b_loc=b_loc,
            S_loc=S_loc,
        ))

    # Shared coupling RHS: d = coupling_scale * x_bound (feasible for any x in box)
    d = coupling_scale * x_bound * np.ones(n_coupling) if n_coupling > 0 else None
    S_coup = np.zeros((n_coupling, n_p)) if n_coupling > 0 else None

    # Parameter space box
    p_lb = -p_bound * np.ones(n_p)
    p_ub =  p_bound * np.ones(n_p)

    return GNEGame(
        agents=agents,
        d=d,
        S_coup=S_coup,
        p_lb=p_lb,
        p_ub=p_ub,
    )
