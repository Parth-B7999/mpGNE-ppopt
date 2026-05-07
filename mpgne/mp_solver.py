"""
mp_solver.py — Offline mpQP solver for each agent's best-response problem.

Paper: Hall & Bemporad (2025), Algorithm 1 Steps 3-6, Eq. (1)-(4).

For each agent i the mpQP is:

    min_{x_i}  1/2 x_i^T Q_i x_i  +  (c_i + F_i p)^T x_i
    s.t.  A_loc_i x_i  <=  b_loc_i + S_loc_i p          (input bounds)
          ±Γ_i x_i     <=  ±x_{ub/lb}_rep ∓ M_θ θ_i    (state bounds)
          C_i x_i      <=  d - C_{-i} x_{-i} + S_coup p  (coupling, if present)
          θ_i          ∈  P  (box on parameter space)

Parameter vector for agent i:
    θ_i = [ x_{-i} ; p ]   ∈  R^{n_theta_i}
    n_theta_i = (n_x_total - n_x_i) + n_p

PPOPT standard form:
    min_{x_i}  1/2 x_i^T Q x_i  +  (H @ θ_i)^T x_i  +  c^T x_i
    s.t.       G x_i  <=  b  +  F @ θ_i
               A_t θ_i  <=  b_t

Matrix derivations
──────────────────
Cost:
    Q    = Q_i                                   (n_x_i, n_x_i)
    H    = [F_cross_i               |  F_i]      (n_x_i, n_theta_i)
    c    = c_i                                   (n_x_i,)

Constraints  (stacked local + state + coupling):
    G    = [A_loc_i ]                            (n_total, n_x_i)
           [±Γ_i    ]
           [C_i     ]
    b    = [b_loc_i      ]                       (n_total,)
           [±x_{ub/lb}_rep]
           [d            ]
    F    = [0          |  S_loc_i ]              (n_total, n_theta_i)
           [∓M_theta            ]
           [-C_{-i}    |  S_coup  ]

Parameter space (box):
    A_t  = [+I; -I]                             (2*n_theta_i, n_theta_i)
    b_t  = [θ_max; -θ_min]                      (2*n_theta_i,)
    where θ_min/max come from other agents' decision bounds + game.p_lb/p_ub
"""

from __future__ import annotations
import numpy as np

from ppopt.mpqp_program import MPQP_Program
from ppopt.mp_solvers.solve_mpqp import solve_mpqp, mpqp_algorithm

from .game import Agent, GNEGame
from .cr_store import AgentSolution, agent_solution_from_ppopt


# ─────────────────────────────────────────────────────────────────────────────
#  Algorithm selection  (mirror of dimpc mp_solver convention)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ALGORITHM = mpqp_algorithm.combinatorial_parallel


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1 — cost matrices
# ─────────────────────────────────────────────────────────────────────────────

def build_cost_matrices(
    game: GNEGame,
    i: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build PPOPT cost matrices for agent i's mpQP.

    Returns
    -------
    Q_pp : (n_x_i, n_x_i)   Hessian  — agent i's local Q_i
    H_pp : (n_x_i, n_theta_i) parametric cost  — (H @ θ_i)^T x_i = (F_i p)^T x_i
    c_pp : (n_x_i,)          constant linear cost = c_i
    """
    ai = game.agents[i]
    n_x_i    = ai.n_x
    n_x_neg  = game.n_x_total - n_x_i   # dimension of x_{-i}
    n_p      = game.n_p
    n_theta  = n_x_neg + n_p

    Q_pp = ai.Q.copy()

    # H @ θ_i = F_i p  (x_{-i} part has zero weight in cost)
    H_pp = np.zeros((n_x_i, n_theta))
    # Cost from x_{-i} (other agents' decisions): populated for MPC games
    if ai.F_cross is not None:
        H_pp[:, :n_x_neg] = ai.F_cross
    # Cost from external parameter p (e.g. x_0 for MPC)
    H_pp[:, n_x_neg:] = ai.F

    c_pp = ai.c.copy()

    return Q_pp, H_pp, c_pp


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2 — constraint matrices
# ─────────────────────────────────────────────────────────────────────────────

def build_constraint_matrices(
    game: GNEGame,
    i: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build PPOPT constraint matrices for agent i's mpQP.

    Constraints: G x_i <= b + F @ θ_i

    Includes:
      - Local input bounds: A_loc x_i <= b_loc + S_loc p
      - State bounds: x_lb_rep <= X <= x_ub_rep
        → ±Gamma_i x_i <= ±x_{ub/lb}_rep ∓ M_theta θ_i
      - Coupling constraint (if present): C_i x_i <= d - C_{-i} x_{-i} + S_coup p

    Returns
    -------
    G_pp : (n_total, n_x_i)
    b_pp : (n_total,)
    F_pp : (n_total, n_theta_i)
    """
    ai      = game.agents[i]
    n_x_i   = ai.n_x
    n_x_neg = game.n_x_total - n_x_i
    n_p     = game.n_p
    n_theta = n_x_neg + n_p

    # ── local constraints ─────────────────────────────────────────────────────
    # A_loc x_i <= b_loc + S_loc p
    # In terms of θ_i = [x_{-i}; p]:  RHS parametric part = [0 | S_loc] θ_i
    G_loc = ai.A_loc                              # (n_loc, n_x_i)
    b_loc = ai.b_loc                              # (n_loc,)
    F_loc = np.zeros((ai.n_loc, n_theta))
    F_loc[:, n_x_neg:] = ai.S_loc                # p part

    # ── state constraints ─────────────────────────────────────────────────────
    # x_lb_rep <= X <= x_ub_rep  with X = Phi_x p + Gamma_i U_i + Σ_{j≠i} Γ_j U_j
    # Upper:  +Gamma_i U_i <= x_ub_rep - M_theta θ_i  →  F = -M_theta
    # Lower:  -Gamma_i U_i <= -x_lb_rep + M_theta θ_i  →  F = +M_theta
    if ai.has_state_constraints:
        Gamma_i   = ai.Gamma_self                 # (Np*nx, n_u_i)
        M_theta_i = ai.M_theta                    # (Np*nx, n_x_neg + n_p)
        x_lb_rep  = ai.x_lb_rep                   # (Np*nx,)
        x_ub_rep  = ai.x_ub_rep                   # (Np*nx,)

        G_state = np.vstack([Gamma_i, -Gamma_i])  # (2*Np*nx, n_u_i)
        b_state = np.concatenate([x_ub_rep, -x_lb_rep])
        F_state = np.vstack([-M_theta_i, M_theta_i])
    else:
        G_state = np.empty((0, n_x_i))
        b_state = np.empty(0)
        F_state = np.empty((0, n_theta))

    # ── coupling constraint (if present) ──────────────────────────────────────
    if game.n_coupling > 0 and ai.C is not None:
        G_coup = ai.C
        b_coup = game.d

        others = [game.agents[j] for j in range(game.N) if j != i]
        C_neg = np.hstack([a.C for a in others])  # (n_coupling, n_x_neg)

        F_coup = np.zeros((game.n_coupling, n_theta))
        F_coup[:, :n_x_neg] = -C_neg              # x_{-i} part: -C_{-i}
        F_coup[:, n_x_neg:] = game.S_coup         # p part: S_coup
    else:
        G_coup = np.empty((0, n_x_i))
        b_coup = np.empty(0)
        F_coup = np.empty((0, n_theta))

    # ── stack ─────────────────────────────────────────────────────────────────
    G_pp = np.vstack([G_loc, G_state, G_coup])
    b_pp = np.concatenate([b_loc, b_state, b_coup])
    F_pp = np.vstack([F_loc, F_state, F_coup])

    return G_pp, b_pp, F_pp


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3 — parameter space
# ─────────────────────────────────────────────────────────────────────────────

def build_parameter_space(
    game: GNEGame,
    i: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the box constraint on θ_i = [x_{-i}; p] for PPOPT.

    Bounds on x_j (j ≠ i) are extracted from agent j's local box constraints,
    assuming A_loc = [+I; -I] and b_loc = [x_ub; x_ub] (symmetric box).
    Falls back to ±1e3 with a warning when the structure does not match.

    Returns
    -------
    A_t : (2*n_theta_i, n_theta_i)  [+I; -I]
    b_t : (2*n_theta_i, 1)          [θ_max; -θ_min]  (column for PPOPT)
    """
    others = [game.agents[j] for j in range(game.N) if j != i]

    lb_parts, ub_parts = [], []
    for aj in others:
        xlb, xub = _extract_box_bounds(aj)
        lb_parts.append(xlb)
        ub_parts.append(xub)

    lb_parts.append(game.p_lb)
    ub_parts.append(game.p_ub)

    theta_min = np.concatenate(lb_parts)   # (n_theta_i,)
    theta_max = np.concatenate(ub_parts)

    n_theta = len(theta_min)
    A_t = np.vstack([ np.eye(n_theta), -np.eye(n_theta)])
    b_t = np.concatenate([theta_max, -theta_min]).reshape(-1, 1)

    return A_t, b_t


def _extract_box_bounds(agent: Agent) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (x_lb, x_ub) from an agent's local constraint A_loc x <= b_loc.

    Assumes A_loc = [+I_{n_x}; -I_{n_x}] (first n_x rows +I, next n_x rows -I),
    which is the structure produced by make_random_game.

    Fallback: returns ±1e3 for all dimensions if pattern not found.
    """
    n = agent.n_x
    A = agent.A_loc
    b = agent.b_loc
    if (A.shape[0] >= 2 * n
            and np.allclose(A[:n],     np.eye(n))
            and np.allclose(A[n:2*n], -np.eye(n))):
        return -b[n:2*n], b[:n]
    # fallback
    import warnings
    warnings.warn(
        f"Agent {agent.index}: A_loc does not match [+I; -I] box structure. "
        "Using ±1e3 bounds for parameter space.",
        stacklevel=3,
    )
    return -1e3 * np.ones(n), 1e3 * np.ones(n)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4 — solve one agent's mpQP
# ─────────────────────────────────────────────────────────────────────────────

def solve_agent_mp(
    game: GNEGame,
    i: int,
    algorithm: mpqp_algorithm = DEFAULT_ALGORITHM,
    verbose: bool = True,
) -> AgentSolution:
    """
    Solve agent i's best-response mpQP and return its AgentSolution.

    Parameters
    ----------
    game      : GNEGame
    i         : agent index (0-based)
    algorithm : PPOPT mpqp_algorithm enum
    verbose   : print progress

    Returns
    -------
    AgentSolution  with all critical regions for agent i
    """
    ai      = game.agents[i]
    n_x_i   = ai.n_x
    n_x_neg = game.n_x_total - n_x_i
    n_theta = n_x_neg + game.n_p

    Q_pp, H_pp, c_pp = build_cost_matrices(game, i)
    G_pp, b_pp, F_pp = build_constraint_matrices(game, i)
    A_t, b_t         = build_parameter_space(game, i)

    n_c = G_pp.shape[0]

    if verbose:
        lam_min = float(np.linalg.eigvalsh(Q_pp).min())
        print(f"\n[mp_solver] Agent {i}:")
        print(f"  n_x_i={n_x_i}, n_theta={n_theta}, n_constraints={n_c}")
        print(f"  λ_min(Q_i)={lam_min:.4f}  (must be > 0)")

    # PPOPT expects column vectors for b and c
    problem = MPQP_Program(
        G_pp,               # A  — constraint LHS
        b_pp.reshape(-1, 1),# b  — constraint RHS
        c_pp.reshape(-1, 1),# c  — constant linear cost
        H_pp,               # H  — parametric cost  (n_x_i × n_theta)
        Q_pp,               # Q  — Hessian           (n_x_i × n_x_i)
        A_t,                # A_t — parameter space
        b_t,                # b_t — parameter space RHS
        F_pp,               # F  — parametric constraint (n_c × n_theta)
    )

    solution = solve_mpqp(problem, algorithm=algorithm)

    if verbose:
        print(f"  → {len(solution.critical_regions)} critical regions")

    return agent_solution_from_ppopt(
        solution,
        agent_index=i,
        n_x_i=n_x_i,
        n_theta_i=n_theta,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Step 5 — solve all agents
# ─────────────────────────────────────────────────────────────────────────────

def solve_all_agents_mp(
    game: GNEGame,
    algorithm: mpqp_algorithm = DEFAULT_ALGORITHM,
    verbose: bool = True,
) -> list[AgentSolution]:
    """
    Solve mpQP for all N agents and return their AgentSolutions.

    This is the full offline precomputation (Algorithm 1, Steps 3-6).
    Each agent's mpQP is solved independently (can be parallelised externally).

    Returns
    -------
    list[AgentSolution]  length N, index i holds agent i's CRs
    """
    solutions = []
    for i in range(game.N):
        sol = solve_agent_mp(game, i, algorithm=algorithm, verbose=verbose)
        solutions.append(sol)

    if verbose:
        total = sum(s.n_cr for s in solutions)
        print(f"\n[mp_solver] Done — {game.N} agents, {total} total CRs")
        for s in solutions:
            print(f"  agent {s.agent_index}: {s.n_cr} CRs")

    return solutions
