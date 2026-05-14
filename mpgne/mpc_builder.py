"""
mpc_builder.py — Factory: Plant → GNEGame for game-theoretic MPC.

Bridges the dimpc Plant structure (with random stable dynamics) to our
GNEGame framework by expanding the MPC problem over the prediction horizon.

Key difference from cooperative dimpc
──────────────────────────────────────
dimpc:  agents share a weighted GLOBAL cost (rho_i * sum of all states/inputs)
GNE:    each agent minimises ONLY its LOCAL cost (own states/inputs)
        + state bound constraints couple agents through the dynamics

Problem formulation
───────────────────
Agent i solves (generalized Nash game-theoretic MPC):

    min_{U_i}  ½ Σ_{k=0}^{Np-1} [ x_k^T Q_i_diag x_k + u_{i,k}^T R_i u_{i,k} ]

    s.t.  x_{k+1} = A x_k + Σ_j B_j u_{j,k}      (coupled dynamics)
          u_lb_i ≤ u_{i,k} ≤ u_ub_i               (local input bounds)
          x_lb ≤ x_k ≤ x_ub   for each k=1..Np    (state bounds, couple agents)
          x_0 = p                                   (initial state is parameter)

where Q_i_diag = blkdiag(0,..., Q_i, ..., 0) — only agent i's state block.

State constraints x_lb ≤ X ≤ x_ub create the "generalized" Nash structure:
each agent's feasible set depends on other agents' decisions through the
predicted state trajectory X = Φ_x p + Σ_j Γ_j U_j.

Expanding over the horizon using prediction matrices (Φ_x, Γ_j):
    X = Φ_x p + Γ_i U_i + Σ_{j≠i} Γ_j U_j

Collecting U_i terms (treating p = x_0 and U_{-i} as parameters θ_i = [U_{-i}; p]):

    H_qp_i   = Γ_i^T Q_i_full Γ_i  +  R_bar_i          (n_u_i × n_u_i)
    F_i       = Γ_i^T Q_i_full Φ_x                       (n_u_i × nx)   ← cost from p
    F_cross_i = Γ_i^T Q_i_full [Γ_{j1} | Γ_{j2} | ...]  (n_u_i × n_u_neg) ← cost from U_{-i}

Constraints in θ_i = [U_{-i}; p] form:
    Input bounds:  [I; -I] U_i  ≤  [u_ub_rep; -u_lb_rep]   (no θ_i dependence)
    State bounds:  ±Γ_i U_i     ≤  ±(x_{ub/lb}_rep) ∓ M_θ θ_i  (θ_i dependence)

Parameter:  p = x_0 ∈ [x_lb_global, x_ub_global]
"""

from __future__ import annotations
import numpy as np
from scipy.linalg import solve_discrete_are, LinAlgError

from .plant import Plant
from .game  import Agent, GNEGame


# ─────────────────────────────────────────────────────────────────────────────
#  Prediction matrices  (identical to dimpc mp_solver)
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_matrices(
    A: np.ndarray,
    B_list: list[np.ndarray],
    Np: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Φ_x and Γ_j — lower block-Toeplitz prediction matrices."""
    nx = A.shape[0]
    A_pows = [np.eye(nx)]
    for _ in range(Np):
        A_pows.append(A_pows[-1] @ A)

    Phi_x = np.vstack([A_pows[l + 1] for l in range(Np)])   # (Np*nx, nx)

    Gamma_list = []
    for B_j in B_list:
        nu_j = B_j.shape[1]
        G = np.zeros((Np * nx, Np * nu_j))
        for l in range(Np):
            for q in range(l + 1):
                G[l*nx:(l+1)*nx, q*nu_j:(q+1)*nu_j] = A_pows[l - q] @ B_j
        Gamma_list.append(G)

    return Phi_x, Gamma_list


# ─────────────────────────────────────────────────────────────────────────────
#  Local Q_full for agent i  (only agent i's state block is non-zero)
# ─────────────────────────────────────────────────────────────────────────────

def _build_local_Q_full(
    plant: Plant,
    Q_i: np.ndarray,
    P_i: np.ndarray,
    i: int,
) -> np.ndarray:
    """
    Block-diagonal state cost matrix over the prediction horizon for agent i.

    Q_full = blkdiag(Q_stage, ..., Q_stage, Q_term)  ← Np blocks
    where Q_stage = blkdiag(0,..., Q_i, ..., 0)  (only agent i's nx_i × nx_i block)
    """
    nx_total = plant.nx
    Np = plant.Np
    sl = plant.state_slice(i)

    def _padded(W):
        M = np.zeros((nx_total, nx_total))
        M[sl, sl] = W
        return M

    Q_stage = _padded(Q_i)
    Q_term  = _padded(P_i)
    blocks  = [Q_stage] * (Np - 1) + [Q_term]

    result = np.zeros((Np * nx_total, Np * nx_total))
    for k, B in enumerate(blocks):
        result[k*nx_total:(k+1)*nx_total, k*nx_total:(k+1)*nx_total] = B
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Default local weights (Q, R, P per agent, no rho scaling)
# ─────────────────────────────────────────────────────────────────────────────

def default_local_weights(
    plant: Plant,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Q_i = I_{nx_i},  R_i = I_{nu_i},  P_i = DARE(A_i, B_{i,i}, Q_i, R_i).

    Unlike dimpc default_weights, there is no rho scaling — each agent
    uses its own full unit-weight cost.
    """
    Q_list, R_list, P_list = [], [], []
    for s in plant.subsystems:
        Q_i = np.eye(s.nx)
        R_i = np.eye(s.nu)
        try:
            P_i = solve_discrete_are(s.A, s.B[s.index], Q_i, R_i)
        except (LinAlgError, ValueError):
            P_i = Q_i
        Q_list.append(Q_i)
        R_list.append(R_i)
        P_list.append(P_i)
    return Q_list, R_list, P_list


# ─────────────────────────────────────────────────────────────────────────────
#  Main factory
# ─────────────────────────────────────────────────────────────────────────────

def make_gne_game_from_plant(
    plant: Plant,
    coupling_mode: str = "state_bounds",
    L_max: float = 5.0,
    Q_list: list[np.ndarray] | None = None,
    R_list: list[np.ndarray] | None = None,
    P_list: list[np.ndarray] | None = None,
) -> GNEGame:
    """
    Convert a dimpc Plant into a GNEGame for game-theoretic MPC.

    Two coupling formulations are supported via ``coupling_mode``:

    ``"state_bounds"`` (default — generalized Nash):
        State constraints  x_lb ≤ x_k ≤ x_ub  for k=1..Np couple agents
        through the predicted trajectory.  Each agent's feasible set depends
        on other agents' decisions via  X = Φ_x p + Σ_j Γ_j U_j.
        Encoded as Gamma_self / M_theta / x_lb_rep / x_ub_rep on each Agent.
        GNEGame has n_coupling = 0  (no global coupling constraint).

    ``"l_max"`` (aggregate-input coupling):
        Shared constraint  Σ_j u_{j,k} ≤ L_max  for k=0..Np-1.
        Encoded as C_i on each Agent and d / S_coup on GNEGame.
        GNEGame has n_coupling = Np.

    Parameters
    ----------
    plant         : Plant  (from mpgne.plant_gen or dimpc plant_gen)
    coupling_mode : ``"state_bounds"`` | ``"l_max"``
    L_max         : aggregate-input limit per step (only used when coupling_mode="l_max")
    Q_list        : local state cost matrices per agent  (default: identity)
    R_list        : local input cost matrices per agent  (default: identity)
    P_list        : terminal cost matrices per agent     (default: DARE solution)

    Returns
    -------
    GNEGame  with N = plant.M agents,  n_p = plant.nx
    """
    if coupling_mode not in ("state_bounds", "l_max"):
        raise ValueError(f"coupling_mode must be 'state_bounds' or 'l_max', got {coupling_mode!r}")
    M  = plant.M
    Np = plant.Np
    nx = plant.nx

    if Q_list is None or R_list is None or P_list is None:
        Q_list, R_list, P_list = default_local_weights(plant)

    # ── prediction matrices ───────────────────────────────────────────────────
    B_list = [plant.B_j(j) for j in range(M)]
    Phi_x, Gamma_list = build_prediction_matrices(plant.A, B_list, Np)

    # ── global state bounds (tiled over horizon) ──────────────────────────────
    x_lb_global = np.concatenate([s.x_lb for s in plant.subsystems])  # (nx,)
    x_ub_global = np.concatenate([s.x_ub for s in plant.subsystems])
    x_lb_rep_global = np.tile(x_lb_global, Np)   # (Np*nx,)
    x_ub_rep_global = np.tile(x_ub_global, Np)

    agents = []
    for i in range(M):
        si    = plant.subsystems[i]
        nu_i  = si.nu
        n_u_i = Np * nu_i

        others  = [j for j in range(M) if j != i]
        n_u_neg = sum(Np * plant.subsystems[j].nu for j in others)

        Gamma_i      = Gamma_list[i]                        # (Np*nx, Np*nu_i)
        Gamma_others = [Gamma_list[j] for j in others]

        # ── local cost matrices ───────────────────────────────────────────────
        Q_full_i = _build_local_Q_full(plant, Q_list[i], P_list[i], i)
        R_bar_i  = np.kron(np.eye(Np), R_list[i])           # (Np*nu_i, Np*nu_i)

        H_qp_i = Gamma_i.T @ Q_full_i @ Gamma_i + R_bar_i  # (n_u_i, n_u_i)

        # ── parametric cost (dimpc ordering: [Phi_x | Gamma_others]) ─────────
        M_theta_dimpc = np.hstack([Phi_x] + Gamma_others)   # (Np*nx, nx + n_u_neg)
        H_par_full    = Gamma_i.T @ Q_full_i @ M_theta_dimpc # (n_u_i, nx + n_u_neg)

        # GNE ordering θ_i = [U_{-i}; x_0]:
        F_i       = H_par_full[:, :nx]    # (n_u_i, nx) — cost from x_0 = p
        F_cross_i = H_par_full[:, nx:]    # (n_u_i, n_u_neg) — cost from U_{-i}

        # ── input bound constraints (local, no parametric dependence on p) ───
        u_ub_rep = np.tile(si.u_ub, Np)   # (Np*nu_i,)
        u_lb_rep = np.tile(si.u_lb, Np)
        A_loc = np.vstack([ np.eye(n_u_i), -np.eye(n_u_i)])  # (2*n_u_i, n_u_i)
        b_loc = np.concatenate([u_ub_rep, -u_lb_rep])         # (2*n_u_i,)
        S_loc = np.zeros((2 * n_u_i, nx))                     # no p-dependence

        # ── coupling-mode-specific constraint data ────────────────────────────
        if coupling_mode == "state_bounds":
            # Generalized Nash: state bounds couple agents through trajectory.
            # ±Gamma_i U_i ≤ ±x_{ub/lb}_rep ∓ M_theta θ_i
            agents.append(Agent(
                index=i,
                n_x=n_u_i,
                Q=H_qp_i,
                c=np.zeros(n_u_i),
                F=F_i,
                F_cross=F_cross_i,
                A_loc=A_loc,
                b_loc=b_loc,
                S_loc=S_loc,
                Gamma_self=Gamma_i.copy(),
                M_theta=M_theta_dimpc.copy(),
                x_lb_rep=x_lb_rep_global.copy(),
                x_ub_rep=x_ub_rep_global.copy(),
            ))
        else:  # "l_max"
            # Aggregate-input coupling: Σ_j u_{j,k} ≤ L_max per step.
            # C_i = I_Np ⊗ ones(1, nu_i)  so that C_i U_i = sum of u_{i,k} per step.
            C_i = np.kron(np.eye(Np), np.ones((1, nu_i)))   # (Np, Np*nu_i)
            agents.append(Agent(
                index=i,
                n_x=n_u_i,
                Q=H_qp_i,
                c=np.zeros(n_u_i),
                F=F_i,
                F_cross=F_cross_i,
                C=C_i,
                A_loc=A_loc,
                b_loc=b_loc,
                S_loc=S_loc,
            ))

    # ── game-level coupling data and parameter space ──────────────────────────
    p_lb = x_lb_global.copy()
    p_ub = x_ub_global.copy()

    if coupling_mode == "l_max":
        d      = L_max * np.ones(Np)       # (Np,)  one limit per prediction step
        S_coup = np.zeros((Np, nx))        # RHS does not depend on x_0
    else:
        d      = None
        S_coup = None

    return GNEGame(
        agents=agents,
        d=d,
        S_coup=S_coup,
        p_lb=p_lb,
        p_ub=p_ub,
    )
