"""
mpc_builder.py — Factory: Plant → GNEGame for game-theoretic MPC.

Bridges the dimpc Plant structure (with random stable dynamics) to our
GNEGame framework by expanding the MPC problem over the prediction horizon.

Key difference from cooperative dimpc
──────────────────────────────────────
dimpc:  agents share a weighted GLOBAL cost (rho_i * sum of all states/inputs)
GNE:    each agent minimises ONLY its LOCAL cost (own states/inputs)
        + they share a COUPLING CONSTRAINT on aggregate input

Problem formulation
───────────────────
Agent i solves (game-theoretic MPC, Hall & Bemporad Eq. 18):

    min_{U_i}  ½ Σ_{k=0}^{Np-1} [ x_k^T Q_i_diag x_k + u_{i,k}^T R_i u_{i,k} ]

    s.t.  x_{k+1} = A x_k + Σ_j B_j u_{j,k}      (coupled dynamics)
          u_lb_i ≤ u_{i,k} ≤ u_ub_i               (local input bounds)
          Σ_j u_{j,k} ≤ L_max   for each k=0..Np-1 (shared coupling)
          x_0 = p                                   (initial state is parameter)

where Q_i_diag = blkdiag(0,..., Q_i, ..., 0) — only agent i's state block.

Expanding over the horizon using prediction matrices (Φ_x, Γ_j):
    X = Φ_x p + Γ_i U_i + Σ_{j≠i} Γ_j U_j

Collecting U_i terms (treating p = x_0 and U_{-i} as parameters θ_i = [U_{-i}; p]):

    H_qp_i   = Γ_i^T Q_i_full Γ_i  +  R_bar_i          (n_u_i × n_u_i)
    F_i       = Γ_i^T Q_i_full Φ_x                       (n_u_i × nx)   ← cost from p
    F_cross_i = Γ_i^T Q_i_full [Γ_{j1} | Γ_{j2} | ...]  (n_u_i × n_u_neg) ← cost from U_{-i}

Constraints in θ_i = [U_{-i}; p] form:
    Input bounds:  [I; -I] U_i  ≤  [u_ub_rep; -u_lb_rep]   (no θ_i dependence)
    Coupling:      C_i U_i      ≤  L_max*1 - C_{-i} U_{-i}  (θ_i dependence via U_{-i})

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
    L_max: float = 5.0,
    Q_list: list[np.ndarray] | None = None,
    R_list: list[np.ndarray] | None = None,
    P_list: list[np.ndarray] | None = None,
) -> GNEGame:
    """
    Convert a dimpc Plant into a GNEGame for game-theoretic MPC.

    Each agent i is a non-cooperative controller minimising only its own
    local MPC cost over the prediction horizon Np, subject to input bounds
    and a shared aggregate-input coupling constraint.

    Parameter vector: p = x_0 ∈ R^{nx_total}  (current global state)

    Coupling constraint: Σ_j u_{j,k} ≤ L_max  for each k = 0..Np-1
    Written in stacked form as:  Σ_j C_j U_j ≤ L_max * 1_{Np}
    where C_j = I_{Np} ⊗ ones(1, nu_j)  (shape: Np × Np*nu_j)

    Parameters
    ----------
    plant  : Plant  (from mpgne.plant_gen or dimpc plant_gen)
    L_max  : aggregate-input coupling limit per time step
    Q_list : local state cost matrices per agent  (default: identity)
    R_list : local input cost matrices per agent  (default: identity)
    P_list : terminal cost matrices per agent     (default: DARE solution)

    Returns
    -------
    GNEGame  with N = plant.M agents,  n_p = plant.nx,  n_coupling = plant.Np
    """
    M  = plant.M
    Np = plant.Np
    nx = plant.nx

    if Q_list is None or R_list is None or P_list is None:
        Q_list, R_list, P_list = default_local_weights(plant)

    # ── prediction matrices ───────────────────────────────────────────────────
    B_list = [plant.B_j(j) for j in range(M)]
    Phi_x, Gamma_list = build_prediction_matrices(plant.A, B_list, Np)

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
        # H_par_full = Gamma_i^T Q_full_i [Phi_x | Gamma_{j1} | ...]
        M_theta_dimpc = np.hstack([Phi_x] + Gamma_others)   # (Np*nx, nx + n_u_neg)
        H_par_full    = Gamma_i.T @ Q_full_i @ M_theta_dimpc # (n_u_i, nx + n_u_neg)

        # Split into GNE ordering θ_i = [U_{-i}; x_0]:
        F_i       = H_par_full[:, :nx]    # (n_u_i, nx) — cost from x_0 = p
        F_cross_i = H_par_full[:, nx:]    # (n_u_i, n_u_neg) — cost from U_{-i}

        # ── input bound constraints (local, no parametric dependence on p) ───
        u_ub_rep = np.tile(si.u_ub, Np)   # (Np*nu_i,)
        u_lb_rep = np.tile(si.u_lb, Np)
        A_loc = np.vstack([ np.eye(n_u_i), -np.eye(n_u_i)])  # (2*n_u_i, n_u_i)
        b_loc = np.concatenate([u_ub_rep, -u_lb_rep])         # (2*n_u_i,)
        S_loc = np.zeros((2 * n_u_i, nx))                     # no p-dependence

        # ── coupling block: C_i = I_Np ⊗ ones(1, nu_i) ───────────────────────
        # C_i U_i = [u_i(0) * ones; u_i(1) * ones; ...] summed → aggregate
        # For nu_i = 1: C_i = I_Np  (each row picks one input)
        C_i = np.kron(np.eye(Np), np.ones((1, nu_i)))         # (Np, Np*nu_i)

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

    # ── shared coupling constraint ────────────────────────────────────────────
    # Σ_j C_j U_j ≤ L_max * 1_{Np}
    d      = L_max * np.ones(Np)          # (Np,)
    S_coup = np.zeros((Np, nx))           # coupling RHS doesn't depend on x_0

    # ── parameter space: p = x_0 ∈ [x_lb_global, x_ub_global] ───────────────
    p_lb = np.concatenate([s.x_lb for s in plant.subsystems])
    p_ub = np.concatenate([s.x_ub for s in plant.subsystems])

    return GNEGame(
        agents=agents,
        d=d,
        S_coup=S_coup,
        p_lb=p_lb,
        p_ub=p_ub,
    )
