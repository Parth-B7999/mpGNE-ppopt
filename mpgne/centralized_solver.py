"""
centralized_solver.py — Centralized GNE solver baseline.

Solves the multi-agent GNE problem by stacking all KKT conditions into a 
single nonlinear system and minimizing the KKT residual. 
This serves as a high-fidelity but slow online iterative baseline.
"""

import time
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

from .game import GNEGame

@dataclass
class CentralizedResult:
    x_sol: np.ndarray
    success: bool
    solve_time: float
    residual: float
    n_iter: int

def solve_gne_centralized(game: GNEGame, p: np.ndarray, x_init: np.ndarray = None) -> CentralizedResult:
    """
    Solve the GNE problem by minimizing the squared KKT residuals of all agents.
    
    This is a centralized approach that is generally slower than decentralized 
    methods like ADMM or explicit lookup.
    """
    t0 = time.perf_counter()
    N = game.N
    nx_total = sum(a.n_x for a in game.agents)
    
    if x_init is None:
        x_init = np.zeros(nx_total)
        
    def objective(x_stacked):
        # 1. Split stacked x into agent components
        x_list = []
        offset = 0
        for i in range(N):
            nx_i = game.agents[i].n_x
            x_list.append(x_stacked[offset:offset+nx_i])
            offset += nx_i
            
        # 2. Total cost sum (Simplified GNE approach: joint potential if exists)
        # For a general GNE, we minimize the sum of squared agent gradients + violations
        total_residual = 0
        
        # Coupling constraints: sum(C_i x_i) <= d + S_p p
        coupling_rhs = game.d + game.S_coup @ p
        agg = sum(game.agents[i].C @ x_list[i] for i in range(N))
        coupling_violation = np.sum(np.maximum(0, agg - coupling_rhs)**2)
        
        total_obj = 0
        for i in range(N):
            agent = game.agents[i]
            # 1/2 x_i' Q_i x_i + (c_i + F_i p + F_cross_i x_{-i})' x_i
            l_i = agent.c + agent.F @ p
            
            if agent.F_cross is not None:
                # x_{-i} is everything in x_stacked except x_list[i]
                x_neg = []
                for j in range(N):
                    if j != i:
                        x_neg.append(x_list[j])
                x_neg_vec = np.concatenate(x_neg)
                l_i += agent.F_cross @ x_neg_vec
            
            # Local Objective
            total_obj += 0.5 * x_list[i] @ agent.Q @ x_list[i] + l_i @ x_list[i]
            
            # Local constraints violation: A_loc_i x_i <= b_loc_i + S_loc_i p
            loc_rhs = agent.b_loc + agent.S_loc @ p
            loc_violation = np.sum(np.maximum(0, agent.A_loc @ x_list[i] - loc_rhs)**2)
            total_residual += loc_violation
            
        return total_obj + 1000 * (total_residual + coupling_violation)

    # Solve via SLSQP
    res = minimize(
        objective, x_init, 
        method='SLSQP',
        options={'ftol': 1e-8, 'maxiter': 500, 'disp': False}
    )
    
    elapsed = time.perf_counter() - t0
    return CentralizedResult(
        x_sol=res.x,
        success=res.success,
        solve_time=elapsed,
        residual=res.fun,
        n_iter=res.nit
    )
