"""
full_case_study.py
==================
Full reproduction of the ACC 2026/CCE case study for mpGNE.
Runs M ∈ [2, 3, 4, 5, 6] agents across 100 random plants each.

Benchmarks:
  - ADMM
  - Jacobi BR
  - ImpGNE
  - FACET-H (Hyperplane Adjacency)
  - FACET-LP (LP Facet Adjacency)

Saves all offline mpQP solutions, neighbor maps, and (for M <= 3) full BFS explicit maps.
Generates performance tables and plots summarizing the benchmark.

Run:
  python tests/full_case_study.py
"""
# %% ──────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pickle
import traceback
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ppopt.mp_solvers.solve_mpqp import mpqp_algorithm

from mpgne.plant_gen import make_random_plants, make_ic
from mpgne.mpc_builder import make_gne_game_from_plant, default_local_weights
from mpgne.mp_solver import solve_all_agents_mp
from mpgne.cr_store import save_agent_solutions, load_agent_solutions, save_gne_solution, load_gne_solution, GNESolution
from mpgne.admm_solver import admm_solve
from mpgne.proj_grad_solver import pg_solve
from mpgne.impdimc_solver import impdimc_solve
from mpgne.facet_gne import (find_all_agent_cr_neighbors, build_gne_solution_facet,
                              solve_gne_online_v2,
                              precompute_point_location_arrays,
                              refine_neighbors_with_lp,
                              build_gne_solution_lp_from_fh)

# %% ── 0. Configuration ─────────────────────────────────────────────────────────
N_PLANTS = 3
T_SIM = 100
M_LIST = [4]
OFFLINE_BFS_MAX_M = 2 # Save full BFS explicit maps for M <= this value (M=3 state_bounds too slow)

ALGO = mpqp_algorithm.combinatorial_parallel_exp
ADMM_RHO = 1.0
ADMM_ITERS = 2000       # iterations for the ADMM benchmark method
ADMM_TOL = 1e-4
FALLBACK_ITERS = 25    # iterations for ADMM used as fallback inside FACET/CI
FALLBACK_TOL = 1e-3     # looser tolerance — fallback only needs to re-seed prev_combo
BR_ITERS = 2000
BR_TOL = 1e-4
IMP_ITERS = 200
IMP_TOL = 1e-4
QP_SOLVER = "osqp"
SEED = 250
# Coupling formulation:
#   "state_bounds" — generalized Nash via x_lb ≤ x_k ≤ x_ub  (default, ACC 2026)
#   "l_max"        — aggregate-input coupling  Σ_j u_{j,k} ≤ L_MAX  (earlier formulation)
COUPLING_MODE = "state_bounds"
L_MAX = 2.5  # only used when COUPLING_MODE = "l_max"

CKPT_DIR = os.path.join(os.path.dirname(__file__), f"full_case_study_data_{COUPLING_MODE}")
os.makedirs(CKPT_DIR, exist_ok=True)

# Module-level variables so every # %% cell can reference them directly.
coupling_label = f"l_max (L_MAX={L_MAX})" if COUPLING_MODE == "l_max" else "state_bounds"

# Pre-load any existing checkpoints so downstream cells work immediately.
def _ckpt_path_early(M, idx, name):
    return os.path.join(CKPT_DIR, f"M{M}_plant{idx:03d}_{name}.pkl")

all_results = {M: [] for M in M_LIST}
for _M in M_LIST:
    for _idx in range(N_PLANTS):
        _p = _ckpt_path_early(_M, _idx, "results")
        if os.path.exists(_p):
            with open(_p, "rb") as _f:
                all_results[_M].append(pickle.load(_f))



# %% ── 1. Checkpointing Helpers ─────────────────────────────────────────────────
def _ckpt_path(M, idx, name, ckpt_dir):
    return os.path.join(ckpt_dir, f"M{M}_plant{idx:03d}_{name}.pkl")

def _save(M, idx, name, data, ckpt_dir):
    with open(_ckpt_path(M, idx, name, ckpt_dir), "wb") as f:
        pickle.dump(data, f)

def _load(M, idx, name, ckpt_dir):
    p = _ckpt_path(M, idx, name, ckpt_dir)
    if not os.path.exists(p): return None
    with open(p, "rb") as f:
        return pickle.load(f)

# %% ── 2. Simulation Helpers ────────────────────────────────────────────────────
def _run_admm_sim(plant, game, x0, T):
    nx, M = plant.nx, plant.M
    x_traj = np.zeros((T+1, nx))
    x_traj[0] = x0.copy()
    times, iters = np.zeros(T), np.zeros(T, dtype=int)
    for k in range(T):
        p = x_traj[k]
        t0 = time.perf_counter()
        res = admm_solve(game, p, rho=ADMM_RHO, max_iter=ADMM_ITERS, tol=ADMM_TOL,
                         qp_solver=QP_SOLVER, verbose=False)
        times[k] = time.perf_counter() - t0
        iters[k] = res.n_iter
        u_k = {i: res.x_sol[i][:plant.subsystems[i].nu] for i in range(M)}
        x_traj[k+1] = plant.step(x_traj[k], u_k)
    return x_traj, times, iters

def _run_br_sim(plant, game, x0, T):
    nx, M = plant.nx, plant.M
    x_traj = np.zeros((T+1, nx))
    x_traj[0] = x0.copy()
    times, iters = np.zeros(T), np.zeros(T, dtype=int)
    prev_x = None                   # warm-start: previous step's x_sol
    for k in range(T):
        p = x_traj[k]
        t0 = time.perf_counter()
        res = pg_solve(game, p, max_iter=BR_ITERS, tol=BR_TOL,
                       qp_solver=QP_SOLVER, verbose=False, x_init=prev_x)
        times[k] = time.perf_counter() - t0
        iters[k] = res.n_iter
        prev_x = res.x_sol          # carry forward for next step
        u_k = {i: res.x_sol[i][:plant.subsystems[i].nu] for i in range(M)}
        x_traj[k+1] = plant.step(x_traj[k], u_k)
    return x_traj, times, iters

def _run_impgne_sim(plant, game, agent_sols, x0, T):
    nx, M = plant.nx, plant.M
    x_traj = np.zeros((T+1, nx))
    x_traj[0] = x0.copy()
    times, iters = np.zeros(T), np.zeros(T, dtype=int)
    prev_U = None                   # warm-start: paper Eq.(13) — U(k-1) → U^(0)(k)
    for k in range(T):
        p = x_traj[k]
        t0 = time.perf_counter()
        res = impdimc_solve(game, p, agent_sols, max_iter=IMP_ITERS, tol=IMP_TOL,
                            verbose=False, U_init=prev_U)
        times[k] = time.perf_counter() - t0
        iters[k] = res.n_iter
        prev_U = res.x_stacked      # carry forward full U_bar for next step
        u_k = {i: res.x_sol[i][:plant.subsystems[i].nu] for i in range(M)}
        x_traj[k+1] = plant.step(x_traj[k], u_k)
    return x_traj, times, iters

def _run_facet_sim(plant, game, agent_sols, x0, T, facet_sol=None):
    """
    Simulate closed-loop with the FACET explicit GNE method.

    Two modes (selected automatically based on whether facet_sol is provided):
      MODE A (M <= OFFLINE_BFS_MAX_M, facet_sol provided):
            Direct lookup in the pre-computed GNESolution (p-space CRs).
            Falls back to ADMM if state leaves the map.
      MODE B (M > OFFLINE_BFS_MAX_M, facet_sol=None):
            solve_gne_online_v2: PointLocation → scored 1-hop filter → linsolve.
            Tier-2 combo_cache scan before ADMM fallback.

    Timing: ADMM cold-start at k=0 is excluded; fallback ADMM at k>0 IS
    included in times[] for honest end-to-end benchmarking.
    """
    nx, M = plant.nx, plant.M
    x_traj = np.zeros((T+1, nx))
    x_traj[0] = x0.copy()
    times, iters = np.zeros(T), np.zeros(T, dtype=int)
    u_traj = {i: np.zeros((T, plant.subsystems[i].nu)) for i in range(M)}
    # Precompute stacked E/f arrays for vectorized PointLocation (once per call).
    # Replaces the O(n_cr) Python loop with a single batched matmul.
    if not hasattr(agent_sols[0], '_E_stack'):
        precompute_point_location_arrays(agent_sols)

    fallbacks   = 0
    prev_x_star = None   # MODE B: stacked U* from previous step (V2 reference point)
    prev_crs    = None   # MODE B: per-agent CR indices from previous step (warm hint)
    combo_cache = {}     # MODE B: {combo → (H_x, h_x)} persisted across all steps

    def _unstack(U_star):
        """Extract per-agent first control action from stacked U."""
        u = {}; off = 0
        for i in range(M):
            nu_i = plant.subsystems[i].nu
            u[i] = U_star[off:off + nu_i]
            off += plant.Np * nu_i
        return u

    for k in range(T):
        p = x_traj[k]
        t_start = time.perf_counter()
        t_admm_penalty = 0.0

        # ══════════════════════════════════════════════════════════════════════
        # MODE A — Explicit map (M <= OFFLINE_BFS_MAX_M)
        #   Direct lookup into the precomputed GNESolution (p-space CRs).
        #   ADMM fallback only if state leaves the precomputed map.
        #   Data transfer = 1 per step (no iterative negotiation).
        # ══════════════════════════════════════════════════════════════════════
        if facet_sol is not None:
            cr_idx = facet_sol.locate(p, tol=1e-6)
            if cr_idx is not None:
                U_star = facet_sol.regions[cr_idx].evaluate(p)
                u_k = _unstack(U_star)
                iters[k] = 1
            else:
                fallbacks += 1
                t_admm_start = time.perf_counter()
                res = admm_solve(game, p, rho=ADMM_RHO, max_iter=FALLBACK_ITERS,
                                 tol=FALLBACK_TOL, qp_solver=QP_SOLVER, verbose=False)
                if k == 0:
                    t_admm_penalty += time.perf_counter() - t_admm_start
                u_k = {i: res.x_sol[i][:plant.subsystems[i].nu] for i in range(M)}
                iters[k] = 1 + res.n_iter

        # ══════════════════════════════════════════════════════════════════════
        # MODE B — V2 online solver (M > OFFLINE_BFS_MAX_M)
        #   Mirrors MATLAB IF_mpDiMPC_V2: PointLocation → per-agent filter
        #   → combo enumeration. No BFS, no LP, no combo cache.
        #   Cold-start seeded by ADMM at k=0; ADMM fallback if V2 fails.
        # ══════════════════════════════════════════════════════════════════════
        else:
            if prev_x_star is None:
                t_admm_start = time.perf_counter()
                res_warm = admm_solve(game, p, rho=ADMM_RHO, max_iter=500,
                                      tol=1e-4, qp_solver=QP_SOLVER, verbose=False)
                if k == 0:
                    t_admm_penalty += time.perf_counter() - t_admm_start
                prev_x_star = res_warm.x_stacked   # seed reference point

            combo, U_star, n_checked = solve_gne_online_v2(
                p, agent_sols, game, prev_x_star=prev_x_star,
                prev_crs=prev_crs, combo_cache=combo_cache)

            if combo is not None:
                prev_x_star = U_star               # carry forward for next step
                prev_crs    = list(combo)          # warm hint for PointLocation
                u_k = _unstack(U_star)
                iters[k] = n_checked               # true data transfer count
            else:
                fallbacks += 1
                t_admm_start = time.perf_counter()
                res = admm_solve(game, p, rho=ADMM_RHO, max_iter=FALLBACK_ITERS,
                                 tol=FALLBACK_TOL, qp_solver=QP_SOLVER, verbose=False)
                if k == 0:
                    t_admm_penalty += time.perf_counter() - t_admm_start
                u_k = {i: res.x_sol[i][:plant.subsystems[i].nu] for i in range(M)}
                prev_x_star = res.x_stacked        # re-seed from fallback ADMM
                prev_crs    = None                 # reset hint after fallback
                iters[k] = 1 + res.n_iter

        times[k] = (time.perf_counter() - t_start) - t_admm_penalty
        for i in range(M): u_traj[i][k] = u_k[i]
        x_traj[k+1] = plant.step(x_traj[k], u_k)

    return x_traj, u_traj, times, iters, fallbacks

def _run_explicit_sim(plant, fsol, x0, T):
    nx, M = plant.nx, plant.M
    x_traj = np.zeros((T+1, nx))
    x_traj[0] = x0.copy()
    times, iters = np.zeros(T), np.zeros(T, dtype=int)
    u_traj = {i: np.zeros((T, plant.subsystems[i].nu)) for i in range(M)}

    for k in range(T):
        p = x_traj[k]
        t0 = time.perf_counter()

        _vi, _best_v = -1, float('inf')
        for v, cr in enumerate(fsol.regions):
            vl = float(np.max(cr.D @ p - cr.e))
            if vl <= 1e-6:
                _vi = v; break
            if vl < _best_v:
                _best_v, _vi = vl, v

        U_star = fsol.regions[_vi].H_x @ p + fsol.regions[_vi].h_x
        times[k] = time.perf_counter() - t0
        iters[k] = 1

        u_k = {}
        offset = 0
        for i in range(M):
            nu_i = plant.subsystems[i].nu
            u_k[i] = U_star[offset:offset+plant.Np*nu_i][:nu_i]
            offset += plant.Np*nu_i
            u_traj[i][k] = u_k[i]

        x_traj[k+1] = plant.step(x_traj[k], u_k)

    return x_traj, u_traj, times, iters

# %% ── 3. Plant Worker ──────────────────────────────────────────────────────────
def run_one_plant(plant, x0, M, idx, coupling_mode=COUPLING_MODE, L_max=L_MAX, ckpt_dir=CKPT_DIR):
    res_dict = _load(M, idx, "results", ckpt_dir)
    if res_dict is not None:
        return res_dict

    res_dict = {
        "M": M, "idx": idx,
        "n_crs": [], "n_combos_fh": None, "n_combos_flp": None,
        "time_mpqp": 0, "time_fh": 0, "time_flp": 0,
        "methods": {}
    }

    try:
        print(f"  [M{M}/{idx:03d}] [1/7] Building game (coupling={coupling_mode})...", flush=True)
        t_step = time.perf_counter()
        Q, R, P = default_local_weights(plant)
        game = make_gne_game_from_plant(plant, coupling_mode=coupling_mode, L_max=L_max,
                                        Q_list=Q, R_list=R, P_list=P)
        print(f"  [M{M}/{idx:03d}] [1/7] Game built ({time.perf_counter()-t_step:.1f}s)", flush=True)
    except Exception as e:
        res_dict["error"] = f"game build: {e}"
        return res_dict

    # 1. Base mpQP
    agent_sols_base = _load(M, idx, "agent_sols_base", ckpt_dir)
    if agent_sols_base is None:
        print(f"  [M{M}/{idx:03d}] [2/7] Solving base mpQP (PPOPT)...", flush=True)
        t0 = time.perf_counter()
        agent_sols_base = solve_all_agents_mp(game, algorithm=ALGO, verbose=False)
        res_dict["time_mpqp"] = time.perf_counter() - t0
        _save(M, idx, "agent_sols_base", agent_sols_base, ckpt_dir)
        print(f"  [M{M}/{idx:03d}] [2/7] mpQP done ({res_dict['time_mpqp']:.1f}s)", flush=True)
    else:
        print(f"  [M{M}/{idx:03d}] [2/7] mpQP loaded from checkpoint", flush=True)
    res_dict["n_crs"] = [s.n_cr for s in agent_sols_base]

    # 2. FACET-H neighbors
    agent_sols_FH = _load(M, idx, "agent_sols_FH", ckpt_dir)
    if agent_sols_FH is None:
        print(f"  [M{M}/{idx:03d}] [3/7] FACET-H neighbor detection...", flush=True)
        agent_sols_FH = copy.deepcopy(agent_sols_base)
        t0 = time.perf_counter()
        find_all_agent_cr_neighbors(agent_sols_FH, method="hyperplane_adjacency", verbose=False)
        res_dict["time_fh"] = time.perf_counter() - t0
        _save(M, idx, "agent_sols_FH", agent_sols_FH, ckpt_dir)
        print(f"  [M{M}/{idx:03d}] [3/7] FACET-H done ({res_dict['time_fh']:.1f}s)", flush=True)
    else:
        print(f"  [M{M}/{idx:03d}] [3/7] FACET-H loaded from checkpoint", flush=True)

    # 3. FACET-LP neighbors — LP-refine FACET-H results (skip O(n²) rescan)
    agent_sols_FLP = _load(M, idx, "agent_sols_FLP", ckpt_dir)
    if agent_sols_FLP is None:
        print(f"  [M{M}/{idx:03d}] [4/7] FACET-LP neighbor detection (LP-refine FH)...", flush=True)
        agent_sols_FLP = copy.deepcopy(agent_sols_FH)
        t0 = time.perf_counter()
        refine_neighbors_with_lp(agent_sols_FLP, verbose=True)
        res_dict["time_flp"] = time.perf_counter() - t0
        _save(M, idx, "agent_sols_FLP", agent_sols_FLP, ckpt_dir)
        print(f"  [M{M}/{idx:03d}] [4/7] FACET-LP done ({res_dict['time_flp']:.1f}s)", flush=True)
    else:
        print(f"  [M{M}/{idx:03d}] [4/7] FACET-LP loaded from checkpoint", flush=True)

    # 4. Offline BFS Map (Save for M <= 3)
    print(f"  [M{M}/{idx:03d}] [5/7] Building BFS GNE maps...", flush=True)
    t_bfs = time.perf_counter()
    facet_sol_FH = None
    facet_sol_FLP = None
    if M <= OFFLINE_BFS_MAX_M:
        facet_sol_FH = _load(M, idx, "facet_sol_FH", ckpt_dir)
        if facet_sol_FH is None or facet_sol_FH.n_cr == 0:
            print(f"  [M{M}/{idx:03d}] [5/7]   BFS FH: searching combos...", flush=True)
            t0 = time.perf_counter()
            fres = build_gne_solution_facet(game, agent_sols_FH, verbose=True)
            facet_sol_FH = fres.gne_sol
            res_dict["n_combos_fh"] = fres.n_combos_checked
            print(f"  [M{M}/{idx:03d}] [5/7]   BFS FH: {facet_sol_FH.n_cr} GNE CRs ({time.perf_counter()-t0:.1f}s)", flush=True)
            if facet_sol_FH.n_cr > 0:
                _save(M, idx, "facet_sol_FH", facet_sol_FH, ckpt_dir)
            else:
                facet_sol_FH = None

        facet_sol_FLP = _load(M, idx, "facet_sol_FLP", ckpt_dir)
        if facet_sol_FLP is None or facet_sol_FLP.n_cr == 0:
            print(f"  [M{M}/{idx:03d}] [5/7]   BFS FLP: deriving from FH solution (fast)...", flush=True)
            t0 = time.perf_counter()
            facet_sol_FLP, _ = build_gne_solution_lp_from_fh(
                agent_sols_FLP, facet_sol_FH, verbose=True)
            res_dict["n_combos_flp"] = facet_sol_FLP.n_cr
            print(f"  [M{M}/{idx:03d}] [5/7]   BFS FLP: {facet_sol_FLP.n_cr} GNE CRs "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)
            if facet_sol_FLP.n_cr > 0:
                _save(M, idx, "facet_sol_FLP", facet_sol_FLP, ckpt_dir)
            else:
                facet_sol_FLP = None
    print(f"  [M{M}/{idx:03d}] [5/7] BFS maps done ({time.perf_counter()-t_bfs:.1f}s)", flush=True)

    # 6. Online Simulations
    print(f"  [M{M}/{idx:03d}] [6/7] Library warmup...", flush=True)
    t_warm = time.perf_counter()
    # Warmup libraries
    _p0 = x0.copy()
    admm_solve(game, _p0, max_iter=1, tol=1e-20, qp_solver=QP_SOLVER)
    pg_solve(game, _p0, max_iter=1, tol=1e-20, qp_solver=QP_SOLVER)
    impdimc_solve(game, _p0, agent_sols_base, max_iter=1, tol=1e-20)
    solve_gne_online_v2(_p0, agent_sols_FH,  game)
    solve_gne_online_v2(_p0, agent_sols_FLP, game)
    if M <= OFFLINE_BFS_MAX_M and facet_sol_FH is not None:
        _, _, _, _ = _run_explicit_sim(plant, facet_sol_FH, _p0, 1)
    import scipy.optimize
    scipy.optimize.linprog(c=[1], A_ub=[[1]], b_ub=[1], bounds=(0,1), method='highs')
    print(f"  [M{M}/{idx:03d}] [6/7] Warmup done ({time.perf_counter()-t_warm:.1f}s)", flush=True)

    try:
        print(f"  [M{M}/{idx:03d}] [7/7] Online sim ({T_SIM} steps × 6 methods)...", flush=True)
        t_online = time.perf_counter()
        # Explicit GNE (FH BFS map)
        if M <= OFFLINE_BFS_MAX_M and facet_sol_FH is not None:
            xt, ut, ts, itrs = _run_explicit_sim(plant, facet_sol_FH, x0, T_SIM)
            res_dict["methods"]["Explicit"] = {"avg_time": ts.mean(), "max_time": ts.max(), "iters": itrs.mean(), "total_iters": int(itrs.sum()), "norm": np.linalg.norm(xt[-1])}
        else:
            res_dict["methods"]["Explicit"] = None

        # ADMM
        print(f"  [M{M}/{idx:03d}] [7/7]   ADMM...", flush=True)
        t_a = time.perf_counter()
        xt, ts, itrs = _run_admm_sim(plant, game, x0, T_SIM)
        res_dict["methods"]["ADMM"] = {"avg_time": ts.mean(), "max_time": ts.max(), "iters": itrs.mean(), "total_iters": int(itrs.sum()), "norm": np.linalg.norm(xt[-1])}
        print(f"  [M{M}/{idx:03d}] [7/7]   ADMM done ({time.perf_counter()-t_a:.1f}s)", flush=True)
        # Jacobi BR
        print(f"  [M{M}/{idx:03d}] [7/7]   BR...", flush=True)
        t_b = time.perf_counter()
        xt, ts, itrs = _run_br_sim(plant, game, x0, T_SIM)
        res_dict["methods"]["BR"] = {"avg_time": ts.mean(), "max_time": ts.max(), "iters": itrs.mean(), "total_iters": int(itrs.sum()), "norm": np.linalg.norm(xt[-1])}
        print(f"  [M{M}/{idx:03d}] [7/7]   BR done ({time.perf_counter()-t_b:.1f}s)", flush=True)
        # ImpGNE
        print(f"  [M{M}/{idx:03d}] [7/7]   ImpGNE...", flush=True)
        t_i = time.perf_counter()
        xt, ts, itrs = _run_impgne_sim(plant, game, agent_sols_base, x0, T_SIM)
        res_dict["methods"]["ImpGNE"] = {"avg_time": ts.mean(), "max_time": ts.max(), "iters": itrs.mean(), "total_iters": int(itrs.sum()), "norm": np.linalg.norm(xt[-1])}
        print(f"  [M{M}/{idx:03d}] [7/7]   ImpGNE done ({time.perf_counter()-t_i:.1f}s)", flush=True)

        # FACET-H: BFS map lookup (M<=3) or online neighbor walk (M>=4) → ADMM fallback
        _fsol_FH = facet_sol_FH if (M <= OFFLINE_BFS_MAX_M and facet_sol_FH is not None) else None
        print(f"  [M{M}/{idx:03d}] [7/7]   FACET-H...", flush=True)
        t_h = time.perf_counter()
        xt, ut, ts, itrs, fbs = _run_facet_sim(plant, game, agent_sols_FH, x0, T_SIM,
                                                facet_sol=_fsol_FH)
        res_dict["methods"]["FACET-H"] = {"avg_time": ts.mean(), "max_time": ts.max(), "iters": itrs.mean(), "total_iters": int(itrs.sum()), "fallbacks": fbs, "norm": np.linalg.norm(xt[-1])}
        print(f"  [M{M}/{idx:03d}] [7/7]   FACET-H done ({time.perf_counter()-t_h:.1f}s)", flush=True)
        if idx == 0:
            res_dict["x_traj"], res_dict["u_traj"] = xt, ut
            res_dict["plant_nx"] = plant.nx
            res_dict["plant_nu"] = {i: plant.subsystems[i].nu for i in range(M)}

        # FACET-LP: BFS map lookup (M<=3) or online neighbor walk (M>=4) → ADMM fallback
        _fsol_FLP = facet_sol_FLP if (M <= OFFLINE_BFS_MAX_M and facet_sol_FLP is not None) else None
        print(f"  [M{M}/{idx:03d}] [7/7]   FACET-LP...", flush=True)
        t_l = time.perf_counter()
        xt, ut, ts, itrs, fbs = _run_facet_sim(plant, game, agent_sols_FLP, x0, T_SIM,
                                                facet_sol=_fsol_FLP)
        res_dict["methods"]["FACET-LP"] = {"avg_time": ts.mean(), "max_time": ts.max(), "iters": itrs.mean(), "total_iters": int(itrs.sum()), "fallbacks": fbs, "norm": np.linalg.norm(xt[-1])}
        print(f"  [M{M}/{idx:03d}] [7/7]   FACET-LP done ({time.perf_counter()-t_l:.1f}s)", flush=True)

        print(f"  [M{M}/{idx:03d}] [7/7] Online sim done ({time.perf_counter()-t_online:.1f}s)", flush=True)

    except Exception as e:
        res_dict["error"] = f"online sim: {e}"
        traceback.print_exc()
        
    _save(M, idx, "results", res_dict, ckpt_dir)
    return res_dict

def _generate_reports(all_results: dict, coupling_label: str, ckpt_dir: str):
    """Print benchmark tables and save plots. Call standalone after loading checkpoints."""
    SEP = "  " + "-"*86

    print("\n\n" + "="*90)
    print("  CASE STUDY RESULTS SUMMARY")
    print("  Coupling: " + coupling_label)
    print("="*90)

    # ── Table 1: Average Online Solution Time per step (ms) ────────────────────────
    print("\n  [TABLE 1]  Average Online Solution Time per Step (ms):")
    print(f"  {'M':>3} | {'Explicit':>10} | {'ADMM':>10} | {'Jacobi BR':>10} | {'ImpGNE':>10} "
          f"| {'FACET-H':>10} | {'FACET-LP':>10}")
    print(SEP)
    for M in M_LIST:
        keys = ["Explicit", "ADMM", "BR", "ImpGNE", "FACET-H", "FACET-LP"]
        times = {k: [] for k in keys}
        for r in all_results[M]:
            if r.get("error"): continue
            for k in keys:
                v = r["methods"].get(k)
                if v is not None:
                    times[k].append(v["avg_time"] * 1000)
        t = {k: np.mean(v) if v else float('nan') for k, v in times.items()}
        print(f"  {M:>3} | {t['Explicit']:>10.4f} | {t['ADMM']:>10.3f} | {t['BR']:>10.3f} "
              f"| {t['ImpGNE']:>10.3f} | {t['FACET-H']:>10.4f} | {t['FACET-LP']:>10.4f}")

    # ── Table 2: FACET Fallback Instances (steps where ADMM fallback was needed) ────
    print("\n  [TABLE 2]  FACET Fallback Instances (ADMM fallback steps out of T_SIM):")
    print(f"  {'M':>3} | {'FACET-H':>12} | {'FACET-LP':>12} | {'T_SIM':>7}")
    print(SEP)
    for M in M_LIST:
        fh_fb, flp_fb = [], []
        for r in all_results[M]:
            if r.get("error"): continue
            for key, lst in [("FACET-H", fh_fb), ("FACET-LP", flp_fb)]:
                v = r["methods"].get(key)
                if v is not None: lst.append(v.get("fallbacks", float('nan')))
        fh_avg    = np.nanmean(fh_fb)    if fh_fb    else float('nan')
        flp_avg   = np.nanmean(flp_fb)   if flp_fb   else float('nan')
        print(f"  {M:>3} | {fh_avg:>12.2f} | {flp_avg:>12.2f} | {T_SIM:>7}")

    # ── Table 3: Average Number of Critical Regions per Agent ──────────────────────
    print("\n  [TABLE 3]  Average Number of Critical Regions per Agent:")
    print(f"  {'M':>3} | {'Avg CRs / agent':>18} | {'Min CRs':>10} | {'Max CRs':>10} | {'Median CRs':>12}")
    print(SEP)
    for M in M_LIST:
        crs_flat = [c for r in all_results[M] if not r.get("error") for c in r["n_crs"]]
        if crs_flat:
            print(f"  {M:>3} | {np.mean(crs_flat):>18.1f} | {np.min(crs_flat):>10.0f} "
                  f"| {np.max(crs_flat):>10.0f} | {np.median(crs_flat):>12.1f}")
        else:
            print(f"  {M:>3} | {'nan':>18} | {'nan':>10} | {'nan':>10} | {'nan':>12}")

    # ── Table 4: Total Data Transfer Instances over T_SIM steps ─────────────────────
    print("\n  [TABLE 4]  Total Data Transfer Instances over T_SIM steps (communication rounds):")
    print(f"  {'M':>3} | {'Explicit':>10} | {'ADMM':>10} | {'Jacobi BR':>10} | {'ImpGNE':>10} "
          f"| {'FACET-H':>10} | {'FACET-LP':>10}")
    print(SEP)
    for M in M_LIST:
        keys = ["Explicit", "ADMM", "BR", "ImpGNE", "FACET-H", "FACET-LP"]
        totals = {k: [] for k in keys}
        for r in all_results[M]:
            if r.get("error"): continue
            for k in keys:
                v = r["methods"].get(k)
                if v is not None:
                    # prefer stored total_iters; fall back to mean * T_SIM for old checkpoints
                    totals[k].append(v.get("total_iters", v["iters"] * T_SIM))
        it = {k: np.mean(v) if v else float('nan') for k, v in totals.items()}
        print(f"  {M:>3} | {it['Explicit']:>10.0f} | {it['ADMM']:>10.0f} | {it['BR']:>10.0f} "
              f"| {it['ImpGNE']:>10.0f} | {it['FACET-H']:>10.0f} | {it['FACET-LP']:>10.0f}")

    # ── Plot 1: CRs boxplot ─────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    cr_data, cr_tick_labels = [], []
    for M in M_LIST:
        crs = [c for r in all_results[M] if not r.get("error") for c in r["n_crs"]]
        if crs:
            cr_data.append(crs); cr_tick_labels.append(str(M))
    if cr_data:
        plt.boxplot(cr_data, tick_labels=cr_tick_labels, patch_artist=True,
                    boxprops=dict(facecolor="lightblue"))
        plt.yscale('log')
        plt.xlabel("Number of Subsystems (M)", fontweight='bold')
        plt.ylabel("Number of Critical Regions per Agent", fontweight='bold')
        plt.title(f"Critical Regions Distribution [{coupling_label}]", fontweight='bold')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "crs_boxplot.png"), dpi=200)
    plt.close('all')

    # ── Plot 2: Online Solve Time boxplot ───────────────────────────────────────────
    plt.figure(figsize=(10, 6))
    time_data, td_tick_labels, colors = [], [], []
    for M in M_LIST:
        t_exp, t_admm, t_br, t_imp, t_fh, t_flp = [], [], [], [], [], []
        for r in all_results[M]:
            if r.get("error"): continue
            if r["methods"].get("Explicit"): t_exp.append(r["methods"]["Explicit"]["avg_time"] * 1000)
            if r["methods"].get("ADMM"):     t_admm.append(r["methods"]["ADMM"]["avg_time"]    * 1000)
            if r["methods"].get("BR"):       t_br.append(r["methods"]["BR"]["avg_time"]         * 1000)
            if r["methods"].get("ImpGNE"):   t_imp.append(r["methods"]["ImpGNE"]["avg_time"]    * 1000)
            if r["methods"].get("FACET-H"):  t_fh.append(r["methods"]["FACET-H"]["avg_time"]    * 1000)
            if r["methods"].get("FACET-LP"): t_flp.append(r["methods"]["FACET-LP"]["avg_time"]  * 1000)
        time_data.extend([t_exp, t_admm, t_br, t_imp, t_fh, t_flp])
        td_tick_labels.extend([f"Exp\nM={M}", f"ADMM\nM={M}", f"BR\nM={M}",
                                f"Imp\nM={M}", f"F-H\nM={M}", f"F-LP\nM={M}"])
        colors.extend(["#A0A0A0", "#E07B54", "#5B8DB8", "#C97BD4", "#4CAF82", "#FFD700"])
    if any(time_data):
        bp = plt.boxplot(time_data, tick_labels=td_tick_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        plt.yscale('log')
        plt.xticks(fontsize=7)
        for i in range(1, len(M_LIST)):
            plt.axvline(x=i*6 + 0.5, color='gray', linestyle='--', alpha=0.5)
        plt.ylabel("Online Solve Time (ms)", fontweight='bold')
        plt.title(f"Online Solve Time per Step [{coupling_label}]", fontweight='bold')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "times_boxplot.png"), dpi=200)
    plt.close('all')

    # ── Plot 3: Data Transfer boxplot (total over full simulation) ─────────────────
    plt.figure(figsize=(10, 6))
    iter_data, it_tick_labels, colors = [], [], []
    for M in M_LIST:
        i_admm, i_br, i_imp, i_fh, i_flp = [], [], [], [], []
        for r in all_results[M]:
            if r.get("error"): continue
            if r["methods"].get("ADMM"):
                v = r["methods"]["ADMM"]
                i_admm.append(v.get("total_iters", v["iters"] * T_SIM))
            if r["methods"].get("BR"):
                v = r["methods"]["BR"]
                i_br.append(v.get("total_iters", v["iters"] * T_SIM))
            if r["methods"].get("ImpGNE"):
                v = r["methods"]["ImpGNE"]
                i_imp.append(v.get("total_iters", v["iters"] * T_SIM))
            if r["methods"].get("FACET-H"):
                v = r["methods"]["FACET-H"]
                i_fh.append(v.get("total_iters", v["iters"] * T_SIM))
            if r["methods"].get("FACET-LP"):
                v = r["methods"]["FACET-LP"]
                i_flp.append(v.get("total_iters", v["iters"] * T_SIM))
        iter_data.extend([i_admm, i_br, i_imp, i_fh, i_flp])
        it_tick_labels.extend([f"ADMM\nM={M}", f"BR\nM={M}", f"Imp\nM={M}",
                                f"F-H\nM={M}", f"F-LP\nM={M}"])
        colors.extend(["#E07B54", "#5B8DB8", "#C97BD4", "#4CAF82", "#FFD700"])
    if any(iter_data):
        bp = plt.boxplot(iter_data, tick_labels=it_tick_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        plt.yscale('log')
        plt.xticks(fontsize=7)
        for i in range(1, len(M_LIST)):
            plt.axvline(x=i*5 + 0.5, color='gray', linestyle='--', alpha=0.5)
        plt.ylabel(f"Total Data Transfer (Communication Rounds, T_SIM={T_SIM})", fontweight='bold')
        plt.title(f"Total Data Transfer over Simulation [{coupling_label}]", fontweight='bold')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(ckpt_dir, "data_transfer_boxplot.png"), dpi=200)
    plt.close('all')

    # ── Plot 4: Trajectories ────────────────────────────────────────────────────────
    for M in M_LIST:
        r0 = next((r for r in all_results[M] if not r.get("error") and "x_traj" in r), None)
        if r0:
            xt, ut, nx = r0["x_traj"], r0["u_traj"], r0["plant_nx"]
            nus = r0["plant_nu"]
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            for i in range(nx):
                ax1.plot(np.arange(len(xt)), xt[:, i], linewidth=1.5)
            ax1.set_ylabel("States $x(k)$", fontweight='bold')
            ax1.set_title(f"Controller Performance for M={M}  [{coupling_label}]", fontweight='bold')
            ax1.grid(True, linestyle="--", alpha=0.7)
            for i in range(M):
                for j in range(nus[i]):
                    ax2.plot(np.arange(len(ut[i])), ut[i][:, j], drawstyle='steps-post', linewidth=1.5)
            ax2.set_ylabel("Control Inputs $u(k)$", fontweight='bold')
            ax2.set_xlabel("Time step $k$", fontweight='bold')
            ax2.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, f"trajectory_M{M}.png"), dpi=200)
            plt.close(fig)

    print(f"\nAll plots and data saved to: {ckpt_dir}")
    print("Done.")


# %% ── 4. Main Runner ────────────────────────────────────────────────────────────
def _run_for_coupling(coupling_mode: str, L_max: float) -> dict:
    """Run the full case study for one coupling formulation; return all_results."""
    global coupling_label
    coupling_label = f"l_max (L_MAX={L_max})" if coupling_mode == "l_max" else "state_bounds"
    ckpt_dir = os.path.join(os.path.dirname(__file__), f"full_case_study_data_{coupling_mode}")
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Coupling: {coupling_label}  |  N_PLANTS={N_PLANTS}, T_SIM={T_SIM}, M_LIST={M_LIST}")
    print(f"  Checkpoints → {ckpt_dir}")
    print(f"{'='*70}")

    all_results = {M: [] for M in M_LIST}

    for M in M_LIST:
        print(f"\n{'='*60}\n  Processing M = {M} ({N_PLANTS} plants)\n{'='*60}")
        plants = make_random_plants(M, N_PLANTS, seed=SEED+M)
        rng_ic = np.random.default_rng(SEED*2 + M)

        for idx, plant in enumerate(plants):
            x0 = make_ic(plant, scale=0.4, rng=rng_ic)
            res = run_one_plant(plant, x0, M, idx,
                                coupling_mode=coupling_mode, L_max=L_max,
                                ckpt_dir=ckpt_dir)
            all_results[M].append(res)
            if res.get("error"):
                print(f"  [{idx+1}/{N_PLANTS}] ERROR: {res['error']}")
            else:
                m = res["methods"]
                if "FACET-H" in m:
                    print(f"  [{idx+1:03d}/{N_PLANTS}] CRs={res['n_crs']} | "
                          f"ADMM: {m['ADMM']['avg_time']*1000:6.2f}ms | "
                          f"BR: {m['BR']['avg_time']*1000:6.2f}ms | "
                          f"ImpGNE: {m['ImpGNE']['avg_time']*1000:6.2f}ms | "
                          f"FH: {m['FACET-H']['avg_time']*1000:6.2f}ms | "
                          f"FLP: {m['FACET-LP']['avg_time']*1000:6.2f}ms")

    _generate_reports(all_results, coupling_label, ckpt_dir)
    return all_results


# %% ── 5. Entry point — script mode AND interactive top-to-bottom ────────────────
# One block only. Workers re-import with __name__='__mp_main__' so they skip this.
if __name__ == '__main__':
    # Load existing result checkpoints; detect any missing (M, idx) pairs.
    all_results = {M: [] for M in M_LIST}
    missing = []
    for M in M_LIST:
        for idx in range(N_PLANTS):
            r = _load(M, idx, "results", CKPT_DIR)
            if r is not None:
                all_results[M].append(r)
            else:
                missing.append((M, idx))

    if not missing:
        print(f"All results loaded from: {CKPT_DIR}")
        _generate_reports(all_results, coupling_label, CKPT_DIR)
    elif len(missing) == sum(N_PLANTS for _ in M_LIST):
        print(f"No checkpoints found for '{COUPLING_MODE}' — running full study...")
        all_results = _run_for_coupling(COUPLING_MODE, L_MAX)
    else:
        # Some results missing — re-run only the missing (M, idx) pairs.
        print(f"Partial results loaded. Re-running {len(missing)} missing plant(s): {missing}")
        for M, idx in missing:
            print(f"\n{'='*60}\n  Re-running M={M}, plant {idx}\n{'='*60}")
            plants = make_random_plants(M, N_PLANTS, seed=SEED + M)
            rng_ic = np.random.default_rng(SEED * 2 + M)
            x0s = [make_ic(plants[i], scale=0.4, rng=rng_ic) for i in range(idx + 1)]
            x0 = x0s[idx]
            res = run_one_plant(plants[idx], x0, M, idx,
                                coupling_mode=COUPLING_MODE, L_max=L_MAX,
                                ckpt_dir=CKPT_DIR)
            all_results[M].append(res)
        _generate_reports(all_results, coupling_label, CKPT_DIR)

# %%
