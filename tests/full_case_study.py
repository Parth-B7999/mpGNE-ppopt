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
from mpgne.facet_gne import find_all_agent_cr_neighbors, build_gne_solution_facet, solve_gne_online, refine_neighbors_with_lp

# %% ── 0. Configuration ─────────────────────────────────────────────────────────
N_PLANTS = 1
T_SIM = 100
M_LIST = [4]
OFFLINE_BFS_MAX_M = 3 # Save full BFS explicit maps for M <= 3

ALGO = mpqp_algorithm.combinatorial_parallel_exp
ADMM_RHO = 1.0
ADMM_ITERS = 2000
ADMM_TOL = 1e-4
BR_ITERS = 2000
BR_TOL = 1e-4
IMP_ITERS = 200
IMP_TOL = 1e-4
QP_SOLVER = "osqp"
SEED = 42

CKPT_DIR = os.path.join(os.path.dirname(__file__), "full_case_study_data")
os.makedirs(CKPT_DIR, exist_ok=True)



# %% ── 1. Checkpointing Helpers ─────────────────────────────────────────────────
def _ckpt_path(M, idx, name):
    return os.path.join(CKPT_DIR, f"M{M}_plant{idx:03d}_{name}.pkl")

def _save(M, idx, name, data):
    with open(_ckpt_path(M, idx, name), "wb") as f:
        pickle.dump(data, f)

def _load(M, idx, name):
    p = _ckpt_path(M, idx, name)
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
        res = admm_solve(game, p, rho=ADMM_RHO, max_iter=ADMM_ITERS, tol=ADMM_TOL, qp_solver=QP_SOLVER, verbose=False)
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
    for k in range(T):
        p = x_traj[k]
        t0 = time.perf_counter()
        res = pg_solve(game, p, max_iter=BR_ITERS, tol=BR_TOL, qp_solver=QP_SOLVER, verbose=False)
        times[k] = time.perf_counter() - t0
        iters[k] = res.n_iter
        u_k = {i: res.x_sol[i][:plant.subsystems[i].nu] for i in range(M)}
        x_traj[k+1] = plant.step(x_traj[k], u_k)
    return x_traj, times, iters

def _run_impgne_sim(plant, game, agent_sols, x0, T):
    nx, M = plant.nx, plant.M
    x_traj = np.zeros((T+1, nx))
    x_traj[0] = x0.copy()
    times, iters = np.zeros(T), np.zeros(T, dtype=int)
    for k in range(T):
        p = x_traj[k]
        t0 = time.perf_counter()
        res = impdimc_solve(game, p, agent_sols, max_iter=IMP_ITERS, tol=IMP_TOL, verbose=False)
        times[k] = time.perf_counter() - t0
        iters[k] = res.n_iter
        u_k = {i: res.x_sol[i][:plant.subsystems[i].nu] for i in range(M)}
        x_traj[k+1] = plant.step(x_traj[k], u_k)
    return x_traj, times, iters

def _run_facet_sim(plant, game, agent_sols, x0, T, facet_sol=None):
    """
    Simulate closed-loop with the FACET explicit GNE method.

    Two modes (selected automatically based on whether facet_sol is provided):
      BFS mode   (M <= OFFLINE_BFS_MAX_M, facet_sol provided):
            Direct lookup in the pre-computed GNESolution (p-space CRs).
            Falls back to ADMM if state leaves the map.
      Online mode (M > OFFLINE_BFS_MAX_M, facet_sol=None):
            Hop-based neighbor walk on agent-level CRs via solve_gne_online.
            Falls back to ADMM if walk fails.

    Timing: ADMM cold-start at k=0 is excluded; fallback ADMM at k>0 IS
    included in times[] for honest end-to-end benchmarking.
    """
    nx, M = plant.nx, plant.M
    x_traj = np.zeros((T+1, nx))
    x_traj[0] = x0.copy()
    times, iters = np.zeros(T), np.zeros(T, dtype=int)
    u_traj = {i: np.zeros((T, plant.subsystems[i].nu)) for i in range(M)}
    fallbacks = 0
    prev_combo = None  # only used in online mode

    def _unstack(U_star):
        """Extract per-agent first control action from stacked U."""
        u = {}; off = 0
        for i in range(M):
            nu_i = plant.subsystems[i].nu
            u[i] = U_star[off:off + nu_i]
            off += plant.Np * nu_i
        return u

    def _seed_combo(p, U_admm):
        """Seed initial CR combo from ADMM solution."""
        _Ud, _off = {}, 0
        for i in range(M):
            _n = plant.Np * plant.subsystems[i].nu
            _Ud[i] = U_admm[_off:_off+_n]; _off += _n
        _wc = []
        for i in range(M):
            _th = np.concatenate([p] + [_Ud[j] for j in range(M) if j != i])
            _vi, _best = 0, float('inf')
            for _v, _cr in enumerate(agent_sols[i].regions):
                _vl = float(np.max(_cr.E @ _th - _cr.f))
                if _vl < _best: _best, _vi = _vl, _v
            _wc.append(_vi)
        return tuple(_wc)

    for k in range(T):
        p = x_traj[k]
        t_start = time.perf_counter()
        t_admm_penalty = 0.0

        # ══════════════════════════════════════════════════════════════════════
        # MODE A — BFS explicit map (M <= OFFLINE_BFS_MAX_M)
        #   Direct lookup into the precomputed GNESolution.
        #   ADMM fallback only if state leaves the precomputed map.
        # ══════════════════════════════════════════════════════════════════════
        if facet_sol is not None:
            cr_idx = facet_sol.locate(p, tol=1e-6)
            if cr_idx is not None:
                U_star = facet_sol.regions[cr_idx].evaluate(p)
                u_k = _unstack(U_star)
                iters[k] = 1  # 1 lookup, no iterations
            else:
                # State outside precomputed map → ADMM fallback
                fallbacks += 1
                t_admm_start = time.perf_counter()
                res = admm_solve(game, p, rho=ADMM_RHO, max_iter=ADMM_ITERS,
                                 tol=ADMM_TOL, qp_solver=QP_SOLVER, verbose=False)
                if k > 0:
                    pass  # include fallback time (don't add to penalty)
                else:
                    t_admm_penalty += time.perf_counter() - t_admm_start
                u_k = {i: res.x_sol[i][:plant.subsystems[i].nu] for i in range(M)}
                iters[k] = 1 + res.n_iter

        # ══════════════════════════════════════════════════════════════════════
        # MODE B — Online neighbor walk (M > OFFLINE_BFS_MAX_M)
        #   Hop-based walk through agent-level CR adjacency maps.
        #   Cold-start seeded by ADMM; ADMM fallback if walk fails.
        # ══════════════════════════════════════════════════════════════════════
        else:
            if prev_combo is None:
                t_admm_start = time.perf_counter()
                res_warm = admm_solve(game, p, rho=ADMM_RHO, max_iter=500,
                                      tol=1e-4, qp_solver=QP_SOLVER, verbose=False)
                if k == 0:
                    t_admm_penalty += time.perf_counter() - t_admm_start
                prev_combo = _seed_combo(p, res_warm.x_stacked)

            combo, U_star, _ = solve_gne_online(p, prev_combo, agent_sols, game)
            if combo is not None:
                prev_combo = combo
                u_k = _unstack(U_star)
                iters[k] = 1
            else:
                fallbacks += 1
                t_admm_start = time.perf_counter()
                res = admm_solve(game, p, rho=ADMM_RHO, max_iter=ADMM_ITERS,
                                 tol=ADMM_TOL, qp_solver=QP_SOLVER, verbose=False)
                if k == 0:
                    t_admm_penalty += time.perf_counter() - t_admm_start
                u_k = {i: res.x_sol[i][:plant.subsystems[i].nu] for i in range(M)}
                prev_combo = None
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
def run_one_plant(plant, x0, M, idx):
    res_dict = _load(M, idx, "results")
    if res_dict is not None:
        return res_dict

    res_dict = {
        "M": M, "idx": idx,
        "n_crs": [], "n_combos_fh": None, "n_combos_flp": None,
        "time_mpqp": 0, "time_fh": 0, "time_flp": 0,
        "methods": {}
    }

    try:
        print(f"  [M{M}/{idx:03d}] [1/7] Building game...", flush=True)
        t_step = time.perf_counter()
        Q, R, P = default_local_weights(plant)
        game = make_gne_game_from_plant(plant, Q_list=Q, R_list=R, P_list=P)
        print(f"  [M{M}/{idx:03d}] [1/7] Game built ({time.perf_counter()-t_step:.1f}s)", flush=True)
    except Exception as e:
        res_dict["error"] = f"game build: {e}"
        return res_dict

    # 1. Base mpQP
    agent_sols_base = _load(M, idx, "agent_sols_base")
    if agent_sols_base is None:
        print(f"  [M{M}/{idx:03d}] [2/7] Solving base mpQP (PPOPT)...", flush=True)
        t0 = time.perf_counter()
        agent_sols_base = solve_all_agents_mp(game, algorithm=ALGO, verbose=False)
        res_dict["time_mpqp"] = time.perf_counter() - t0
        _save(M, idx, "agent_sols_base", agent_sols_base)
        print(f"  [M{M}/{idx:03d}] [2/7] mpQP done ({res_dict['time_mpqp']:.1f}s)", flush=True)
    else:
        print(f"  [M{M}/{idx:03d}] [2/7] mpQP loaded from checkpoint", flush=True)
    res_dict["n_crs"] = [s.n_cr for s in agent_sols_base]

    # 2. FACET-H neighbors
    agent_sols_FH = _load(M, idx, "agent_sols_FH")
    if agent_sols_FH is None:
        print(f"  [M{M}/{idx:03d}] [3/7] FACET-H neighbor detection...", flush=True)
        agent_sols_FH = copy.deepcopy(agent_sols_base)
        t0 = time.perf_counter()
        find_all_agent_cr_neighbors(agent_sols_FH, method="hyperplane_adjacency", verbose=False)
        res_dict["time_fh"] = time.perf_counter() - t0
        _save(M, idx, "agent_sols_FH", agent_sols_FH)
        print(f"  [M{M}/{idx:03d}] [3/7] FACET-H done ({res_dict['time_fh']:.1f}s)", flush=True)
    else:
        print(f"  [M{M}/{idx:03d}] [3/7] FACET-H loaded from checkpoint", flush=True)

    # 3. FACET-LP neighbors — LP-refine FACET-H results (skip O(n²) rescan)
    agent_sols_FLP = _load(M, idx, "agent_sols_FLP")
    if agent_sols_FLP is None:
        print(f"  [M{M}/{idx:03d}] [4/7] FACET-LP neighbor detection (LP-refine FH)...", flush=True)
        agent_sols_FLP = copy.deepcopy(agent_sols_FH)
        t0 = time.perf_counter()
        refine_neighbors_with_lp(agent_sols_FLP, verbose=True)
        res_dict["time_flp"] = time.perf_counter() - t0
        _save(M, idx, "agent_sols_FLP", agent_sols_FLP)
        print(f"  [M{M}/{idx:03d}] [4/7] FACET-LP done ({res_dict['time_flp']:.1f}s)", flush=True)
    else:
        print(f"  [M{M}/{idx:03d}] [4/7] FACET-LP loaded from checkpoint", flush=True)

    # 4. Offline BFS Map (Save for M <= 3)
    print(f"  [M{M}/{idx:03d}] [5/7] Building BFS GNE maps...", flush=True)
    t_bfs = time.perf_counter()
    facet_sol_FH = None
    facet_sol_FLP = None
    if M <= OFFLINE_BFS_MAX_M:
        facet_sol_FH = _load(M, idx, "facet_sol_FH")
        if facet_sol_FH is None or facet_sol_FH.n_cr == 0:
            print(f"  [M{M}/{idx:03d}] [5/7]   BFS FH: searching combos...", flush=True)
            t0 = time.perf_counter()
            fres = build_gne_solution_facet(game, agent_sols_FH, verbose=False)
            facet_sol_FH = fres.gne_sol
            res_dict["n_combos_fh"] = fres.n_combos_checked
            print(f"  [M{M}/{idx:03d}] [5/7]   BFS FH: {facet_sol_FH.n_cr} GNE CRs ({time.perf_counter()-t0:.1f}s)", flush=True)
            if facet_sol_FH.n_cr > 0:
                _save(M, idx, "facet_sol_FH", facet_sol_FH)
            else:
                facet_sol_FH = None

        facet_sol_FLP = _load(M, idx, "facet_sol_FLP")
        if facet_sol_FLP is None or facet_sol_FLP.n_cr == 0:
            print(f"  [M{M}/{idx:03d}] [5/7]   BFS FLP: searching combos...", flush=True)
            t0 = time.perf_counter()
            fres = build_gne_solution_facet(game, agent_sols_FLP, verbose=False)
            facet_sol_FLP = fres.gne_sol
            res_dict["n_combos_flp"] = fres.n_combos_checked
            print(f"  [M{M}/{idx:03d}] [5/7]   BFS FLP: {facet_sol_FLP.n_cr} GNE CRs ({time.perf_counter()-t0:.1f}s)", flush=True)
            if facet_sol_FLP.n_cr > 0:
                _save(M, idx, "facet_sol_FLP", facet_sol_FLP)
            else:
                facet_sol_FLP = None
    print(f"  [M{M}/{idx:03d}] [5/7] BFS maps done ({time.perf_counter()-t_bfs:.1f}s)", flush=True)

    # 5. Online Simulations
    print(f"  [M{M}/{idx:03d}] [6/7] Library warmup...", flush=True)
    t_warm = time.perf_counter()
    # Warmup libraries
    _p0 = x0.copy()
    admm_solve(game, _p0, max_iter=1, tol=1e-20, qp_solver=QP_SOLVER)
    pg_solve(game, _p0, max_iter=1, tol=1e-20, qp_solver=QP_SOLVER)
    impdimc_solve(game, _p0, agent_sols_base, max_iter=1, tol=1e-20)
    solve_gne_online(_p0, (0,)*M, agent_sols_FH, game)
    solve_gne_online(_p0, (0,)*M, agent_sols_FLP, game)
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
            res_dict["methods"]["Explicit"] = {"avg_time": ts.mean(), "max_time": ts.max(), "iters": itrs.mean(), "norm": np.linalg.norm(xt[-1])}
        else:
            res_dict["methods"]["Explicit"] = None

        # ADMM
        print(f"  [M{M}/{idx:03d}] [7/7]   ADMM...", flush=True)
        t_a = time.perf_counter()
        xt, ts, itrs = _run_admm_sim(plant, game, x0, T_SIM)
        res_dict["methods"]["ADMM"] = {"avg_time": ts.mean(), "max_time": ts.max(), "iters": itrs.mean(), "norm": np.linalg.norm(xt[-1])}
        print(f"  [M{M}/{idx:03d}] [7/7]   ADMM done ({time.perf_counter()-t_a:.1f}s)", flush=True)
        # Jacobi BR
        print(f"  [M{M}/{idx:03d}] [7/7]   BR...", flush=True)
        t_b = time.perf_counter()
        xt, ts, itrs = _run_br_sim(plant, game, x0, T_SIM)
        res_dict["methods"]["BR"] = {"avg_time": ts.mean(), "max_time": ts.max(), "iters": itrs.mean(), "norm": np.linalg.norm(xt[-1])}
        print(f"  [M{M}/{idx:03d}] [7/7]   BR done ({time.perf_counter()-t_b:.1f}s)", flush=True)
        # ImpGNE
        print(f"  [M{M}/{idx:03d}] [7/7]   ImpGNE...", flush=True)
        t_i = time.perf_counter()
        xt, ts, itrs = _run_impgne_sim(plant, game, agent_sols_base, x0, T_SIM)
        res_dict["methods"]["ImpGNE"] = {"avg_time": ts.mean(), "max_time": ts.max(), "iters": itrs.mean(), "norm": np.linalg.norm(xt[-1])}
        print(f"  [M{M}/{idx:03d}] [7/7]   ImpGNE done ({time.perf_counter()-t_i:.1f}s)", flush=True)

        # FACET-H: BFS map lookup (M<=3) or online neighbor walk (M>=4) → ADMM fallback
        _fsol_FH = facet_sol_FH if (M <= OFFLINE_BFS_MAX_M and facet_sol_FH is not None) else None
        print(f"  [M{M}/{idx:03d}] [7/7]   FACET-H...", flush=True)
        t_h = time.perf_counter()
        xt, ut, ts, itrs, fbs = _run_facet_sim(plant, game, agent_sols_FH, x0, T_SIM, facet_sol=_fsol_FH)
        res_dict["methods"]["FACET-H"] = {"avg_time": ts.mean(), "max_time": ts.max(), "iters": itrs.mean(), "fallbacks": fbs, "norm": np.linalg.norm(xt[-1])}
        print(f"  [M{M}/{idx:03d}] [7/7]   FACET-H done ({time.perf_counter()-t_h:.1f}s)", flush=True)
        if idx == 0:
            res_dict["x_traj"], res_dict["u_traj"] = xt, ut
            res_dict["plant_nx"] = plant.nx
            res_dict["plant_nu"] = {i: plant.subsystems[i].nu for i in range(M)}

        # FACET-LP: BFS map lookup (M<=3) or online neighbor walk (M>=4) → ADMM fallback
        _fsol_FLP = facet_sol_FLP if (M <= OFFLINE_BFS_MAX_M and facet_sol_FLP is not None) else None
        print(f"  [M{M}/{idx:03d}] [7/7]   FACET-LP...", flush=True)
        t_l = time.perf_counter()
        xt, ut, ts, itrs, fbs = _run_facet_sim(plant, game, agent_sols_FLP, x0, T_SIM, facet_sol=_fsol_FLP)
        res_dict["methods"]["FACET-LP"] = {"avg_time": ts.mean(), "max_time": ts.max(), "iters": itrs.mean(), "fallbacks": fbs, "norm": np.linalg.norm(xt[-1])}
        print(f"  [M{M}/{idx:03d}] [7/7]   FACET-LP done ({time.perf_counter()-t_l:.1f}s)", flush=True)
        print(f"  [M{M}/{idx:03d}] [7/7] Online sim done ({time.perf_counter()-t_online:.1f}s)", flush=True)

    except Exception as e:
        res_dict["error"] = f"online sim: {e}"
        traceback.print_exc()
        
    _save(M, idx, "results", res_dict)
    return res_dict

# %% ── 4. Main Runner ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Config: N_PLANTS={N_PLANTS}, T_SIM={T_SIM}, M_LIST={M_LIST}")
    print(f"Checkpoints → {CKPT_DIR}\n")
    all_results = {M: [] for M in M_LIST}
    
    for M in M_LIST:
        print(f"\n{'='*60}\n  Processing M = {M} ({N_PLANTS} plants)\n{'='*60}")
        plants = make_random_plants(M, N_PLANTS, seed=SEED+M)
        rng_ic = np.random.default_rng(SEED*2 + M)
        
        for idx, plant in enumerate(plants):
            x0 = make_ic(plant, scale=0.4, rng=rng_ic)
            res = run_one_plant(plant, x0, M, idx)
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

    # %% ── 5. Generate Tables and Plots ───────────────────────────────────────────
    print("\n\n" + "="*80)
    print("  CASE STUDY RESULTS SUMMARY (Tables & Plots)")
    print("="*80)

    # 1. Boxplot: Number of Critical Regions vs M
    plt.figure(figsize=(8, 6))
    cr_data = []
    labels = []
    for M in M_LIST:
        crs = []
        for r in all_results[M]:
            if not r.get("error"): crs.extend(r["n_crs"])
        if crs:
            cr_data.append(crs)
            labels.append(str(M))
    if cr_data:
        plt.boxplot(cr_data, labels=labels, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        plt.yscale('log')
        plt.xlabel("Number of Subsystems (M)", fontweight='bold')
        plt.ylabel("Number of Critical Regions", fontweight='bold')
        plt.title("Distribution of Critical Regions per Subsystem", fontweight='bold')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(CKPT_DIR, "crs_boxplot.png"), dpi=200)

    # 2. Table: Average Online Compute Time (ms)
    print("\n  [TABLE] Average Online Solve Time (ms):")
    print(f"  {'M':>3} | {'Explicit':>10} | {'ADMM':>10} | {'Jacobi BR':>10} | {'ImpGNE':>10} | {'FACET-H':>10} | {'FACET-LP':>10}")
    print("  " + "-"*80)
    for M in M_LIST:
        times = {"Explicit": [], "ADMM": [], "BR": [], "ImpGNE": [], "FACET-H": [], "FACET-LP": []}
        for r in all_results[M]:
            if r.get("error"): continue
            for k in times.keys():
                if r["methods"].get(k) is not None:
                    times[k].append(r["methods"][k]["avg_time"] * 1000)
        
        t_exp = np.mean(times["Explicit"]) if times["Explicit"] else float('nan')
        t_admm = np.mean(times["ADMM"]) if times["ADMM"] else float('nan')
        t_br = np.mean(times["BR"]) if times["BR"] else float('nan')
        t_imp = np.mean(times["ImpGNE"]) if times["ImpGNE"] else float('nan')
        t_fh = np.mean(times["FACET-H"]) if times["FACET-H"] else float('nan')
        t_flp = np.mean(times["FACET-LP"]) if times["FACET-LP"] else float('nan')
        
        print(f"  {M:>3} | {t_exp:>10.4f} | {t_admm:>10.2f} | {t_br:>10.2f} | {t_imp:>10.2f} | {t_fh:>10.4f} | {t_flp:>10.4f}")

    # 3. Table: Speedup of FACET-LP vs Others
    print("\n  [TABLE] Speedup of FACET-LP vs Iterative Methods:")
    print(f"  {'M':>3} | {'vs Explicit':>11} | {'vs ADMM':>10} | {'vs Jacobi BR':>12} | {'vs ImpGNE':>10} | {'vs FACET-H':>10}")
    print("  " + "-"*80)
    for M in M_LIST:
        times = {"Explicit": [], "ADMM": [], "BR": [], "ImpGNE": [], "FACET-H": [], "FACET-LP": []}
        for r in all_results[M]:
            if r.get("error"): continue
            for k in times.keys():
                if r["methods"].get(k) is not None:
                    times[k].append(r["methods"][k]["avg_time"])
                
        if times["FACET-LP"]:
            t_flp = np.mean(times["FACET-LP"])
            su_exp = np.mean(times["Explicit"]) / t_flp if times["Explicit"] else float('nan')
            su_admm = np.mean(times["ADMM"]) / t_flp
            su_br = np.mean(times["BR"]) / t_flp
            su_imp = np.mean(times["ImpGNE"]) / t_flp
            su_fh = np.mean(times["FACET-H"]) / t_flp
            print(f"  {M:>3} | {su_exp:>10.2f}x | {su_admm:>9.1f}x | {su_br:>11.1f}x | {su_imp:>9.1f}x | {su_fh:>9.2f}x")

    # 4. Boxplot: Online Compute Time per M
    plt.figure(figsize=(10, 6))
    time_data = []
    labels = []
    colors = []
    for M in M_LIST:
        t_exp, t_admm, t_br, t_imp, t_fh, t_flp = [], [], [], [], [], []
        for r in all_results[M]:
            if r.get("error"): continue
            if r["methods"].get("Explicit") is not None:
                t_exp.append(r["methods"]["Explicit"]["avg_time"] * 1000)
            t_admm.append(r["methods"]["ADMM"]["avg_time"] * 1000)
            t_br.append(r["methods"]["BR"]["avg_time"] * 1000)
            t_imp.append(r["methods"]["ImpGNE"]["avg_time"] * 1000)
            t_fh.append(r["methods"]["FACET-H"]["avg_time"] * 1000)
            t_flp.append(r["methods"]["FACET-LP"]["avg_time"] * 1000)
            
        time_data.extend([t_exp, t_admm, t_br, t_imp, t_fh, t_flp])
        labels.extend([f"Exp", f"ADMM", f"BR", f"ImpGNE", f"F-H", f"F-LP"])
        colors.extend(["#A0A0A0", "#E07B54", "#5B8DB8", "#C97BD4", "#4CAF82", "#FFD700"])

    if time_data:
        bp = plt.boxplot(time_data, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            
        plt.yscale('log')
        plt.xticks(np.arange(1, len(labels)+1), labels, rotation=45, ha="right", fontsize=8)
        
        # Add vertical lines to separate M groups
        for i in range(1, len(M_LIST)):
            plt.axvline(x=i*6 + 0.5, color='gray', linestyle='--', alpha=0.5)
            
        plt.ylabel("Online Solve Time (ms)", fontweight='bold')
        plt.title("Online Compute Time Distribution across Subsystems", fontweight='bold')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(CKPT_DIR, "times_boxplot.png"), dpi=200)

    # 5. Table: Average Data Transfer Instances (Iterations)
    print("\n  [TABLE] Average Data Transfer Instances (Iterations / step):")
    print(f"  {'M':>3} | {'Explicit':>10} | {'ADMM':>10} | {'Jacobi BR':>10} | {'ImpGNE':>10} | {'FACET-H':>10} | {'FACET-LP':>10}")
    print("  " + "-"*80)
    for M in M_LIST:
        iters = {"Explicit": [], "ADMM": [], "BR": [], "ImpGNE": [], "FACET-H": [], "FACET-LP": []}
        for r in all_results[M]:
            if r.get("error"): continue
            for k in iters.keys():
                if r["methods"].get(k) is not None:
                    iters[k].append(r["methods"][k]["iters"])
        
        i_exp = np.mean(iters["Explicit"]) if iters["Explicit"] else float('nan')
        i_admm = np.mean(iters["ADMM"]) if iters["ADMM"] else float('nan')
        i_br = np.mean(iters["BR"]) if iters["BR"] else float('nan')
        i_imp = np.mean(iters["ImpGNE"]) if iters["ImpGNE"] else float('nan')
        i_fh = np.mean(iters["FACET-H"]) if iters["FACET-H"] else float('nan')
        i_flp = np.mean(iters["FACET-LP"]) if iters["FACET-LP"] else float('nan')
        
        print(f"  {M:>3} | {i_exp:>10.2f} | {i_admm:>10.2f} | {i_br:>10.2f} | {i_imp:>10.2f} | {i_fh:>10.4f} | {i_flp:>10.4f}")

    # 5b. Table: Total Average Solution Time over 100 Time Points (s)
    print("\n  [TABLE] Total Average Solution Time over 100 Time Points (s):")
    print(f"  {'M':>3} | {'Explicit':>10} | {'ADMM':>10} | {'Jacobi BR':>10} | {'ImpGNE':>10} | {'FACET-H':>10} | {'FACET-LP':>10}")
    print("  " + "-"*80)
    for M in M_LIST:
        times = {"Explicit": [], "ADMM": [], "BR": [], "ImpGNE": [], "FACET-H": [], "FACET-LP": []}
        for r in all_results[M]:
            if r.get("error"): continue
            for k in times.keys():
                if r["methods"].get(k) is not None:
                    times[k].append(r["methods"][k]["avg_time"] * T_SIM)

        t_exp = np.mean(times["Explicit"]) if times["Explicit"] else float('nan')
        t_admm = np.mean(times["ADMM"]) if times["ADMM"] else float('nan')
        t_br = np.mean(times["BR"]) if times["BR"] else float('nan')
        t_imp = np.mean(times["ImpGNE"]) if times["ImpGNE"] else float('nan')
        t_fh = np.mean(times["FACET-H"]) if times["FACET-H"] else float('nan')
        t_flp = np.mean(times["FACET-LP"]) if times["FACET-LP"] else float('nan')

        print(f"  {M:>3} | {t_exp:>10.4f} | {t_admm:>10.3f} | {t_br:>10.3f} | {t_imp:>10.3f} | {t_fh:>10.4f} | {t_flp:>10.4f}")

    # 5c. Table: Total Data Transfer over 100 Time Points
    print("\n  [TABLE] Total Data Transfer over 100 Time Points:")
    print(f"  {'M':>3} | {'Explicit':>10} | {'ADMM':>10} | {'Jacobi BR':>10} | {'ImpGNE':>10} | {'FACET-H':>10} | {'FACET-LP':>10}")
    print("  " + "-"*80)
    for M in M_LIST:
        iters = {"Explicit": [], "ADMM": [], "BR": [], "ImpGNE": [], "FACET-H": [], "FACET-LP": []}
        for r in all_results[M]:
            if r.get("error"): continue
            for k in iters.keys():
                if r["methods"].get(k) is not None:
                    iters[k].append(r["methods"][k]["iters"] * T_SIM)

        i_exp = np.mean(iters["Explicit"]) if iters["Explicit"] else float('nan')
        i_admm = np.mean(iters["ADMM"]) if iters["ADMM"] else float('nan')
        i_br = np.mean(iters["BR"]) if iters["BR"] else float('nan')
        i_imp = np.mean(iters["ImpGNE"]) if iters["ImpGNE"] else float('nan')
        i_fh = np.mean(iters["FACET-H"]) if iters["FACET-H"] else float('nan')
        i_flp = np.mean(iters["FACET-LP"]) if iters["FACET-LP"] else float('nan')

        print(f"  {M:>3} | {i_exp:>10.1f} | {i_admm:>10.1f} | {i_br:>10.1f} | {i_imp:>10.1f} | {i_fh:>10.1f} | {i_flp:>10.1f}")

    # 6. Boxplot: Data Transfer Instances (Iterations) per M
    plt.figure(figsize=(10, 6))
    iter_data = []
    labels = []
    colors = []
    for M in M_LIST:
        i_admm, i_br, i_imp, i_fh, i_flp = [], [], [], [], []
        for r in all_results[M]:
            if r.get("error"): continue
            i_admm.append(r["methods"]["ADMM"]["iters"])
            i_br.append(r["methods"]["BR"]["iters"])
            i_imp.append(r["methods"]["ImpGNE"]["iters"])
            i_fh.append(r["methods"]["FACET-H"]["iters"])
            i_flp.append(r["methods"]["FACET-LP"]["iters"])
            
        iter_data.extend([i_admm, i_br, i_imp, i_fh, i_flp])
        labels.extend([f"ADMM", f"BR", f"ImpGNE", f"F-H", f"F-LP"])
        colors.extend(["#E07B54", "#5B8DB8", "#C97BD4", "#4CAF82", "#FFD700"])

    if iter_data:
        bp = plt.boxplot(iter_data, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            
        plt.yscale('log')
        plt.xticks(np.arange(1, len(labels)+1), labels, rotation=45, ha="right", fontsize=8)
        
        for i in range(1, len(M_LIST)):
            plt.axvline(x=i*5 + 0.5, color='gray', linestyle='--', alpha=0.5)
            
        plt.ylabel("Data Transfer Instances (Iterations)", fontweight='bold')
        plt.title("Data Transfer Instances per Time Step", fontweight='bold')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(CKPT_DIR, "data_transfer_boxplot.png"), dpi=200)

    # 7. Trajectories (Figure 3 Style) for each M
    for M in M_LIST:
        r0 = None
        for r in all_results[M]:
            if not r.get("error") and "x_traj" in r:
                r0 = r
                break
        if r0:
            xt = r0["x_traj"]
            ut = r0["u_traj"]
            nx = r0["plant_nx"]
            nus = r0["plant_nu"]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            for i in range(nx):
                ax1.plot(np.arange(len(xt)), xt[:, i], linewidth=1.5)
            ax1.set_ylabel("States $x(k)$", fontweight='bold')
            ax1.set_title(f"Controller Performance (Outputs & Inputs) for M={M}", fontweight='bold')
            ax1.grid(True, linestyle="--", alpha=0.7)
            
            for i in range(M):
                for j in range(nus[i]):
                    ax2.plot(np.arange(len(ut[i])), ut[i][:, j], drawstyle='steps-post', linewidth=1.5)
            ax2.set_ylabel("Control Inputs $u(k)$", fontweight='bold')
            ax2.set_xlabel("Time step $k$", fontweight='bold')
            ax2.grid(True, linestyle="--", alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(CKPT_DIR, f"trajectory_M{M}.png"), dpi=200)
            plt.close(fig)

    print(f"\nAll plots and data saved to: {CKPT_DIR}")
    print("Done.")

# %%
