"""
test_case_study.py
==================
GNE case study — mirrors dimpc_ppopt/tests/test_case_study.py structure.

Compares three methods across M ∈ M_LIST agents on N_PLANTS random plants:

  ADMM         — iterative ADMM online solver (baseline; like DiMPC)
  Explicit-GNE — offline mpQP + exhaustive combination search (like IF-mpDiMPC)
                 skipped for M >= EXPLICIT_MAX_M (combinatorial explosion)
  FACET-GNE    — offline mpQP + BFS combination traversal (our contribution)
                 falls back to ADMM for that timestep when p is outside all CRs

Metrics (paper-style):
  Offline time  : mpQP solve + facet detection
  Online time   : per step solve time (ms)
  FACET speedup : n_combos_checked / n_combos_total
  ||x_T||       : final state norm (all methods → 0 for stable plants)
  ADMM iters    : iterations to convergence per step

Run:
    cd mpgne_ppopt
    python tests/test_case_study.py

Checkpoints are saved after every plant so interrupted runs resume.

USER CONFIG — edit the block below.
"""

# %% ── 0. Config ──────────────────────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pickle
import traceback
import numpy as np
from ppopt.mp_solvers.solve_mpqp import mpqp_algorithm

# ═══════════════════════════════════════════════════════════════════════════════
N_PLANTS      = 5          # paper uses 100  →  increase for full reproduction
T_SIM         = 100    # closed-loop simulation steps
M_LIST        = [3]  # number of agents to test
EXPLICIT_MAX_M = 4         # skip Explicit-GNE (exhaustive) for M >= this value
OFFLINE_BFS_MAX_M = 4      # skip Offline BFS map-building for M >= this value
FORCE_OFFLINE_BFS = False  # Set to True to force offline BFS for M >= 4 (Backup option)

ALGO        = mpqp_algorithm.combinatorial_parallel   # PPOPT mpQP algorithm
ADMM_RHO    = 0.5        # ADMM penalty parameter (reduced for slower/accurate convergence)
ADMM_ITERS  = 1000       # max ADMM iterations per step
ADMM_TOL    = 1e-8       # ADMM convergence tolerance (tightened for paper benchmark)
# L_MAX coupling constraint removed — state bounds now couple agents
FACET_METHOD = "hyperplane"   # "hyperplane" (fast) or "lp" (rigorous)

SEED        = 2025
CKPT_DIR    = os.path.join(os.path.dirname(__file__), "checkpoints_gne")
# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs(CKPT_DIR, exist_ok=True)

from mpgne.plant_gen    import make_random_plants, make_ic
from mpgne.mpc_builder  import make_gne_game_from_plant, default_local_weights
from mpgne.mp_solver    import solve_all_agents_mp
from mpgne.gne_combiner import build_gne_solution
from mpgne.admm_solver      import admm_solve
from mpgne.facet_gne        import find_all_agent_cr_neighbors, build_gne_solution_facet

print(f"Config: N_PLANTS={N_PLANTS}, T_SIM={T_SIM}, M_LIST={M_LIST}")
print(f"Algorithm: {ALGO}  |  ADMM_RHO={ADMM_RHO}")
print(f"Checkpoints → {CKPT_DIR}\n")


# %% ── 1. Checkpoint helpers ──────────────────────────────────────────────────

def _ckpt_path(M, idx):
    return os.path.join(CKPT_DIR, f"M{M}_plant{idx:03d}.pkl")

def _save(M, idx, data):
    with open(_ckpt_path(M, idx), "wb") as f:
        pickle.dump(data, f)

def _load(M, idx):
    p = _ckpt_path(M, idx)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# %% ── 2. Closed-loop simulation helpers ─────────────────────────────────────

def _run_explicit_sim(plant, game, gne_sol, x0, T):
    """Closed-loop simulation using the explicit GNE map (lookup only)."""
    nx  = plant.nx
    M   = plant.M
    x_traj = np.zeros((T + 1, nx))
    x_traj[0] = x0.copy()
    u_traj  = {i: np.zeros((T, plant.subsystems[i].nu)) for i in range(M)}
    times   = np.zeros(T)
    found   = np.zeros(T, dtype=bool)

    for k in range(T):
        p = x_traj[k]
        t0 = time.perf_counter()
        cr_idx = gne_sol.locate(p)
        times[k] = time.perf_counter() - t0

        if cr_idx is None:
            # p outside all CRs: hold previous or zero
            u_k = {i: np.zeros(plant.subsystems[i].nu) for i in range(M)}
        else:
            found[k] = True
            U_star = gne_sol[cr_idx].evaluate(p)    # stacked input sequences
            u_k = {}
            offset = 0
            for i in range(M):
                nu_i = plant.subsystems[i].nu
                Np   = plant.Np
                u_k[i] = U_star[offset:offset + nu_i]  # first action only
                offset += Np * nu_i

        for i in range(M):
            u_traj[i][k] = u_k[i]
        x_traj[k + 1] = plant.step(x_traj[k], u_k)

    return x_traj, u_traj, times, found


def _run_admm_sim(plant, game, x0, T):
    """Closed-loop simulation using ADMM at each step."""
    nx  = plant.nx
    M   = plant.M
    x_traj  = np.zeros((T + 1, nx))
    x_traj[0] = x0.copy()
    u_traj  = {i: np.zeros((T, plant.subsystems[i].nu)) for i in range(M)}
    times   = np.zeros(T)
    iters   = np.zeros(T, dtype=int)
    conv    = np.zeros(T, dtype=bool)
    x_warm  = None

    for k in range(T):
        p  = x_traj[k]
        t0 = time.perf_counter()
        res = admm_solve(game, p, rho=ADMM_RHO, max_iter=ADMM_ITERS,
                         tol=ADMM_TOL, verbose=False, x_init=x_warm)
        times[k] = time.perf_counter() - t0
        iters[k] = res.n_iter
        conv[k]  = res.converged
        x_warm   = res.x_sol   # warm-start next step

        u_k = {}
        for i in range(M):
            nu_i = plant.subsystems[i].nu
            Np   = plant.Np
            u_k[i] = res.x_sol[i][:nu_i]   # first action of U_i*

        for i in range(M):
            u_traj[i][k] = u_k[i]
        x_traj[k + 1] = plant.step(x_traj[k], u_k)

    return x_traj, u_traj, times, iters, conv


def _run_facet_fallback_sim(plant, game, gne_sol_facet, x0, T):
    """
    FACET-GNE online sim with ADMM fallback.

    For each step:
      - Try explicit GNE lookup in gne_sol_facet (O(n_GNE_CR × n_ineq)).
      - If p is outside all GNE CRs, run admm_solve for that step only.
    """
    nx   = plant.nx
    M    = plant.M
    x_traj  = np.zeros((T + 1, nx))
    x_traj[0] = x0.copy()
    u_traj  = {i: np.zeros((T, plant.subsystems[i].nu)) for i in range(M)}
    times      = np.zeros(T)
    found      = np.zeros(T, dtype=bool)
    n_fallback = 0

    for k in range(T):
        p  = x_traj[k]
        t0 = time.perf_counter()
        cr_idx = gne_sol_facet.locate(p)

        if cr_idx is not None:
            found[k] = True
            U_star = gne_sol_facet[cr_idx].evaluate(p)
            u_k = {}
            offset = 0
            for i in range(M):
                nu_i = plant.subsystems[i].nu
                Np   = plant.Np
                u_k[i] = U_star[offset:offset + nu_i]
                offset += Np * nu_i
        else:
            # Fallback: ADMM for this step only
            n_fallback += 1
            res = admm_solve(
                game, p,
                rho=ADMM_RHO, max_iter=ADMM_ITERS,
                tol=ADMM_TOL, verbose=False,
            )
            u_k = {}
            for i in range(M):
                nu_i = plant.subsystems[i].nu
                u_k[i] = res.x_sol[i][:nu_i]

        times[k] = time.perf_counter() - t0
        for i in range(M):
            u_traj[i][k] = u_k[i]
        x_traj[k + 1] = plant.step(x_traj[k], u_k)

    return x_traj, u_traj, times, found, n_fallback


def _run_online_facet_gne_sim(plant, game, agent_sols, x0, T):
    """
    Online FACET-GNE search for combination (like DiMPC online).
    """
    from mpgne.facet_gne import solve_gne_online
    nx   = plant.nx
    M    = plant.M
    x_traj  = np.zeros((T + 1, nx))
    x_traj[0] = x0.copy()
    u_traj  = {i: np.zeros((T, plant.subsystems[i].nu)) for i in range(M)}
    times      = np.zeros(T)
    found      = np.zeros(T, dtype=bool)
    combos_checked_arr = np.zeros(T, dtype=int)
    n_fallback = 0
    prev_combo = None

    for k in range(T):
        p  = x_traj[k]
        t0 = time.perf_counter()
        if prev_combo is None:
            # Cold start: Run ADMM to get a warm start U
            from mpgne.admm_solver import admm_solve
            res_warm = admm_solve(game, p, max_iter=500, tol=1e-4)
            U_warm_stacked = res_warm.x_stacked
            
            # Find the local region for each agent
            warm_combo = []
            offset = 0
            U_warm_dict = {}
            for i in range(M):
                nu_i = plant.subsystems[i].nu
                Np_nu = plant.Np * nu_i
                U_warm_dict[i] = U_warm_stacked[offset:offset+Np_nu]
                offset += Np_nu
                
            for i in range(M):
                theta_i = [p]
                for j in range(M):
                    if j != i: theta_i.append(U_warm_dict[j])
                theta_i_vec = np.concatenate(theta_i)
                
                v_i = 0
                min_violation = float('inf')
                for v, cr in enumerate(agent_sols[i].regions):
                    violation = np.max(cr.E @ theta_i_vec - cr.f)
                    if violation < min_violation:
                        min_violation = violation
                        v_i = v
                warm_combo.append(v_i)
            prev_combo = tuple(warm_combo)
            
        combo, U_star, checked = solve_gne_online(p, prev_combo, agent_sols, game)
        combos_checked_arr[k] = checked

        if combo is not None:
            found[k] = True
            prev_combo = combo
            u_k = {}
            offset = 0
            for i in range(M):
                nu_i = plant.subsystems[i].nu
                Np   = plant.Np
                u_k[i] = U_star[offset:offset + nu_i]
                offset += Np * nu_i
        else:
            n_fallback += 1
            from mpgne.admm_solver import admm_solve
            res = admm_solve(
                game, p,
                rho=ADMM_RHO, max_iter=ADMM_ITERS,
                tol=ADMM_TOL, verbose=False,
            )
            u_k = {}
            for i in range(M):
                nu_i = plant.subsystems[i].nu
                u_k[i] = res.x_sol[i][:nu_i]
            prev_combo = None

        times[k] = time.perf_counter() - t0
        for i in range(M):
            u_traj[i][k] = u_k[i]
        x_traj[k + 1] = plant.step(x_traj[k], u_k)

    return x_traj, u_traj, times, found, n_fallback, combos_checked_arr


def _run_centralized_sim(plant, game, x0, T):
    """
    Centralized baseline simulation using scipy SLSQP on the joint KKT residual.
    """
    from mpgne.centralized_solver import solve_gne_centralized
    nx   = plant.nx
    M    = plant.M
    x_traj  = np.zeros((T + 1, nx))
    x_traj[0] = x0.copy()
    u_traj  = {i: np.zeros((T, plant.subsystems[i].nu)) for i in range(M)}
    times      = np.zeros(T)
    
    U_warm = None
    for k in range(T):
        p = x_traj[k]
        res = solve_gne_centralized(game, p, x_init=U_warm)
        times[k] = res.solve_time
        U_warm = res.x_sol.copy()
        
        u_k = {}
        offset = 0
        for i in range(M):
            nu_i = plant.subsystems[i].nu
            u_k[i] = res.x_sol[offset:offset + nu_i]
            u_traj[i][k] = u_k[i]
            offset += plant.Np * nu_i
            
        x_traj[k + 1] = plant.step(x_traj[k], u_k)
        
    return x_traj, u_traj, times


# %% ── 3. Run one plant ────────────────────────────────────────────────────────────────────

def run_one_plant(plant, x0, M, plant_idx):
    result = {
        "M": M, "plant_idx": plant_idx,
        "nx": plant.nx, "nu": plant.nu, "Np": plant.Np,
        "n_crs": None, "n_combos_total": None, "n_combos_facet": None,
        "offline_explicit_s": None, "offline_facet_s": None,
        "methods": {}, "error": None,
    }

    run_explicit = (M < EXPLICIT_MAX_M)

    # ── build GNE game from plant ────────────────────────────────────────────────────
    try:
        Q_list, R_list, P_list = default_local_weights(plant)
        game = make_gne_game_from_plant(plant,
                                        Q_list=Q_list, R_list=R_list, P_list=P_list)
    except Exception as e:
        result["error"] = f"game build: {e}"
        return result

    # ── offline: mpQP solve for all agents ───────────────────────────────────
    try:
        t0 = time.perf_counter()
        print(f"    [offline] mpQP solve ({ALGO})...", flush=True)
        agent_sols = solve_all_agents_mp(game, algorithm=ALGO, verbose=False)
        crs = [s.n_cr for s in agent_sols]
        n_combos = 1
        for c in crs:
            n_combos *= c

        result["n_crs"] = crs
        result["n_combos_total"] = n_combos
        print(f"           CRs/agent={crs}  total_combos={n_combos}")

        # ── Explicit-GNE: exhaustive (skipped for M >= EXPLICIT_MAX_M) ────────
        gne_sol_exh = None
        if run_explicit:
            print(f"    [offline] exhaustive combination search...", flush=True)
            gne_sol_exh = build_gne_solution(game, agent_sols, verbose=False)
            result["offline_explicit_s"] = time.perf_counter() - t0
            print(f"           {gne_sol_exh.summary()}  "
                  f"({result['offline_explicit_s']:.1f}s)")
        else:
            print(f"    [offline] Explicit-GNE skipped "
                  f"(M={M} >= EXPLICIT_MAX_M={EXPLICIT_MAX_M})")

        # ── FACET-GNE: facet detection + BFS ─────────────────────────────────
        print(f"    [offline] facet detection ({FACET_METHOD}) + BFS...", flush=True)
        t1 = time.perf_counter()
        find_all_agent_cr_neighbors(agent_sols, method=FACET_METHOD, verbose=False)
        
        run_offline_bfs = (M < OFFLINE_BFS_MAX_M) or FORCE_OFFLINE_BFS
        facet_res = None
        if run_offline_bfs:
            print(f"    [offline] building full explicit map via BFS...", flush=True)
            facet_res = build_gne_solution_facet(game, agent_sols, verbose=False)
            result["offline_facet_s"] = time.perf_counter() - t1
            result["n_combos_facet"]  = facet_res.n_combos_checked
            print(f"           BFS: {facet_res.n_combos_checked}/{n_combos} combos "
                  f"({facet_res.reduction_ratio*100:.0f}%)  "
                  f"({result['offline_facet_s']:.1f}s)")
        else:
            print(f"    [offline] global explicit map skipped (M={M} >= {OFFLINE_BFS_MAX_M}). Will solve online.")
            result["offline_facet_s"] = time.perf_counter() - t1
            result["n_combos_facet"]  = 0

    except Exception as e:
        result["error"] = f"offline: {e}"
        traceback.print_exc()
        return result

    # ── online: Explicit-GNE sim (skipped for M >= EXPLICIT_MAX_M) ───────────
    if run_explicit:
        try:
            x_traj, _, times, found = _run_explicit_sim(
                plant, game, gne_sol_exh, x0, T_SIM)
            result["methods"]["Explicit-GNE"] = {
                "avg_time_ms":  float(times.mean()) * 1000,
                "max_time_ms":  float(times.max()) * 1000,
                "final_norm":   float(np.linalg.norm(x_traj[-1])),
                "found_pct":    float(found.mean()) * 100,
            }
        except Exception as e:
            result["methods"]["Explicit-GNE"] = {"error": str(e)}
            traceback.print_exc()

    # ── online: FACET-GNE sim with ADMM fallback ──────────────────────────
    try:
        if run_offline_bfs:
            gne_sol_facet = facet_res.gne_sol
            x_traj, _, times, found, n_fb = _run_facet_fallback_sim(
                plant, game, gne_sol_facet, x0, T_SIM)
            result["methods"]["FACET-GNE"] = {
                "avg_time_ms":  float(times.mean()) * 1000,
                "max_time_ms":  float(times.max()) * 1000,
                "final_norm":   float(np.linalg.norm(x_traj[-1])),
                "found_pct":    float(found.mean()) * 100,
                "n_fallback":   n_fb,
            }
        else:
            x_traj, _, times, found, n_fb, combos_checked = _run_online_facet_gne_sim(
                plant, game, agent_sols, x0, T_SIM)
            result["methods"]["FACET-GNE"] = {
                "avg_time_ms":  float(times.mean()) * 1000,
                "max_time_ms":  float(times.max()) * 1000,
                "final_norm":   float(np.linalg.norm(x_traj[-1])),
                "found_pct":    float(found.mean()) * 100,
                "n_fallback":   n_fb,
                "avg_combos_checked": float(combos_checked.mean()),
            }
    except Exception as e:
        result["methods"]["FACET-GNE"] = {"error": str(e)}
        traceback.print_exc()

    # ── online: ADMM sim ──────────────────────────────────────────────────────
    try:
        x_traj_admm, _, times, iters, conv = _run_admm_sim(plant, game, x0, T_SIM)
        result["methods"]["ADMM"] = {
            "avg_time_ms":  float(times.mean()) * 1000,
            "max_time_ms":  float(times.max()) * 1000,
            "final_norm":   float(np.linalg.norm(x_traj_admm[-1])),
            "avg_iters":    float(iters.mean()),
            "max_iters":    int(iters.max()),
            "conv_pct":     float(conv.mean()) * 100,
        }
    except Exception as e:
        result["methods"]["ADMM"] = {"error": str(e)}
        traceback.print_exc()

    # ── online: Centralized sim (Baseline) ──────────────────────────────────
    try:
        x_traj_cen, _, times = _run_centralized_sim(plant, game, x0, T_SIM)
        result["methods"]["Centralized"] = {
            "avg_time_ms":  float(times.mean()) * 1000,
            "max_time_ms":  float(times.max()) * 1000,
            "final_norm":   float(np.linalg.norm(x_traj_cen[-1])),
        }
    except Exception as e:
        result["methods"]["Centralized"] = {"error": str(e)}
        traceback.print_exc()

    return result



# %% ── 4. Case study for one M ────────────────────────────────────────────────

def run_case_study_M(M: int) -> list[dict]:
    print(f"\n{'='*65}")
    print(f"  M = {M} agents  |  N = {N_PLANTS} plants  |  T = {T_SIM}")
    print(f"{'='*65}")

    plants = make_random_plants(M, N_PLANTS, seed=SEED)
    rng_ic = np.random.default_rng(SEED + 1000 * M)
    all_results = []

    for idx, plant in enumerate(plants):
        ckpt = _load(M, idx)
        if ckpt is not None:
            print(f"\n  Plant {idx+1}/{N_PLANTS} — loaded from checkpoint")
            _print_row(ckpt)
            all_results.append(ckpt)
            continue

        x0 = make_ic(plant, scale=0.1, rng=rng_ic)
        print(f"\n  ── Plant {idx+1}/{N_PLANTS}  "
              f"(nx={plant.nx}, nu={plant.nu}, Np={plant.Np}) ──")

        res = run_one_plant(plant, x0, M, idx)
        _save(M, idx, res)
        _print_row(res)
        all_results.append(res)

    return all_results


def _print_row(res: dict):
    if res.get("error"):
        print(f"    ERROR: {res['error']}")
        return

    exh_s = res.get("offline_explicit_s")
    exh_str = f"{exh_s:.1f}s" if exh_s is not None else "skipped"
    print(f"    CRs={res['n_crs']}  "
          f"combos_total={res['n_combos_total']}  "
          f"combos_facet={res['n_combos_facet']}  "
          f"offline_exh={exh_str}  "
          f"offline_facet={res['offline_facet_s']:.1f}s")
    print(f"    {'Method':<14} {'AvgMs':>8} {'||xT||':>9} {'Extra':>30}")
    print(f"    {'-'*65}")
    order = ["Explicit-GNE", "FACET-GNE", "ADMM", "Centralized"]
    for name in order:
        m = res["methods"].get(name)
        if m is None:
            continue   # Explicit-GNE skipped for large M
        if "error" in m:
            print(f"    {name:<14}  ERROR: {m['error']}")
        elif name == "ADMM":
            print(f"    {name:<14} {m['avg_time_ms']:>8.3f} "
                  f"{m['final_norm']:>9.5f} "
                  f"  iters={m['avg_iters']:.1f} conv={m['conv_pct']:.0f}%")
        elif name == "FACET-GNE":
            reduction_pct = (res['n_combos_facet']/res['n_combos_total']*100) if res['n_combos_total'] else 0
            print(f"    {name:<14} {m['avg_time_ms']:>8.3f} "
                  f"{m['final_norm']:>9.5f} "
                  f"  reduction={reduction_pct:.0f}%  "
                  f"found={m['found_pct']:.0f}%  fallback={m.get('n_fallback',0)}")
        elif name == "Centralized":
            print(f"    {name:<14} {m['avg_time_ms']:>8.3f} "
                  f"{m['final_norm']:>9.5f}")
        else:
            print(f"    {name:<14} {m['avg_time_ms']:>8.3f} "
                  f"{m['final_norm']:>9.5f} "
                  f"  found={m.get('found_pct', 0):.0f}%")



# %% ── 5. Run all M ───────────────────────────────────────────────────────────


# %% ── 6. Aggregate summary ───────────────────────────────────────────────────

def _collect(results, method, key):
    return [r["methods"][method][key]
            for r in results
            if method in r.get("methods", {}) and key in r["methods"][method]]


def print_summary(all_M_results):
    print(f"\n\n{'='*80}")
    print(f"  AGGREGATE SUMMARY  —  N={N_PLANTS} plants/M, T={T_SIM}")
    print(f"{'='*80}")

    nan = float('nan')

    print(f"\n  Online solve time (ms/step):")
    print(f"  {'M':>3}  {'Centralized':>12}  {'Explicit-GNE':>14}  {'FACET-GNE':>11}  {'ADMM':>8}")
    print(f"  {'-'*56}")
    for M, results in all_M_results.items():
        c_t = _collect(results, "Centralized",  "avg_time_ms")
        e_t = _collect(results, "Explicit-GNE", "avg_time_ms")
        f_t = _collect(results, "FACET-GNE",    "avg_time_ms")
        a_t = _collect(results, "ADMM",          "avg_time_ms")
        print(f"  {M:>3}  "
              f"{np.mean(c_t) if c_t else nan:>12.4f}  "
              f"{np.mean(e_t) if e_t else nan:>14.4f}  "
              f"{np.mean(f_t) if f_t else nan:>11.4f}  "
              f"{np.mean(a_t) if a_t else nan:>8.2f}")

    print(f"\n  Speedup vs Central and vs ADMM (FACET):")
    print(f"  {'M':>3}  {'vs Central':>14}  {'vs ADMM':>14}  "
          f"{'ADMM avg iters':>15}  {'Conv%':>6}")
    print(f"  {'-'*65}")
    for M, results in all_M_results.items():
        c_t = _collect(results, "Centralized",  "avg_time_ms")
        e_t = _collect(results, "Explicit-GNE", "avg_time_ms")
        f_t = _collect(results, "FACET-GNE",    "avg_time_ms")
        a_t = _collect(results, "ADMM",          "avg_time_ms")
        a_i = _collect(results, "ADMM",          "avg_iters")
        a_c = _collect(results, "ADMM",          "conv_pct")
        
        sp_central = (np.mean(c_t) / np.mean(f_t)) if (c_t and f_t) else nan
        sp_admm    = (np.mean(a_t) / np.mean(f_t)) if (a_t and f_t) else nan
        
        print(f"  {M:>3}  "
              f"{sp_central:>14.2f}×  "
              f"{sp_admm:>14.2f}×  "
              f"{np.mean(a_i) if a_i else 0:>15.1f}  "
              f"{np.mean(a_c) if a_c else 0:>5.0f}%")

    print(f"\n  FACET-GNE combination reduction (offline):")
    print(f"  {'M':>3}  {'Avg CRs/agent':>14}  {'Avg combos':>12}  "
          f"{'Avg FACET combos':>17}  {'Reduction':>10}")
    print(f"  {'-'*62}")
    for M, results in all_M_results.items():
        valid = [r for r in results if r.get("n_crs") is not None]
        if not valid:
            continue
        avg_crs = np.mean([np.mean(r["n_crs"]) for r in valid])
        avg_tot = np.mean([r["n_combos_total"] for r in valid])
        avg_fac = np.mean([r["n_combos_facet"] for r in valid
                           if r["n_combos_facet"] is not None])
        reduction = avg_fac / avg_tot if avg_tot > 0 else 1.0
        print(f"  {M:>3}  {avg_crs:>14.1f}  {avg_tot:>12.0f}  "
              f"{avg_fac:>17.0f}  {reduction*100:>9.0f}%")

    print(f"\n  Control quality — avg ||x_T|| (should → 0 for stable plants):")
    print(f"  {'M':>3}  {'Explicit-GNE':>14}  {'FACET-GNE':>11}  {'ADMM':>8}")
    print(f"  {'-'*42}")
    for M, results in all_M_results.items():
        e_n = _collect(results, "Explicit-GNE", "final_norm")
        f_n = _collect(results, "FACET-GNE",    "final_norm")
        a_n = _collect(results, "ADMM",          "final_norm")
        print(f"  {M:>3}  "
              f"{np.mean(e_n) if e_n else nan:>14.5f}  "
              f"{np.mean(f_n) if f_n else nan:>11.5f}  "
              f"{np.mean(a_n) if a_n else nan:>8.5f}")

    print(f"\n  Offline time per plant (s)  (Explicit skipped for M >= {EXPLICIT_MAX_M}):")
    print(f"  {'M':>3}  {'Explicit (mpQP+enum)':>22}  "
          f"{'FACET (facet+BFS)':>20}")
    print(f"  {'-'*48}")
    for M, results in all_M_results.items():
        valid_e = [r for r in results if r.get("offline_explicit_s") is not None]
        valid_f = [r for r in results if r.get("offline_facet_s") is not None]
        e_str = f"{np.mean([r['offline_explicit_s'] for r in valid_e]):>22.1f}" \
                if valid_e else f"{'skipped':>22}"
        f_str = f"{np.mean([r['offline_facet_s'] for r in valid_f]):>20.1f}" \
                if valid_f else f"{'N/A':>20}"
        print(f"  {M:>3}  {e_str}  {f_str}")

    print(f"\n{'='*80}")


# %% ── 7. Assertions ──────────────────────────────────────────────────────────

def run_assertions(all_M_results):
    print("\nRunning assertions...")
    errors = []

    for M, results in all_M_results.items():
        valid = [r for r in results if not r.get("error")]
        if not valid:
            print(f"  M={M}: no valid results, skipping")
            continue

        # Explicit-GNE vs FACET-GNE consistency (only when Explicit ran)
        for r in valid:
            exh_n = r["methods"].get("Explicit-GNE", {}).get("final_norm")
            fac_n = r["methods"].get("FACET-GNE", {}).get("final_norm")
            if exh_n is not None and fac_n is not None:
                if abs(exh_n - fac_n) > 0.5:
                    errors.append(
                        f"M={M} plant {r['plant_idx']}: "
                        f"Explicit ||xT||={exh_n:.4f} vs FACET={fac_n:.4f}")

        # FACET must not check MORE combos than exhaustive
        for r in valid:
            if r["n_combos_facet"] and r["n_combos_total"]:
                if r["n_combos_facet"] > r["n_combos_total"]:
                    errors.append(
                        f"M={M} plant {r['plant_idx']}: "
                        f"FACET checked {r['n_combos_facet']} > "
                        f"total {r['n_combos_total']}")

        # ADMM should converge >= 50% of steps
        conv = _collect(valid, "ADMM", "conv_pct")
        if conv and np.mean(conv) < 50:
            errors.append(f"M={M} ADMM: conv {np.mean(conv):.0f}% < 50%")

        # All methods: ||x_T|| < 20 (stable plant, not diverging)
        for name in ["Explicit-GNE", "FACET-GNE", "ADMM", "Centralized"]:
            norms = _collect(valid, name, "final_norm")
            if norms and np.mean(norms) >= 20.0:
                errors.append(f"M={M} {name}: avg ||xT||={np.mean(norms):.2f} >= 20")

    if errors:
        print("  FAILED assertions:")
        for e in errors:
            print(f"    ✗ {e}")
    else:
        print("  All assertions passed.")
    return len(errors) == 0


# %% ── 8. Entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nNote: N_PLANTS={N_PLANTS}, T_SIM={T_SIM}.")
    print("For full reproduction: set N_PLANTS=100, T_SIM=50.")
    print(f"Checkpoints saved in: {CKPT_DIR}")
    
    all_M_results: dict[int, list[dict]] = {}
    for M in M_LIST:
        all_M_results[M] = run_case_study_M(M)
        
    print_summary(all_M_results)
    run_assertions(all_M_results)

# %%
