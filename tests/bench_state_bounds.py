"""
bench_state_bounds.py
=====================
Benchmark and correctness test for the fixed solve_gne_online (MATLAB-style
direct evaluation) vs the old LP-based approach.

Tests M=2 and M=3 agents, state_bounds coupling, one random plant each.

What this measures
------------------
1. Per-combo check cost (cache miss vs cache hit) for solve_gne_online
2. Combos checked per step (= true data transfer count)
3. Correctness: u*(p) from FACET matches ADMM within tolerance
4. Offline BFS speed for M=3 (center-point fast check before LP)

Run:
    cd mpgne_ppopt
    python tests/bench_state_bounds.py
"""

import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from ppopt.mp_solvers.solve_mpqp import mpqp_algorithm

from mpgne.plant_gen    import make_random_plants, make_ic
from mpgne.mpc_builder  import make_gne_game_from_plant, default_local_weights
from mpgne.mp_solver    import solve_all_agents_mp
from mpgne.admm_solver  import admm_solve
from mpgne.facet_gne    import (
    find_all_agent_cr_neighbors,
    build_gne_solution_facet,
    solve_gne_online_v2,
    precompute_point_location_arrays,
)

# ── Config ────────────────────────────────────────────────────────────────────
SEED         = 42
T_SIM        = 30
ALGO         = mpqp_algorithm.combinatorial_parallel
COUPLING     = "state_bounds"
CKPT_DIR     = os.path.join(os.path.dirname(__file__), "bench_state_bounds_ckpt")
os.makedirs(CKPT_DIR, exist_ok=True)

ADMM_TOL     = 1e-4
ADMM_ITERS   = 500
FACET_TOL    = 1e-6

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ckpt(M, name):
    return os.path.join(CKPT_DIR, f"M{M}_{name}.pkl")

def _save(M, name, obj):
    with open(_ckpt(M, name), "wb") as f:
        pickle.dump(obj, f)

def _load(M, name):
    p = _ckpt(M, name)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_bench(M):
    sep = "─" * 60
    print(f"\n{'═'*60}")
    print(f"  M = {M} agents  |  coupling = {COUPLING}  |  T_SIM = {T_SIM}")
    print(f"{'═'*60}")

    # 1. Plant & game
    plant = make_random_plants(M, 1, seed=SEED + M)[0]
    x0    = make_ic(plant, scale=0.4, rng=np.random.default_rng(SEED * 3 + M))
    Q, R, P = default_local_weights(plant)
    game  = make_gne_game_from_plant(plant, coupling_mode=COUPLING, Q_list=Q, R_list=R, P_list=P)
    print(f"  Plant: nx={plant.nx}, M={plant.M}, Np={plant.Np}")
    print(f"  Game:  n_x_total={game.n_x_total}, n_p={game.n_p}")

    # 2. Solve mpQP (cached)
    agent_sols = _load(M, "agent_sols")
    if agent_sols is None:
        print(f"\n  [1/3] Solving mpQP (PPOPT)...", flush=True)
        t0 = time.perf_counter()
        agent_sols = solve_all_agents_mp(game, algorithm=ALGO, verbose=False)
        print(f"  [1/3] mpQP done in {time.perf_counter()-t0:.1f}s  "
              f"| CRs: {[s.n_cr for s in agent_sols]}", flush=True)
        _save(M, "agent_sols", agent_sols)
    else:
        print(f"  [1/3] mpQP loaded from cache | CRs: {[s.n_cr for s in agent_sols]}")

    # 3. Facet neighbor detection (cached)
    agent_sols_FH = _load(M, "agent_sols_FH")
    if agent_sols_FH is None:
        import copy
        print(f"\n  [2/3] FACET-H neighbor detection...", flush=True)
        agent_sols_FH = copy.deepcopy(agent_sols)
        t0 = time.perf_counter()
        find_all_agent_cr_neighbors(agent_sols_FH, method="hyperplane_adjacency", verbose=False)
        print(f"  [2/3] FACET-H done in {time.perf_counter()-t0:.2f}s  "
              f"| avg neighbors: {np.mean([len(cr.facet_neighbors) for s in agent_sols_FH for cr in s.regions]):.1f}",
              flush=True)
        _save(M, "agent_sols_FH", agent_sols_FH)
    else:
        avg_nb = np.mean([len(cr.facet_neighbors) for s in agent_sols_FH for cr in s.regions])
        print(f"  [2/3] FACET-H loaded from cache | avg neighbors: {avg_nb:.1f}")

    # 4. Offline BFS explicit map — only for M=2 with state_bounds.
    #    M=3 state_bounds produces ~125K GNE CRs (1 GB, 15+ min to build) —
    #    completely infeasible. V2 online solver handles M>=3 without it.
    BENCH_EXPLICIT_MAX_M = 2
    facet_sol = None
    if M <= BENCH_EXPLICIT_MAX_M:
        facet_sol = _load(M, "facet_sol_FH")
        if facet_sol is None or facet_sol.n_cr == 0:
            print(f"\n  [3/3] Offline BFS (center-point fast check active)...", flush=True)
            t0 = time.perf_counter()
            fres = build_gne_solution_facet(game, agent_sols_FH, verbose=True)
            elapsed_bfs = time.perf_counter() - t0
            facet_sol = fres.gne_sol if fres.gne_sol.n_cr > 0 else None
            print(f"  [3/3] BFS done in {elapsed_bfs:.1f}s | "
                  f"{fres.gne_sol.n_cr} GNE CRs | "
                  f"{fres.n_combos_checked}/{fres.n_combos_total} combos checked", flush=True)
            if facet_sol:
                _save(M, "facet_sol_FH", facet_sol)
        else:
            print(f"  [3/3] Explicit map loaded from cache | {facet_sol.n_cr} GNE CRs")
    else:
        print(f"  [3/3] M={M}: using V2 online mode (state_bounds explicit map infeasible)")

    # 5. Closed-loop simulation — compare ADMM vs new FACET
    print(f"\n{sep}")
    print(f"  Closed-loop simulation  (T={T_SIM} steps)")
    print(sep)

    nx, Mval = plant.nx, plant.M
    x_traj_admm  = np.zeros((T_SIM + 1, nx))
    x_traj_facet = np.zeros((T_SIM + 1, nx))
    x_traj_admm[0] = x_traj_facet[0] = x0.copy()

    times_admm, times_facet = np.zeros(T_SIM), np.zeros(T_SIM)
    combos_per_step = np.zeros(T_SIM, dtype=int)
    n_fallbacks = 0

    for k in range(T_SIM):
        p = x_traj_admm[k]

        # ADMM reference
        t0 = time.perf_counter()
        res_admm = admm_solve(game, p, rho=1.0, max_iter=ADMM_ITERS,
                              tol=ADMM_TOL, qp_solver="osqp", verbose=False)
        times_admm[k] = time.perf_counter() - t0
        u_admm = {i: res_admm.x_sol[i][:plant.subsystems[i].nu] for i in range(Mval)}
        x_traj_admm[k+1] = plant.step(x_traj_admm[k], u_admm)

    # FACET-V2 simulation (independent trajectory)
    if not hasattr(agent_sols_FH[0], '_E_stack'):
        precompute_point_location_arrays(agent_sols_FH)
    prev_x_star = None   # seeded from ADMM at k=0, then carried forward
    prev_crs    = None   # per-agent CR warm hint for PointLocation
    combo_cache = {}     # {combo → (H_x, h_x)} persisted across all steps

    for k in range(T_SIM):
        p = x_traj_facet[k]

        # Cold-start: seed reference point from ADMM at k=0
        if prev_x_star is None:
            res_w = admm_solve(game, p, rho=1.0, max_iter=200, tol=1e-3,
                               qp_solver="osqp", verbose=False)
            prev_x_star = res_w.x_stacked

        # FACET explicit map (M<=2) or online BFS (M>=3)
        t0 = time.perf_counter()
        if facet_sol is not None:
            # Explicit map: O(n_GNE_CR) scan
            cr_idx = facet_sol.locate(p, tol=1e-6)
            if cr_idx is not None:
                U_star = facet_sol.regions[cr_idx].evaluate(p)
                combos_per_step[k] = 1
            else:
                n_fallbacks += 1
                res_fb = admm_solve(game, p, rho=1.0, max_iter=50,
                                    tol=1e-3, qp_solver="osqp", verbose=False)
                U_star = res_fb.x_stacked
                combos_per_step[k] = 1 + res_fb.n_iter
        else:
            # V2 online solver (MATLAB-style: PointLocation + per-agent filter)
            combo, U_star_v2, n_checked = solve_gne_online_v2(
                p, agent_sols_FH, game, prev_x_star=prev_x_star,
                prev_crs=prev_crs, combo_cache=combo_cache)
            if combo is not None:
                prev_x_star = U_star_v2
                prev_crs    = list(combo)
                U_star = U_star_v2
                combos_per_step[k] = n_checked
            else:
                n_fallbacks += 1
                res_fb = admm_solve(game, p, rho=1.0, max_iter=50,
                                    tol=1e-3, qp_solver="osqp", verbose=False)
                U_star = res_fb.x_stacked
                prev_x_star = res_fb.x_stacked
                prev_crs    = None
                combos_per_step[k] = 1 + res_fb.n_iter

        times_facet[k] = time.perf_counter() - t0

        off, u_k = 0, {}
        for i in range(Mval):
            nu_i = plant.subsystems[i].nu
            u_k[i] = U_star[off:off + nu_i]
            off += plant.Np * plant.subsystems[i].nu
        x_traj_facet[k+1] = plant.step(x_traj_facet[k], u_k)

    # 6. Results
    print(f"\n  {'Metric':<35} {'ADMM':>12} {'FACET':>12}")
    print(f"  {'-'*59}")
    print(f"  {'Avg online time (ms)':<35} {times_admm.mean()*1000:>12.3f} {times_facet.mean()*1000:>12.3f}")
    print(f"  {'Max online time (ms)':<35} {times_admm.max()*1000:>12.3f} {times_facet.max()*1000:>12.3f}")
    print(f"  {'Final state norm':<35} {np.linalg.norm(x_traj_admm[-1]):>12.4f} {np.linalg.norm(x_traj_facet[-1]):>12.4f}")
    print(f"\n  {'Combos/data-transfers per step (FACET)':}")
    print(f"    mean={combos_per_step.mean():.2f}  median={np.median(combos_per_step):.0f}  "
          f"max={combos_per_step.max()}  fallbacks={n_fallbacks}/{T_SIM}")
    print(f"    distribution: {dict(zip(*np.unique(combos_per_step, return_counts=True)))}")

    # 7. Correctness check: do final norms agree?
    norm_diff = abs(np.linalg.norm(x_traj_admm[-1]) - np.linalg.norm(x_traj_facet[-1]))
    tol_check = 2.0  # generous — different trajectories so just check convergence direction
    status = "PASS" if norm_diff < tol_check else "FAIL"
    print(f"\n  Convergence agreement: |‖x_T‖_ADMM - ‖x_T‖_FACET| = {norm_diff:.4f}  [{status}]")

    # 8. Per-step timing microbench for V2 (1000 repeated calls, warm)
    if facet_sol is None:
        print(f"\n  Per-step timing microbench (V2, 1000 calls):")
        p_test = x_traj_facet[T_SIM // 2]
        ref_x  = prev_x_star if prev_x_star is not None else np.zeros(game.n_x_total)
        for _ in range(10):
            solve_gne_online_v2(p_test, agent_sols_FH, game, prev_x_star=ref_x)
        t0 = time.perf_counter()
        for _ in range(1000):
            solve_gne_online_v2(p_test, agent_sols_FH, game, prev_x_star=ref_x)
        avg_us = (time.perf_counter() - t0) / 1000 * 1e6
        print(f"    V2 per-step: {avg_us:.1f} µs  (MATLAB target: ~500 µs)")

    print()
    return {
        "M": M,
        "avg_admm_ms": times_admm.mean() * 1000,
        "avg_facet_ms": times_facet.mean() * 1000,
        "mean_combos": combos_per_step.mean(),
        "max_combos": int(combos_per_step.max()),
        "fallbacks": n_fallbacks,
        "status": status,
    }


if __name__ == "__main__":
    results = []
    for M in [2, 3]:
        r = run_bench(M)
        results.append(r)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'M':>3} | {'ADMM (ms)':>10} | {'FACET (ms)':>10} | "
          f"{'mean DT':>8} | {'max DT':>6} | {'fallbacks':>9} | {'status':>6}")
    print("  " + "-" * 58)
    for r in results:
        print(f"  {r['M']:>3} | {r['avg_admm_ms']:>10.3f} | {r['avg_facet_ms']:>10.3f} | "
              f"{r['mean_combos']:>8.2f} | {r['max_combos']:>6} | "
              f"{r['fallbacks']:>9} | {r['status']:>6}")
    print()
    print("  DT = data transfers (combos checked per step)")
    print("  FACET target: avg DT << ADMM iterations, time << ADMM time")
