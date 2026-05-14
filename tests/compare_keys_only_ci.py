"""
compare_keys_only_ci.py
=======================
TEMPORARY benchmark comparing two FH-CI storage strategies:

  1. Full CI   — dict[combo → (H_x, h_x)]   ← current implementation
                 ~89 MB on disk for M=4, ~3 µs/check (dict lookup + matvec)

  2. Keys-only — set[combo]                 ← proposed lighter variant
                 ~5 MB on disk for M=4, recomputes (H_x, h_x) via
                 equilibrium solve on every check (~50 µs each)

Loads the existing M=4 checkpoints from full_case_study, runs T_SIM
steps with each variant, and reports timing and trajectory diff.
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from collections import deque

from mpgne.plant_gen import make_random_plants, make_ic
from mpgne.mpc_builder import make_gne_game_from_plant
from mpgne.facet_gne import (
    _assemble_equilibrium_system, _solve_equilibrium,
    solve_gne_online_ci,
)
from mpgne.admm_solver import admm_solve

# ── Config — mirror full_case_study so trajectories are identical ─────────────
M           = 4
PLANT_IDX   = 0
T_SIM       = 100
SEED        = 42
ADMM_RHO    = 1.0
QP_SOLVER   = "osqp"
COUPLING_MODE = "l_max"
L_MAX       = 5.0
FHCI_MAX_HOPS = 3

CKPT_DIR = os.path.join(os.path.dirname(__file__), f"full_case_study_data_{COUPLING_MODE}")

def _load(name):
    p = os.path.join(CKPT_DIR, f"M{M}_plant{PLANT_IDX:03d}_{name}.pkl")
    with open(p, "rb") as f:
        return pickle.load(f)


# ── Keys-only solver — same structure as solve_gne_online_ci but recomputes ───
def solve_gne_online_keys_only(p, prev_combo, agent_sols, combo_keys, game,
                                tol=1e-6, max_hops=3,
                                exhaustive_fallback=True, tol_rank=1e-8):
    """
    Same two-tier (BFS + exhaustive) search as solve_gne_online_ci, but
    storage is just a set of feasible combo tuples — (H_x, h_x) recomputed
    on every check via _assemble_equilibrium_system + _solve_equilibrium.
    """
    M_ = game.N
    n_checked = 0

    if prev_combo is None:
        return None, None, 0

    def _check(combo):
        """Recompute (H_x, h_x) on the fly, then verify CR validity."""
        Mx, Mp, M1 = _assemble_equilibrium_system(combo, agent_sols, game)
        eq = _solve_equilibrium(Mx, Mp, M1, tol_rank=tol_rank)
        if not eq.solvable:
            return False, None
        x_star = eq.H_x @ p + eq.h_x
        for i, j_i in enumerate(combo):
            cr = agent_sols[i].regions[j_i]
            others = [j for j in range(M_) if j != i]
            x_neg = np.concatenate([x_star[game.x_slice(j)] for j in others])
            theta_i = np.concatenate([x_neg, p])
            if np.any(cr.E @ theta_i > cr.f + tol):
                return False, x_star
        return True, x_star

    base = tuple(prev_combo)
    seen = {base}
    queue = deque([(base, 0)])

    # Tier 1: BFS
    while queue:
        combo, depth = queue.popleft()
        n_checked += 1
        if combo in combo_keys:
            ok, x_star = _check(combo)
            if ok:
                return combo, x_star, n_checked
        if max_hops is not None and depth >= max_hops:
            continue
        for i in range(M_):
            for nbr in agent_sols[i].regions[combo[i]].facet_neighbors:
                nxt_t = combo[:i] + (nbr,) + combo[i + 1:]
                if nxt_t not in seen:
                    seen.add(nxt_t)
                    queue.append((nxt_t, depth + 1))

    # Tier 2: exhaustive scan over the key set
    if exhaustive_fallback:
        for combo in combo_keys:
            if combo in seen:
                continue
            n_checked += 1
            ok, x_star = _check(combo)
            if ok:
                return combo, x_star, n_checked

    return None, None, n_checked


# ── Load checkpoints ──────────────────────────────────────────────────────────
print(f"Loading M={M} plant {PLANT_IDX} checkpoints...")
agent_sols_FH = _load("agent_sols_FH")
combo_index   = _load("combo_index_FH")           # dict[combo → (H_x, h_x)]
combo_keys    = set(combo_index.keys())           # set[combo]

# Rebuild plant + game (same RNG as full_case_study)
plants = make_random_plants(M, 1, seed=SEED + M)
rng_ic = np.random.default_rng(SEED * 2 + M)
x0 = make_ic(plants[PLANT_IDX], scale=0.4, rng=rng_ic)
plant = plants[PLANT_IDX]
game = make_gne_game_from_plant(plant, coupling_mode=COUPLING_MODE, L_max=L_MAX)

print(f"  combo_index entries: {len(combo_index):,}")
print(f"  combo_keys entries:  {len(combo_keys):,}")
print(f"  Storage (current):   {sys.getsizeof(combo_index)/1024:.0f} KB in memory")
print(f"  Storage (keys-only): {sys.getsizeof(combo_keys)/1024:.0f} KB in memory")


# ── Simulation helpers ────────────────────────────────────────────────────────
def _unstack(U_star):
    u = {}; off = 0
    for i in range(M):
        nu_i = plant.subsystems[i].nu
        u[i] = U_star[off:off + nu_i]
        off += plant.Np * nu_i
    return u

def _seed(p, U_admm):
    _Ud, _off = {}, 0
    for i in range(M):
        _n = plant.Np * plant.subsystems[i].nu
        _Ud[i] = U_admm[_off:_off + _n]; _off += _n
    _wc = []
    for i in range(M):
        _th = np.concatenate([p] + [_Ud[j] for j in range(M) if j != i])
        _vi, _best = 0, float('inf')
        for _v, _cr in enumerate(agent_sols_FH[i].regions):
            _vl = float(np.max(_cr.E @ _th - _cr.f))
            if _vl < _best:
                _best, _vi = _vl, _v
        _wc.append(_vi)
    return tuple(_wc)


def run_sim(solver_name, solver_fn):
    """Run T_SIM closed-loop steps using solver_fn for the online lookup."""
    x_traj = np.zeros((T_SIM + 1, plant.nx))
    x_traj[0] = x0.copy()
    times = np.zeros(T_SIM)
    n_checked_arr = np.zeros(T_SIM, dtype=int)
    fallbacks = 0
    prev_combo = None

    for k in range(T_SIM):
        p = x_traj[k]
        t_start = time.perf_counter()

        if prev_combo is None:
            res_warm = admm_solve(game, p, rho=ADMM_RHO, max_iter=500,
                                  tol=1e-4, qp_solver=QP_SOLVER, verbose=False)
            prev_combo = _seed(p, res_warm.x_stacked)

        combo, x_star, n_checked = solver_fn(p, prev_combo)
        n_checked_arr[k] = n_checked

        if combo is not None:
            prev_combo = combo
            u_k = _unstack(x_star)
        else:
            fallbacks += 1
            res = admm_solve(game, p, rho=ADMM_RHO, max_iter=200,
                             tol=1e-3, qp_solver=QP_SOLVER, verbose=False)
            u_k = {i: res.x_sol[i][:plant.subsystems[i].nu] for i in range(M)}
            prev_combo = _seed(p, res.x_stacked)

        times[k] = time.perf_counter() - t_start
        x_traj[k + 1] = plant.step(x_traj[k], u_k)

    print(f"\n[{solver_name}]")
    print(f"  Avg time/step:   {times.mean()*1000:8.3f} ms")
    print(f"  Max time/step:   {times.max()*1000:8.3f} ms")
    print(f"  Total time:      {times.sum()*1000:8.1f} ms over {T_SIM} steps")
    print(f"  Avg candidates:  {n_checked_arr.mean():8.1f}")
    print(f"  Max candidates:  {n_checked_arr.max():8d}")
    print(f"  Fallbacks:       {fallbacks}")
    return x_traj, times


def run_admm_sim():
    """Pure ADMM closed-loop — full 2000-iter cap, 1e-4 tol (same as benchmark)."""
    x_traj = np.zeros((T_SIM + 1, plant.nx))
    x_traj[0] = x0.copy()
    times = np.zeros(T_SIM)
    iters_arr = np.zeros(T_SIM, dtype=int)

    for k in range(T_SIM):
        p = x_traj[k]
        t_start = time.perf_counter()
        res = admm_solve(game, p, rho=ADMM_RHO, max_iter=2000, tol=1e-4,
                         qp_solver=QP_SOLVER, verbose=False)
        times[k] = time.perf_counter() - t_start
        iters_arr[k] = res.n_iter
        u_k = {i: res.x_sol[i][:plant.subsystems[i].nu] for i in range(M)}
        x_traj[k + 1] = plant.step(x_traj[k], u_k)

    print(f"\n[ADMM (no FACET, full 2000-iter cap)]")
    print(f"  Avg time/step:   {times.mean()*1000:8.3f} ms")
    print(f"  Max time/step:   {times.max()*1000:8.3f} ms")
    print(f"  Total time:      {times.sum()*1000:8.1f} ms over {T_SIM} steps")
    print(f"  Avg iterations:  {iters_arr.mean():8.1f}")
    print(f"  Max iterations:  {iters_arr.max():8d}")
    print(f"  Total comm rounds (iters sum): {iters_arr.sum():,}")
    return x_traj, times, iters_arr


# ── Run all three methods ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Running {T_SIM} steps with three approaches")
print("=" * 60)

x_full, t_full = run_sim(
    "FULL CI  (dict[combo → (H_x,h_x)])",
    lambda p, c: solve_gne_online_ci(
        p, c, agent_sols_FH, combo_index, game,
        max_hops=FHCI_MAX_HOPS, exhaustive_fallback=True),
)

x_keys, t_keys = run_sim(
    "KEYS-ONLY (set[combo], recompute every check)",
    lambda p, c: solve_gne_online_keys_only(
        p, c, agent_sols_FH, combo_keys, game,
        max_hops=FHCI_MAX_HOPS, exhaustive_fallback=True),
)

x_admm, t_admm, iters_admm = run_admm_sim()

# ── Compare ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Side-by-side")
print("=" * 60)
print(f"  {'Method':<24} | {'Avg ms':>10} | {'Total s':>9} | {'vs ADMM':>9}")
print(f"  {'-'*24}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}")
admm_avg = t_admm.mean() * 1000
for name, ts in [("FULL CI", t_full), ("KEYS-ONLY", t_keys), ("ADMM", t_admm)]:
    avg_ms = ts.mean() * 1000
    total_s = ts.sum()
    speedup = admm_avg / avg_ms
    print(f"  {name:<24} | {avg_ms:>10.3f} | {total_s:>9.2f} | {speedup:>8.1f}×")

print(f"\nTrajectory comparison:")
print(f"  ||x_full - x_keys||_∞ = {np.max(np.abs(x_full - x_keys)):.2e}")
print(f"  ||x_full - x_admm||_∞ = {np.max(np.abs(x_full - x_admm)):.2e}")
print(f"  ||x_keys - x_admm||_∞ = {np.max(np.abs(x_keys - x_admm)):.2e}")
print(f"  (differences ~0 mean same trajectory; large = different valid GNE selected)")

print("\nDone.")
