# %% ── 1. Imports & Configuration ───────────────────────────────────────────────
import sys, os

_curr = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
_found_root = None
for _ in range(4):
    if os.path.isdir(os.path.join(_curr, "mpgne")):
        _found_root = _curr
        break
    _curr = os.path.dirname(_curr)

if _found_root:
    _base_path = _found_root
    if _base_path not in sys.path:
        sys.path.insert(0, _base_path)
else:
    _base_path = os.getcwd()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from ppopt.mp_solvers.solve_mpqp import mpqp_algorithm

from mpgne.plant_gen import make_random_plants, make_ic
from mpgne.mpc_builder import make_gne_game_from_plant, default_local_weights
from mpgne.mp_solver import solve_all_agents_mp
from mpgne.facet_gne import find_all_agent_cr_neighbors, build_gne_solution_facet
from mpgne.cr_store import (
    save_agent_solutions, load_agent_solutions,
    save_gne_solution, load_gne_solution
)
from mpgne.admm_solver import admm_solve
from mpgne.proj_grad_solver import pg_solve
# %% ── 2. Settings ────────────────────────────────────────────────────────────
M                 = 6
T_SIM             = 100
L_MAX             = 2.5
OFFLINE_BFS_MAX_M = 4

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"  Multi-Agent GNE Demo (M = {M})")
    print("=" * 60)

    # ── Generate plant ────────────────────────────────────────────────────────
    print(f"\n[1/5] Generating random system with M={M} agents...")
    plant = make_random_plants(M, 1, seed=20)[0]

    # ── Build game ────────────────────────────────────────────────────────────
    print(f"[2/5] Building game formulation...")
    Q_list, R_list, P_list = default_local_weights(plant)
    game = make_gne_game_from_plant(
        plant, L_max=L_MAX,
        Q_list=Q_list, R_list=R_list, P_list=P_list
    )

    # %% ── 3. Offline mpQP Solving (with Checkpoints) ────────────────────────
    ckpt_dir = os.path.join(_base_path, "checkpoints_demo")
    os.makedirs(ckpt_dir, exist_ok=True)
    agent_ckpt = os.path.join(ckpt_dir, f"agent_sols_M{M}_L{L_MAX}.pkl")
    facet_ckpt = os.path.join(ckpt_dir, f"facet_sol_M{M}_L{L_MAX}.pkl")

    if os.path.exists(agent_ckpt) and os.path.exists(facet_ckpt):
        print(f"\n[3-4/5] Loading precomputed solutions from checkpoints_demo...")
        agent_sols = load_agent_solutions(agent_ckpt)
        facet_sol  = load_gne_solution(facet_ckpt)
    else:
        print("\n[3/5] Solving offline mpQP for all agents...")
        t0 = time.perf_counter()
        agent_sols = solve_all_agents_mp(
            game,
            algorithm=mpqp_algorithm.combinatorial_parallel,
            verbose=False
        )
        print(f"      -> Completed in {time.perf_counter() - t0:.2f}s")

        print("\n[4/5] Detecting facet neighbors (Parallel)...")
        t0 = time.perf_counter()
        find_all_agent_cr_neighbors(agent_sols, method="hyperplane", verbose=False)
        print(f"      -> Neighbors found in {time.perf_counter() - t0:.2f}s")

        save_agent_solutions(agent_sols, agent_ckpt)

        nb_counts = [sum(len(cr.facet_neighbors) for cr in s.regions) for s in agent_sols]
        print(f"      -> Diagnostic: Neighbor counts per agent: {nb_counts}")

        if M < OFFLINE_BFS_MAX_M:
            print("      Building FACET-GNE explicit solution via BFS...")
            t0 = time.perf_counter()
            facet_res = build_gne_solution_facet(game, agent_sols, verbose=True)
            facet_sol = facet_res.gne_sol
            print(f"      -> Solution built in {time.perf_counter() - t0:.2f}s")
            print(f"      -> Total combinations checked: {facet_res.n_combos_checked}")
            save_gne_solution(facet_sol, facet_ckpt)
        else:
            print(f"      -> Skipping global BFS map (M={M} >= {OFFLINE_BFS_MAX_M}). Will solve online.")
            from mpgne.cr_store import GNESolution
            facet_sol = GNESolution([], game.n_p, game.N)
            save_gne_solution(facet_sol, facet_ckpt)
            
        # RELOAD from disk to guarantee memory layout matches Run 2
        agent_sols = load_agent_solutions(agent_ckpt)
        facet_sol  = load_gne_solution(facet_ckpt)

    # %% ── 5. Online Closed-Loop Simulation ───────────────────────────────────
    print("\n[5/5] Running closed-loop simulation...")
    from mpgne.facet_gne import solve_gne_online

    x_traj = np.zeros((T_SIM + 1, plant.nx))
    u_traj = {i: np.zeros((T_SIM, plant.subsystems[i].nu)) for i in range(M)}
    x_traj[0] = make_ic(plant, scale=0.4, rng=np.random.default_rng(2025))

    facet_times       = []
    admm_times        = []
    admm_iters        = []
    pg_times          = []
    pg_iters          = []
    error_gaps        = []
    admm_x_warm       = None
    prev_combo        = None
    admm_conv_hist_k0 = None
    pg_conv_hist_k0   = None

    # ── Minimal warmup: forces scipy/BLAS/HiGHS libs to load before timing ───
    # Each call runs exactly 1 iteration — total cost < 10 ms.
    # Without this, the FIRST call to solve_gne_online pays a ~400 ms
    # one-time OS page-cache penalty (loading shared libraries from disk),
    # making Run 1 timings differ from Run 2+.
    print("  [warmup] Loading solver libraries (1 iter each)...")
    _p0 = x_traj[0]
    admm_solve(game, _p0, x_init=None, max_iter=1, tol=1e-20)   # warms BLAS
    pg_solve(game, _p0,   max_iter=1,   tol=1e-20)               # warms SLSQP
    solve_gne_online(_p0, (0,) * M, agent_sols, game)            # warms Python overheads
    import scipy.optimize
    scipy.optimize.linprog(c=[1], A_ub=[[1]], b_ub=[1], bounds=(0, 1), method='highs') # warms HiGHS
    print("  [warmup] Done — starting timed benchmark.")

    for k in range(T_SIM):
        p = x_traj[k]

        # ── Benchmark 1: ADMM cold-start ──────────────────────────────────────
        t0_admm = time.perf_counter()
        res_admm = admm_solve(game, p, x_init=None, max_iter=2000, tol=1e-4)
        admm_times.append(time.perf_counter() - t0_admm)
        admm_iters.append(res_admm.n_iter)
        U_admm      = res_admm.x_stacked
        admm_x_warm = U_admm.copy()
        if k == 0:
            admm_conv_hist_k0 = res_admm.primal_hist

        # ── Benchmark 2: Projected Gradient / Jacobi BR ───────────────────────
        t0_pg = time.perf_counter()
        res_pg = pg_solve(game, p, max_iter=2000, tol=1e-4)
        pg_times.append(time.perf_counter() - t0_pg)
        pg_iters.append(res_pg.n_iter)
        if k == 0:
            pg_conv_hist_k0 = res_pg.conv_hist

        # ── FACET explicit / online solve ─────────────────────────────────────
        t0_facet = time.perf_counter()
        used_fb  = False

        if M < OFFLINE_BFS_MAX_M:
            cr_idx = facet_sol.locate(p)
            if cr_idx is not None:
                U_explicit = facet_sol[cr_idx].evaluate(p)
            else:
                U_explicit = U_admm.copy()
                used_fb    = True
        else:
            if prev_combo is None:
                # Use U_admm (already computed above, free) to seed initial region
                _Ud, _off = {}, 0
                for _i in range(M):
                    _n = plant.Np * plant.subsystems[_i].nu
                    _Ud[_i] = U_admm[_off:_off+_n]; _off += _n
                _wc = []
                for _i in range(M):
                    _th = np.concatenate([p] + [_Ud[_j] for _j in range(M) if _j != _i])
                    _vi, _best = 0, float('inf')
                    for _v, _cr in enumerate(agent_sols[_i].regions):
                        _vl = float(np.max(_cr.E @ _th - _cr.f))
                        if _vl < _best: _best, _vi = _vl, _v
                    _wc.append(_vi)
                prev_combo = tuple(_wc)

            combo, U_star, _ = solve_gne_online(p, prev_combo, agent_sols, game)
            if combo is not None:
                U_explicit  = U_star
                prev_combo  = combo
                admm_x_warm = U_star
            else:
                U_explicit = U_admm.copy()   # free — already computed
                used_fb    = True
                prev_combo = None

        facet_times.append(time.perf_counter() - t0_facet)
        error_gaps.append(np.linalg.norm(U_explicit - U_admm))

        # ── Plant step ────────────────────────────────────────────────────────
        u_k = {i: U_explicit[game.x_slice(i)][:plant.subsystems[i].nu]
               for i in range(M)}
        for i in range(M):
            u_traj[i][k] = u_k[i]
        x_traj[k + 1] = plant.step(x_traj[k], u_k)

        if (k + 1) % 20 == 0 or k == 0:
            print(f"  step {k+1:3d}/{T_SIM}  "
                  f"ADMM:{res_admm.n_iter:4d}itr  "
                  f"BR:{res_pg.n_iter:4d}itr  "
                  f"FACET:{facet_times[-1]*1000:.3f}ms"
                  f"{'  [fallback]' if used_fb else ''}")

    # %% ── 6. Performance Summary ─────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  PERFORMANCE COMPARISON: EXPLICIT vs ITERATIVE")
    print("=" * 75)
    print(f"{'Method':<25} | {'Avg Time':>10} | {'Max Time':>10} | {'Avg Iters':>10}")
    print("-" * 75)
    print(f"{'Iterative (ADMM)':<25} | "
          f"{np.mean(admm_times)*1000:>8.3f} ms | "
          f"{np.max(admm_times)*1000:>8.3f} ms | "
          f"{np.mean(admm_iters):>10.1f}")
    print(f"{'Iterative (Jacobi BR)':<25} | "
          f"{np.mean(pg_times)*1000:>8.3f} ms | "
          f"{np.max(pg_times)*1000:>8.3f} ms | "
          f"{np.mean(pg_iters):>10.1f}")
    print(f"{'Explicit (FACET)':<25} | "
          f"{np.mean(facet_times)*1000:>8.3f} ms | "
          f"{np.max(facet_times)*1000:>8.3f} ms | "
          f"{'N/A':>10}")
    print("-" * 75)
    print(f"  -> Speedup vs ADMM   : {np.mean(admm_times)/np.mean(facet_times):.1f}x faster")
    print(f"  -> Speedup vs BR     : {np.mean(pg_times)/np.mean(facet_times):.1f}x faster")
    print(f"  -> Final state norm  : {np.linalg.norm(x_traj[-1]):.2e}")
    print(f"  -> Avg optimality gap: {np.mean(error_gaps):.2e}")
    print("=" * 75)

    # %% ── 7. Plots ───────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    C_ADMM  = "#E07B54"
    C_PG    = "#5B8DB8"
    C_FACET = "#4CAF82"

    # Figure 1: Closed-loop trajectories
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig1.patch.set_facecolor("#1a1a2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#16213e"); ax.tick_params(colors='white')
        ax.spines[:].set_color("#444")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('white')
    for i in range(plant.nx):
        ax1.plot(np.arange(T_SIM+1), x_traj[:, i], linewidth=1.5)
    ax1.set_ylabel("States $x(k)$", fontsize=12, fontweight='bold', color='white')
    ax1.set_title(f"Distributed MPC via FACET-GNE (M={M} Agents)",
                  fontsize=14, fontweight='bold', color='white')
    ax1.grid(True, linestyle="--", alpha=0.7)
    for i in range(M):
        for j in range(plant.subsystems[i].nu):
            ax2.plot(np.arange(T_SIM), u_traj[i][:, j],
                     drawstyle='steps-post', linewidth=1.5)
    ax2.set_ylabel("Control Inputs $u(k)$", fontsize=12, fontweight='bold', color='white')
    ax2.set_xlabel("Time step $k$", fontsize=12, fontweight='bold', color='white')
    ax2.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    p1 = os.path.join(os.path.dirname(__file__), "demo_trajectory.png")
    fig1.savefig(p1, dpi=200, bbox_inches='tight', facecolor=fig1.get_facecolor())
    print(f"Plot saved: {p1}")

    # Figure 2: Benchmark comparison (2×2)
    fig2 = plt.figure(figsize=(14, 10))
    fig2.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.42, wspace=0.35)

    def _style(ax, title, xlabel, ylabel):
        ax.set_facecolor("#16213e"); ax.tick_params(colors='white')
        ax.spines[:].set_color("#444")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('white')
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel(xlabel, color='white', fontsize=10)
        ax.set_ylabel(ylabel, color='white', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.35, color='#888')

    ax_c = fig2.add_subplot(gs[0, 0])
    if admm_conv_hist_k0:
        ax_c.semilogy(admm_conv_hist_k0, color=C_ADMM, lw=2,
                      label=f"ADMM ({len(admm_conv_hist_k0)} itr)")
    if pg_conv_hist_k0:
        ax_c.semilogy(pg_conv_hist_k0, color=C_PG, lw=2,
                      label=f"Jacobi BR ({len(pg_conv_hist_k0)} itr)")
    ax_c.axhline(1e-4, color='white', ls=':', lw=1, alpha=0.6, label="tol=1e-4")
    _style(ax_c, "Convergence at Step k=0", "Iteration", "Residual / ‖Δx‖")
    ax_c.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', framealpha=0.7)

    ax_i = fig2.add_subplot(gs[0, 1])
    ax_i.plot(admm_iters, color=C_ADMM, lw=1.5, label="ADMM")
    ax_i.plot(pg_iters,   color=C_PG,   lw=1.5, label="Jacobi BR")
    _style(ax_i, "Iterations per Time Step", "Time step $k$", "# Iterations")
    ax_i.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', framealpha=0.7)

    ax_t = fig2.add_subplot(gs[1, 0])
    bp = ax_t.boxplot(
        [np.array(admm_times)*1000, np.array(pg_times)*1000,
         np.array(facet_times)*1000],
        patch_artist=True, widths=0.5,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(color='#aaa'), capprops=dict(color='#aaa'),
        flierprops=dict(marker='o', color='#aaa', markersize=3),
    )
    for patch, c in zip(bp['boxes'], [C_ADMM, C_PG, C_FACET]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax_t.set_yscale('log')
    ax_t.set_xticks([1, 2, 3])
    ax_t.set_xticklabels(["ADMM\n(warm)", "Jacobi BR\n(cold)", "FACET-GNE\n(explicit)"],
                          color='white', fontsize=9)
    _style(ax_t, "Solve-Time Distribution (log scale)", "Method", "Time (ms)")

    ax_s = fig2.add_subplot(gs[1, 1])
    su_a = np.mean(admm_times) / np.mean(facet_times)
    su_p = np.mean(pg_times)   / np.mean(facet_times)
    bars = ax_s.bar(["ADMM\nvs FACET", "Jacobi BR\nvs FACET"],
                    [su_a, su_p], color=[C_ADMM, C_PG],
                    alpha=0.8, width=0.4, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, [su_a, su_p]):
        ax_s.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                  f"{val:.1f}×", ha='center', va='bottom',
                  color='white', fontsize=12, fontweight='bold')
    _style(ax_s, "FACET-GNE Speedup vs Iterative", "Comparison", "Speedup (×)")

    fig2.suptitle(f"Benchmark: FACET-GNE vs Iterative Solvers (M={M}, L_max={L_MAX})",
                  color='white', fontsize=13, fontweight='bold', y=1.01)
    p2 = os.path.join(os.path.dirname(__file__), "demo_benchmark.png")
    fig2.savefig(p2, dpi=200, bbox_inches='tight', facecolor=fig2.get_facecolor())
    print(f"Plot saved: {p2}")

    try:
        plt.show()
    except Exception:
        pass
# %%
