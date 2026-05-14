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
from mpgne.impdimc_solver import impdimc_solve

# %% ── 2. Settings ────────────────────────────────────────────────────────────
M                 = 2
T_SIM             = 100
OFFLINE_BFS_MAX_M = 4

# Inner QP solver for ADMM and Jacobi BR.
# "osqp"  — OSQP (fast, cold-start). Requires: pip install osqp
# "slsqp" — scipy SLSQP (no extra install, larger speedup gap vs FACET)
QP_SOLVER         = "osqp"

# Coupling formulation for the GNE game:
#   "state_bounds" — generalized Nash via  x_lb ≤ x_k ≤ x_ub  (default, ACC 2026)
#   "l_max"        — aggregate-input coupling  Σ_j u_{j,k} ≤ L_MAX per step
COUPLING_MODE     = "state_bounds"
L_MAX             = 5.0   # only used when COUPLING_MODE = "l_max"

# Neighbor-finding methods to benchmark:
#   "FACET-H"  — Hyperplane Adjacency (fast offline, over-inclusive neighbor sets)
#   "FACET-LP" — LP Facet Adjacency   (exact, compact — ACC 2026 paper)
NB_METHODS = {
    "FACET-H":  "hyperplane_adjacency",
    "FACET-LP": "facet_adjacency",
}

# ─────────────────────────────────────────────────────────────────────────────
def _run_for_mode(coupling_mode: str, L_max: float) -> None:
    """Run the full demo pipeline for one coupling formulation."""
    from mpgne.cr_store import GNESolution
    import copy

    _coupling_label = f"l_max (L_MAX={L_max})" if coupling_mode == "l_max" else "state_bounds"

    print("\n" + "=" * 60)
    print(f"  Multi-Agent GNE Demo (M = {M})")
    print(f"  Coupling: {_coupling_label}")
    print("=" * 60)

    # ── Generate plant ────────────────────────────────────────────────────────
    print(f"\n[1/5] Generating random system with M={M} agents...")
    plant = make_random_plants(M, 1, seed=202)[0]

    # ── Build game ────────────────────────────────────────────────────────────
    print(f"[2/5] Building game formulation (coupling={coupling_mode})...")
    Q_list, R_list, P_list = default_local_weights(plant)
    game = make_gne_game_from_plant(
        plant,
        coupling_mode=coupling_mode, L_max=L_max,
        Q_list=Q_list, R_list=R_list, P_list=P_list
    )

    # %% ── 3. Offline mpQP Solving ────────────────────────────────────────────
    ckpt_dir   = os.path.join(_base_path, f"checkpoints_demo_{coupling_mode}")
    os.makedirs(ckpt_dir, exist_ok=True)
    base_ckpt  = os.path.join(ckpt_dir, f"agent_sols_base_M{M}.pkl")

    # Solve mpQP once (shared by both neighbor methods)
    if os.path.exists(base_ckpt):
        print(f"\n[3/5] Loading base mpQP solutions from checkpoint...")
        agent_sols_base = load_agent_solutions(base_ckpt)
    else:
        print("\n[3/5] Solving offline mpQP for all agents...")
        t0 = time.perf_counter()
        agent_sols_base = solve_all_agents_mp(
            game,
            algorithm=mpqp_algorithm.combinatorial_parallel,
            verbose=False
        )
        print(f"      -> Completed in {time.perf_counter() - t0:.2f}s")
        save_agent_solutions(agent_sols_base, base_ckpt)

    # %% ── 4. Neighbor Detection — both methods ───────────────────────────────
    print("\n[4/5] Computing neighbor maps (both methods)...")
    from mpgne.cr_store import GNESolution
    import copy

    agent_sols_dict = {}   # "FACET-H" / "FACET-LP" -> agent_sols with neighbors
    facet_sol_dict  = {}   # "FACET-H" / "FACET-LP" -> GNESolution

    for label, method in NB_METHODS.items():
        nb_ckpt     = os.path.join(ckpt_dir, f"agent_sols_{label}_M{M}.pkl")
        facet_ckpt  = os.path.join(ckpt_dir, f"facet_sol_{label}_M{M}.pkl")

        if os.path.exists(nb_ckpt) and os.path.exists(facet_ckpt):
            print(f"  [{label}] Loading from checkpoint...")
            agent_sols_dict[label] = load_agent_solutions(nb_ckpt)
            facet_sol_dict[label]  = load_gne_solution(facet_ckpt)
        else:
            print(f"  [{label}] Building neighbor map (method={method})...")
            # Deep-copy so both methods start from clean mpQP solutions
            sols = copy.deepcopy(agent_sols_base)
            t0 = time.perf_counter()
            find_all_agent_cr_neighbors(sols, method=method, verbose=False)
            nb_counts = [sum(len(cr.facet_neighbors) for cr in s.regions) for s in sols]
            print(f"      -> Done in {time.perf_counter()-t0:.2f}s  "
                  f"neighbor counts/agent: {nb_counts}")
            save_agent_solutions(sols, nb_ckpt)

            if M < OFFLINE_BFS_MAX_M:
                print(f"      -> Building GNE BFS map...")
                t0 = time.perf_counter()
                facet_res = build_gne_solution_facet(game, sols, verbose=False)
                fsol = facet_res.gne_sol
                print(f"      -> BFS done in {time.perf_counter()-t0:.2f}s  "
                      f"({facet_res.n_combos_checked} combos checked)")
            else:
                print(f"      -> Skipping global BFS (M={M} >= {OFFLINE_BFS_MAX_M}). Online only.")
                fsol = GNESolution([], game.n_p, game.N)

            save_gne_solution(fsol, facet_ckpt)
            # Reload to guarantee memory layout
            agent_sols_dict[label] = load_agent_solutions(nb_ckpt)
            facet_sol_dict[label]  = load_gne_solution(facet_ckpt)

    # Use FACET-H as reference agent_sols for ImpGNE (same mpQP, doesn't matter)
    agent_sols_ref = agent_sols_dict["FACET-H"]

    # %% ── 5. Online Closed-Loop Simulation ───────────────────────────────────
    print("\n[5/5] Running closed-loop simulation...")
    from mpgne.facet_gne import solve_gne_online

    x_traj = np.zeros((T_SIM + 1, plant.nx))
    u_traj = {i: np.zeros((T_SIM, plant.subsystems[i].nu)) for i in range(M)}
    x_traj[0] = make_ic(plant, scale=0.4, rng=np.random.default_rng(2025))

    admm_times        = []
    admm_iters        = []
    pg_times          = []
    pg_iters          = []
    impd_times        = []
    impd_iters        = []
    fh_times          = []    # FACET-H times
    flp_times         = []    # FACET-LP times
    fh_fallbacks      = 0
    flp_fallbacks     = 0
    error_gaps        = []
    prev_combo_H      = None
    prev_combo_LP     = None
    admm_conv_hist_k0 = None
    pg_conv_hist_k0   = None
    impd_conv_hist_k0 = None

    # ── Warmup ────────────────────────────────────────────────────────────────
    print(f"  [warmup] qp_solver={QP_SOLVER!r}...")
    _p0 = x_traj[0]
    admm_solve(game, _p0, x_init=None, max_iter=1, tol=1e-20, qp_solver=QP_SOLVER)
    pg_solve(game, _p0, max_iter=1, tol=1e-20, qp_solver=QP_SOLVER)
    impdimc_solve(game, _p0, agent_sols_ref, max_iter=1, tol=1e-20)
    solve_gne_online(_p0, (0,)*M, agent_sols_dict["FACET-H"],  game)
    solve_gne_online(_p0, (0,)*M, agent_sols_dict["FACET-LP"], game)
    import scipy.optimize
    scipy.optimize.linprog(c=[1], A_ub=[[1]], b_ub=[1], bounds=(0,1), method='highs')
    print("  [warmup] Done — starting timed benchmark.")

    def _seed_combo(p, U_admm, agent_sols):
        """Seed initial CR combo from U_admm."""
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
        return tuple(_wc)

    for k in range(T_SIM):
        p = x_traj[k]

        # ── ADMM ──────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        res_admm = admm_solve(game, p, x_init=None, max_iter=2000, tol=1e-4,
                              qp_solver=QP_SOLVER)
        admm_times.append(time.perf_counter() - t0)
        admm_iters.append(res_admm.n_iter)
        U_admm = res_admm.x_stacked
        if k == 0: admm_conv_hist_k0 = res_admm.primal_hist

        # ── Jacobi BR ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        res_pg = pg_solve(game, p, max_iter=2000, tol=1e-4, qp_solver=QP_SOLVER)
        pg_times.append(time.perf_counter() - t0)
        pg_iters.append(res_pg.n_iter)
        if k == 0: pg_conv_hist_k0 = res_pg.conv_hist

        # ── ImpGNE ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        res_impd = impdimc_solve(game, p, agent_sols_ref, max_iter=200, tol=1e-4)
        impd_times.append(time.perf_counter() - t0)
        impd_iters.append(res_impd.n_iter)
        if k == 0: impd_conv_hist_k0 = res_impd.conv_hist

        # ── FACET-H (Hyperplane neighbors) ────────────────────────────────────
        t_start_H = time.perf_counter()
        t_penalty_H = 0.0
        if prev_combo_H is None:
            prev_combo_H = _seed_combo(p, U_admm, agent_sols_dict["FACET-H"])
            if k > 0: t_penalty_H += admm_times[-1]
            
        combo_H, U_H, _ = solve_gne_online(p, prev_combo_H, agent_sols_dict["FACET-H"], game)
        if combo_H is not None:
            U_facet_H    = U_H
            prev_combo_H = combo_H
        else:
            U_facet_H    = U_admm.copy()
            fh_fallbacks += 1
            prev_combo_H  = None
            t_penalty_H += admm_times[-1]
        fh_times.append((time.perf_counter() - t_start_H) + t_penalty_H)

        # ── FACET-LP (LP facet neighbors) ─────────────────────────────────────
        t_start_LP = time.perf_counter()
        t_penalty_LP = 0.0
        if prev_combo_LP is None:
            prev_combo_LP = _seed_combo(p, U_admm, agent_sols_dict["FACET-LP"])
            if k > 0: t_penalty_LP += admm_times[-1]
            
        combo_LP, _, _ = solve_gne_online(p, prev_combo_LP, agent_sols_dict["FACET-LP"], game)
        if combo_LP is not None:
            prev_combo_LP = combo_LP
        else:
            flp_fallbacks += 1
            prev_combo_LP  = None
            t_penalty_LP += admm_times[-1]
        flp_times.append((time.perf_counter() - t_start_LP) + t_penalty_LP)

        error_gaps.append(np.linalg.norm(U_facet_H - U_admm))

        # ── Plant step (use FACET-H as control) ───────────────────────────────
        u_k = {i: U_facet_H[game.x_slice(i)][:plant.subsystems[i].nu] for i in range(M)}
        for i in range(M): u_traj[i][k] = u_k[i]
        x_traj[k + 1] = plant.step(x_traj[k], u_k)

        if (k + 1) % 20 == 0 or k == 0:
            print(f"  step {k+1:3d}/{T_SIM}  "
                  f"ADMM:{res_admm.n_iter:4d}itr  "
                  f"BR:{res_pg.n_iter:4d}itr  "
                  f"ImpGNE:{res_impd.n_iter:3d}itr  "
                  f"FACET-H:{fh_times[-1]*1000:.3f}ms  "
                  f"FACET-LP:{flp_times[-1]*1000:.3f}ms")

    # %% ── 6. Performance Summary ─────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("  PERFORMANCE COMPARISON")
    print("=" * 84)
    print(f"{'Method':<28} | {'Avg Time':>10} | {'Max Time':>10} | {'Avg Iters':>10} | {'Fallbacks':>9}")
    print("-" * 84)
    def _row(name, times, iters=None, fb=None):
        istr = f"{np.mean(iters):>10.1f}" if iters is not None else f"{'N/A':>10}"
        fbstr = f"{fb:>9}" if fb is not None else f"{'—':>9}"
        print(f"{name:<28} | {np.mean(times)*1000:>8.3f} ms | "
              f"{np.max(times)*1000:>8.3f} ms | {istr} | {fbstr}")

    _row("ADMM",       admm_times, admm_iters)
    _row("Jacobi BR",  pg_times,   pg_iters)
    _row("ImpGNE",     impd_times, impd_iters)
    _row("FACET-H  (Hyperplane)", fh_times,  fb=fh_fallbacks)
    _row("FACET-LP (LP Facet)",   flp_times, fb=flp_fallbacks)
    print("-" * 84)
    ref = np.mean(fh_times)
    print(f"  -> FACET-H  speedup vs ADMM    : {np.mean(admm_times)/ref:.1f}x")
    print(f"  -> FACET-H  speedup vs Jacobi  : {np.mean(pg_times)/ref:.1f}x")
    print(f"  -> FACET-H  speedup vs ImpGNE  : {np.mean(impd_times)/ref:.1f}x")
    print(f"  -> FACET-LP speedup vs ADMM    : {np.mean(admm_times)/np.mean(flp_times):.1f}x")
    print(f"  -> FACET-LP vs FACET-H         : {np.mean(fh_times)/np.mean(flp_times):.2f}x")
    print(f"  -> Final state norm            : {np.linalg.norm(x_traj[-1]):.2e}")
    print(f"  -> Avg optimality gap (FACET-H): {np.mean(error_gaps):.2e}")
    print("=" * 84)

    # %% ── 7. Plots ───────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    C_ADMM  = "#E07B54"
    C_PG    = "#5B8DB8"
    C_IMPD  = "#C97BD4"
    C_FH    = "#4CAF82"   # FACET-H  — green
    C_FLP   = "#FFD700"   # FACET-LP — gold

    def _style(ax, title, xlabel, ylabel):
        ax.set_facecolor("#16213e"); ax.tick_params(colors='white')
        ax.spines[:].set_color("#444")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('white')
        ax.set_title(title, color='white', fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel(xlabel, color='white', fontsize=10)
        ax.set_ylabel(ylabel, color='white', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.35, color='#888')

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
    ax1.set_title(f"Distributed MPC via FACET-GNE (M={M} Agents, {_coupling_label})",
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
    p1 = os.path.join(ckpt_dir, "demo_trajectory.png")
    fig1.savefig(p1, dpi=200, bbox_inches='tight', facecolor=fig1.get_facecolor())
    plt.close(fig1)
    print(f"Plot saved: {p1}")

    # Figure 2: Benchmark (2×2)
    fig2 = plt.figure(figsize=(16, 10))
    fig2.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.42, wspace=0.35)

    # [0,0] Convergence at k=0
    ax_c = fig2.add_subplot(gs[0, 0])
    if admm_conv_hist_k0:
        ax_c.semilogy(admm_conv_hist_k0, color=C_ADMM, lw=2,
                      label=f"ADMM ({len(admm_conv_hist_k0)} itr)")
    if pg_conv_hist_k0:
        ax_c.semilogy(pg_conv_hist_k0, color=C_PG, lw=2,
                      label=f"Jacobi BR ({len(pg_conv_hist_k0)} itr)")
    if impd_conv_hist_k0:
        ax_c.semilogy(impd_conv_hist_k0, color=C_IMPD, lw=2, ls='--',
                      label=f"ImpGNE ({len(impd_conv_hist_k0)} itr)")
    ax_c.axhline(1e-4, color='white', ls=':', lw=1, alpha=0.6, label="tol=1e-4")
    _style(ax_c, "Convergence at Step k=0", "Iteration", "Residual / ‖Δx‖")
    ax_c.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', framealpha=0.7)

    # [0,1] Iterations per step
    ax_i = fig2.add_subplot(gs[0, 1])
    ax_i.plot(admm_iters, color=C_ADMM, lw=1.5, label="ADMM")
    ax_i.plot(pg_iters,   color=C_PG,   lw=1.5, label="Jacobi BR")
    ax_i.plot(impd_iters, color=C_IMPD, lw=1.5, ls='--', label="ImpGNE")
    _style(ax_i, "Iterations per Time Step", "Time step $k$", "# Iterations")
    ax_i.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white', framealpha=0.7)

    # [1,0] Solve-time boxplot
    ax_t = fig2.add_subplot(gs[1, 0])
    bp = ax_t.boxplot(
        [np.array(admm_times)*1000, np.array(pg_times)*1000,
         np.array(impd_times)*1000,
         np.array(fh_times)*1000, np.array(flp_times)*1000],
        patch_artist=True, widths=0.5,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(color='#aaa'), capprops=dict(color='#aaa'),
        flierprops=dict(marker='o', color='#aaa', markersize=3),
    )
    for patch, c in zip(bp['boxes'], [C_ADMM, C_PG, C_IMPD, C_FH, C_FLP]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax_t.set_yscale('log')
    ax_t.set_xticks([1, 2, 3, 4, 5])
    ax_t.set_xticklabels(["ADMM", "Jacobi\nBR", "ImpGNE", "FACET-H\n(Hyper)", "FACET-LP\n(LP)"],
                          color='white', fontsize=8)
    _style(ax_t, "Solve-Time Distribution (log scale)", "Method", "Time (ms)")

    # [1,1] Speedup bars vs FACET-H
    ax_s = fig2.add_subplot(gs[1, 1])
    ref_t = np.mean(fh_times)
    su_vals   = [np.mean(admm_times)/ref_t, np.mean(pg_times)/ref_t,
                 np.mean(impd_times)/ref_t, np.mean(flp_times)/ref_t]
    su_labels = ["ADMM\nvs FACET-H", "Jacobi BR\nvs FACET-H",
                 "ImpGNE\nvs FACET-H", "FACET-LP\nvs FACET-H"]
    su_colors = [C_ADMM, C_PG, C_IMPD, C_FLP]
    bars = ax_s.bar(su_labels, su_vals, color=su_colors,
                    alpha=0.8, width=0.5, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, su_vals):
        ax_s.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                  f"{val:.1f}×", ha='center', va='bottom',
                  color='white', fontsize=11, fontweight='bold')
    _style(ax_s, "FACET-H Speedup vs Other Methods", "Comparison", "Speedup (×)")

    fig2.suptitle(
        f"Benchmark: FACET-H vs FACET-LP vs Iterative Solvers  (M={M}, {_coupling_label})",
        color='white', fontsize=13, fontweight='bold', y=1.01)
    p2 = os.path.join(ckpt_dir, "demo_benchmark.png")
    fig2.savefig(p2, dpi=200, bbox_inches='tight', facecolor=fig2.get_facecolor())
    plt.close(fig2)
    print(f"Plot saved: {p2}")
    print(f"  -> All outputs in: {ckpt_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    modes = [("state_bounds", L_MAX), ("l_max", L_MAX)]
    for _mode, _lmax in modes:
        _run_for_mode(_mode, _lmax)
    try:
        plt.show()
    except Exception:
        pass
# %%
