import time
import warnings
import multiprocessing
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .game import GNEGame
from .cr_store import AgentCR, AgentSolution, GNECriticalRegion, GNESolution
from .gne_combiner import (
    _assemble_equilibrium_system,
    _solve_equilibrium,
    _project_crs_to_p_space,
    _cr_nonempty,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Geometric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _has_opposite_boundary(e, f, cr_k, tol=1e-6):
    e_norm = e / (np.linalg.norm(e) + 1e-14)
    f_s = f / (np.linalg.norm(e) + 1e-14)
    for l in range(cr_k.n_ineq):
        ek = cr_k.E[l]; nrm = np.linalg.norm(ek) + 1e-14
        if np.linalg.norm(e_norm + ek/nrm) < tol and abs(f_s + cr_k.f[l]/nrm) < tol:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Part 1a — Hyperplane Adjacency worker (fast, over-inclusive)
#
#  Two regions R_i and R_k are declared neighbors if they share any common
#  supporting hyperplane, i.e. there exist facets j of R_i and l of R_k such
#  that  e_{i,j} / ‖e_{i,j}‖ = −e_{k,l} / ‖e_{k,l}‖  and
#         f_{i,j} / ‖e_{i,j}‖ = −f_{k,l} / ‖e_{k,l}‖.
#  This is a *necessary* condition for sharing a (d-1)-dimensional facet, but
#  not sufficient — the intersection could be lower-dimensional.
#  Name in ACC 2026 paper: "Hyperplane Adjacency"
# ─────────────────────────────────────────────────────────────────────────────

def _hyperplane_adjacency_worker(in_q, out_q):
    """Parallel worker: hyperplane-adjacency test for one CR against all others."""
    while True:
        task = in_q.get()
        if task is None:
            break
        agent_idx, cr_idx, cr_i_E, cr_i_f, other_crs_data = task
        neighbors = []
        for k, (cr_k_E, cr_k_f) in enumerate(other_crs_data):
            if k == cr_idx:
                continue
            found = False
            for j in range(len(cr_i_f)):
                e_ij = cr_i_E[j]
                f_ij = cr_i_f[j]
                e_norm = e_ij / (np.linalg.norm(e_ij) + 1e-14)
                f_s    = f_ij / (np.linalg.norm(e_ij) + 1e-14)
                for l in range(len(cr_k_f)):
                    ek  = cr_k_E[l]
                    nrm = np.linalg.norm(ek) + 1e-14
                    if (np.linalg.norm(e_norm + ek / nrm) < 1e-6 and
                            abs(f_s + cr_k_f[l] / nrm) < 1e-6):
                        neighbors.append(k)
                        found = True
                        break
                if found:
                    break
        out_q.put((agent_idx, cr_idx, neighbors))


# ─────────────────────────────────────────────────────────────────────────────
#  Part 1b — Facet Adjacency worker (rigorous LP, compact neighbor sets)
#
#  For each pair of CRs (R_i, R_k) that pass the hyperplane test, solve the LP:
#
#      max  t
#      s.t. E_i θ ≤ f_i
#           E_k θ ≤ f_k
#           e_{i,j}^T θ = f_{i,j}            (on shared hyperplane)
#           t ≤ f_{i,l} − e_{i,l}^T θ,  ∀ l ≠ j   (interior margin)
#
#  R_i and R_k are facet-adjacent iff  t* > ε,  i.e. the shared intersection
#  has positive (d-1)-dimensional volume (a true facet, not just an edge/point).
#  Name in ACC 2026 paper: "Facet Adjacency"
# ─────────────────────────────────────────────────────────────────────────────

def _facet_lp_test(
    E_i: np.ndarray, f_i: np.ndarray,
    E_k: np.ndarray, f_k: np.ndarray,
    j: int,
    tol: float = 1e-6,
) -> bool:
    """
    LP test: is the intersection of R_i ∩ R_k on facet j of R_i full-dimensional?

    Solves (with variable [θ; t]):
        max  t
        s.t. E_i θ ≤ f_i
             E_k θ ≤ f_k
             e_{i,j}^T θ = f_{i,j}
             t ≤ f_{i,l} - e_{i,l}^T θ  for all l ≠ j

    Returns True iff t* > tol.
    """
    n_theta  = E_i.shape[1]
    nf_i     = len(f_i)
    n_var    = n_theta + 1          # [θ; t]

    # Collect inequality rows: E_i θ ≤ f_i
    rows_A, rows_b = [], []
    for l in range(nf_i):
        row = np.zeros(n_var)
        row[:n_theta] = E_i[l]
        rows_A.append(row); rows_b.append(f_i[l])

    # E_k θ ≤ f_k
    for l in range(len(f_k)):
        row = np.zeros(n_var)
        row[:n_theta] = E_k[l]
        rows_A.append(row); rows_b.append(f_k[l])

    # Margin constraints:  t ≤ f_{i,l} - e_{i,l}^T θ  ↔  e_{i,l}^T θ + t ≤ f_{i,l}  (for l ≠ j)
    for l in range(nf_i):
        if l == j:
            continue
        row = np.zeros(n_var)
        row[:n_theta] = E_i[l]
        row[n_theta]  = 1.0           # +t
        rows_A.append(row); rows_b.append(f_i[l])

    A_ub = np.array(rows_A)
    b_ub = np.array(rows_b)

    # Equality: e_{i,j}^T θ = f_{i,j}
    A_eq = np.zeros((1, n_var))
    A_eq[0, :n_theta] = E_i[j]
    b_eq = np.array([f_i[j]])

    # Objective: max t  ↔  min -t
    c = np.zeros(n_var)
    c[n_theta] = -1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=[(None, None)] * n_var, method='highs')

    return res.success and (-res.fun) > tol


def _facet_lp_refine_worker(in_q, out_q):
    """Parallel worker: LP-refine hyperplane-based candidate neighbors.

    Receives *only* the candidate pairs that already passed the hyperplane
    pre-filter (Phase 1).  Re-finds which shared hyperplane they match on
    and runs the rigorous LP test to confirm true (d-1)-dimensional adjacency.
    """
    while True:
        task = in_q.get()
        if task is None:
            break
        agent_idx, cr_idx, cr_i_E, cr_i_f, candidate_data = task
        # candidate_data: list of (k_idx, cr_k_E, cr_k_f) — hyperplane neighbors only
        refined = []
        for k, cr_k_E, cr_k_f in candidate_data:
            # Re-find which hyperplane they share (cheap, we need shared_j for LP)
            shared_j = None
            for j in range(len(cr_i_f)):
                e_ij   = cr_i_E[j]
                f_ij   = cr_i_f[j]
                e_norm = e_ij / (np.linalg.norm(e_ij) + 1e-14)
                f_s    = f_ij / (np.linalg.norm(e_ij) + 1e-14)
                for l in range(len(cr_k_f)):
                    ek  = cr_k_E[l]
                    nrm = np.linalg.norm(ek) + 1e-14
                    if (np.linalg.norm(e_norm + ek / nrm) < 1e-6 and
                            abs(f_s + cr_k_f[l] / nrm) < 1e-6):
                        shared_j = j
                        break
                if shared_j is not None:
                    break

            if shared_j is not None and _facet_lp_test(cr_i_E, cr_i_f, cr_k_E, cr_k_f, shared_j):
                refined.append(k)

        out_q.put((agent_idx, cr_idx, refined))


def _facet_adjacency_worker(in_q, out_q):
    """Parallel worker: facet-adjacency LP test for one CR against all others.
    (Legacy — kept for direct use; two-phase path via _facet_lp_refine_worker is
    preferred when hyperplane results are already available.)"""
    while True:
        task = in_q.get()
        if task is None:
            break
        agent_idx, cr_idx, cr_i_E, cr_i_f, other_crs_data = task
        neighbors = []

        for k, (cr_k_E, cr_k_f) in enumerate(other_crs_data):
            if k == cr_idx:
                continue

            # Step 1: cheap hyperplane pre-filter
            shared_j = None
            for j in range(len(cr_i_f)):
                e_ij   = cr_i_E[j]
                f_ij   = cr_i_f[j]
                e_norm = e_ij / (np.linalg.norm(e_ij) + 1e-14)
                f_s    = f_ij / (np.linalg.norm(e_ij) + 1e-14)
                for l in range(len(cr_k_f)):
                    ek  = cr_k_E[l]
                    nrm = np.linalg.norm(ek) + 1e-14
                    if (np.linalg.norm(e_norm + ek / nrm) < 1e-6 and
                            abs(f_s + cr_k_f[l] / nrm) < 1e-6):
                        shared_j = j
                        break
                if shared_j is not None:
                    break

            if shared_j is None:
                continue          # no shared hyperplane at all — skip LP

            # Step 2: rigorous LP test (ACC 2026, Section III)
            if _facet_lp_test(cr_i_E, cr_i_f, cr_k_E, cr_k_f, shared_j):
                neighbors.append(k)

        out_q.put((agent_idx, cr_idx, neighbors))


# ─────────────────────────────────────────────────────────────────────────────
#  Shared multiprocessing pool helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_worker_pool(agent_solutions, worker_fn, task_builder):
    """Generic multiprocessing pool: queue tasks, collect results, update CRs.

    task_builder(agent_idx, cr_idx, cr, agent_solutions[a_idx]) -> tuple
        Returns the task tuple to put on the input queue for this CR.
    """
    t0 = time.perf_counter()
    in_q  = multiprocessing.Queue()
    out_q = multiprocessing.Queue()
    num_workers = multiprocessing.cpu_count()
    workers = [
        multiprocessing.Process(target=worker_fn, args=(in_q, out_q))
        for _ in range(num_workers)
    ]
    for w in workers:
        w.start()

    total_tasks = 0
    for a_idx, s in enumerate(agent_solutions):
        for cr_idx, cr in enumerate(s.regions):
            task = task_builder(a_idx, cr_idx, cr, s)
            in_q.put(task)
            total_tasks += 1

    for _ in range(total_tasks):
        a_idx, cr_idx, neighbors = out_q.get()
        agent_solutions[a_idx].regions[cr_idx].facet_neighbors = neighbors

    for _ in range(num_workers):
        in_q.put(None)
    for w in workers:
        w.join()

    return total_tasks, time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
#  Unified public API
# ─────────────────────────────────────────────────────────────────────────────

def find_all_agent_cr_neighbors(
    agent_solutions: list[AgentSolution],
    method: str = "hyperplane_adjacency",
    verbose: bool = True,
) -> list[AgentSolution]:
    """
    Detect neighboring critical regions for every agent offline.

    Parameters
    ----------
    agent_solutions : list of AgentSolution (one per agent)
    method          : neighbor-detection strategy —

        "hyperplane_adjacency"  (fast, over-inclusive)
            Two CRs are neighbors if they share any supporting hyperplane
            (necessary but not sufficient for a shared facet).  Produces
            larger neighbor sets; faster to compute offline; may require
            slightly more online hops.

        "facet_adjacency"  (rigorous, compact — ACC 2026 paper)
            Two-phase: (1) hyperplane pre-filter to find candidate pairs,
            then (2) LP refinement on candidates only.  Produces exact,
            compact neighbor sets.

    verbose : print timing summary

    Returns
    -------
    agent_solutions  (modified in-place, facet_neighbors populated)
    """
    if method in ("facet_adjacency", "facet"):
        # ── Phase 1: Hyperplane adjacency (fast pre-filter) ──────────────────
        if verbose:
            print("[find_neighbors] Phase 1/2: Hyperplane pre-filter...")
        _run_worker_pool(
            agent_solutions, _hyperplane_adjacency_worker,
            task_builder=lambda a_idx, cr_idx, cr, s: (
                a_idx, cr_idx, cr.E, cr.f,
                [(c.E, c.f) for c in s.regions]
            ),
        )
        # Save hyperplane-based candidates before LP overwrites them
        hp_neighbors = {}
        n_hp = 0
        for a_idx, s in enumerate(agent_solutions):
            for cr_idx, cr in enumerate(s.regions):
                hp_neighbors[(a_idx, cr_idx)] = list(cr.facet_neighbors)
                n_hp += len(cr.facet_neighbors)

        if verbose:
            print(f"[find_neighbors] Phase 1 done — {n_hp} candidate pairs")

        # ── Phase 2: LP refinement on candidates only ────────────────────────
        if verbose:
            print(f"[find_neighbors] Phase 2/2: LP refinement ({n_hp} LPs)...")
        t2 = time.perf_counter()
        # Build per-agent fast lookup: cr_idx → (E, f)
        cr_lookup = {}
        for a_idx, s in enumerate(agent_solutions):
            cr_lookup[a_idx] = {k: (cr.E, cr.f) for k, cr in enumerate(s.regions)}

        n_tasks, _ = _run_worker_pool(
            agent_solutions, _facet_lp_refine_worker,
            task_builder=lambda a_idx, cr_idx, cr, s: (
                a_idx, cr_idx, cr.E, cr.f,
                [(k, cr_lookup[a_idx][k][0], cr_lookup[a_idx][k][1])
                 for k in hp_neighbors.get((a_idx, cr_idx), [])]
            ),
        )
        elapsed_2 = time.perf_counter() - t2

        if verbose:
            n_refined = sum(len(cr.facet_neighbors)
                           for s in agent_solutions for cr in s.regions)
            print(f"[find_neighbors] Phase 2 done in {elapsed_2:.1f}s "
                  f"| {n_refined} facet neighbors "
                  f"(filtered {n_hp - n_refined} false positives)")

    else:   # default: hyperplane_adjacency
        if verbose:
            print("[find_neighbors] method=Hyperplane Adjacency (fast)")
        _run_worker_pool(
            agent_solutions, _hyperplane_adjacency_worker,
            task_builder=lambda a_idx, cr_idx, cr, s: (
                a_idx, cr_idx, cr.E, cr.f,
                [(c.E, c.f) for c in s.regions]
            ),
        )
        if verbose:
            total_nb = sum(len(cr.facet_neighbors)
                           for s in agent_solutions for cr in s.regions)
            print(f"[find_neighbors] Done | total neighbor pairs: {total_nb}")

    return agent_solutions


def refine_neighbors_with_lp(
    agent_solutions: list[AgentSolution],
    verbose: bool = True,
) -> list[AgentSolution]:
    """
    Refine existing hyperplane-based facet_neighbors via rigorous LP test.

    Takes agent solutions whose facet_neighbors are already populated by
    hyperplane adjacency and replaces them with LP-verified facet neighbors.
    This avoids re-running the O(n_cr²) hyperplane scan — use when FACET-H
    results are already available.

    Parameters
    ----------
    agent_solutions : list of AgentSolution with facet_neighbors populated
    verbose         : print timing

    Returns
    -------
    agent_solutions  (modified in-place)
    """
    # Snapshot hyperplane-based candidates
    hp_neighbors = {}
    n_hp = 0
    for a_idx, s in enumerate(agent_solutions):
        for cr_idx, cr in enumerate(s.regions):
            hp_neighbors[(a_idx, cr_idx)] = list(cr.facet_neighbors)
            n_hp += len(cr.facet_neighbors)

    if verbose:
        print(f"[lp_refine] Refining {n_hp} hyperplane candidates via LP...")

    t0 = time.perf_counter()

    # Build per-agent fast lookup: cr_idx → (E, f)
    cr_lookup = {}
    for a_idx, s in enumerate(agent_solutions):
        cr_lookup[a_idx] = {k: (cr.E, cr.f) for k, cr in enumerate(s.regions)}

    _run_worker_pool(
        agent_solutions, _facet_lp_refine_worker,
        task_builder=lambda a_idx, cr_idx, cr, s: (
            a_idx, cr_idx, cr.E, cr.f,
            [(k, cr_lookup[a_idx][k][0], cr_lookup[a_idx][k][1])
             for k in hp_neighbors.get((a_idx, cr_idx), [])]
        ),
    )

    elapsed = time.perf_counter() - t0
    n_refined = sum(len(cr.facet_neighbors)
                    for s in agent_solutions for cr in s.regions)

    if verbose:
        print(f"[lp_refine] Done in {elapsed:.1f}s "
              f"| {n_refined} facet neighbors "
              f"(filtered {n_hp - n_refined} false positives)")

    return agent_solutions


# ─────────────────────────────────────────────────────────────────────────────
#  Part 2 — Parallel Seed Finding & Sequential BFS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FacetGNEResult:
    gne_sol: GNESolution
    n_combos_checked: int
    n_combos_total: int
    reduction_ratio: float
    used_fallback: bool
    elapsed: float

def _process_combo_kernel(combo, agent_sols, game, tol_rank, tol_nonempty, D_box, e_box):
    Mx, Mp, M1 = _assemble_equilibrium_system(combo, agent_sols, game)
    eq = _solve_equilibrium(Mx, Mp, M1, tol_rank=tol_rank)
    if not eq.solvable: return None
    D_crs, e_crs = _project_crs_to_p_space(combo, agent_sols, game, eq.H_x, eq.h_x)
    D_full = np.vstack([D_crs, D_box]); e_full = np.concatenate([e_crs, e_box])
    n_p = D_full.shape[1]; nrms = np.linalg.norm(D_full, axis=1, keepdims=True)
    A_lp = np.hstack([D_full, nrms]); b_lp = e_full
    c_lp = np.zeros(n_p + 1); c_lp[-1] = -1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = linprog(c_lp, A_ub=A_lp, b_ub=b_lp, bounds=[(None, None)]*(n_p+1), method='highs')
    if res.success and res.x[-1] > -tol_nonempty:
        return GNECriticalRegion(combination=tuple(combo), D=D_full, e=e_full, H_x=eq.H_x, h_x=eq.h_x, Mx=Mx, Mp=Mp, M1=M1, is_unique=eq.is_unique)
    return None

def solve_gne_online(p: np.ndarray, prev_combo: tuple | None, agent_sols: list[AgentSolution], game: GNEGame, tol_rank: float = 1e-8):
    """
    Online search for GNE combination via 1-hop neighbor walk.

    If prev_combo is given, checks:
      1. The current combo (same as previous step)
      2. All combos where exactly ONE agent moves to a neighbor CR
    This keeps the per-step search O(sum of neighbor counts) — sub-millisecond.

    Falls back to ADMM (via returning None) if the 1-hop neighborhood is exhausted.
    Falls back to full exhaustive search if prev_combo is None (cold-start).
    Returns (combo, u_star, combos_checked).
    """
    import itertools
    M = game.N
    combos_checked = 0

    if prev_combo is not None:
        warm_tup = tuple(prev_combo)

        def local_search():
            # 0-hop: same combo as previous step
            yield warm_tup
            # 1-hop: exactly one agent moves to a neighbor CR
            for i in range(M):
                v_i = warm_tup[i]
                for nbr in agent_sols[i].regions[v_i].facet_neighbors:
                    nxt = list(warm_tup)
                    nxt[i] = nbr
                    yield tuple(nxt)

        for combo in local_search():
            combos_checked += 1
            Mx, Mp, M1 = _assemble_equilibrium_system(combo, agent_sols, game)
            eq = _solve_equilibrium(Mx, Mp, M1, tol_rank=tol_rank)
            if not eq.solvable:
                continue
            D_crs, e_crs = _project_crs_to_p_space(combo, agent_sols, game, eq.H_x, eq.h_x)
            if np.all(D_crs @ p <= e_crs + 1e-6):
                u_star = eq.H_x @ p + eq.h_x
                return combo, u_star, combos_checked

        # 1-hop exhausted → ADMM fallback
        return None, None, combos_checked

    # Full exhaustive search (cold-start only, prev_combo is None)
    for combo in itertools.product(*[range(s.n_cr) for s in agent_sols]):
        combos_checked += 1
        Mx, Mp, M1 = _assemble_equilibrium_system(combo, agent_sols, game)
        eq = _solve_equilibrium(Mx, Mp, M1, tol_rank=tol_rank)
        if not eq.solvable:
            continue
        D_crs, e_crs = _project_crs_to_p_space(combo, agent_sols, game, eq.H_x, eq.h_x)
        if np.all(D_crs @ p <= e_crs + 1e-6):
            u_star = eq.H_x @ p + eq.h_x
            return combo, u_star, combos_checked

    return None, None, combos_checked

def _seed_worker(in_q, out_q, agent_sols, game, tol_rank, tol_nonempty, D_box, e_box):
    while True:
        try: combo = in_q.get(timeout=1)
        except: break
        if combo is None: break
        res = _process_combo_kernel(combo, agent_sols, game, tol_rank, tol_nonempty, D_box, e_box)
        if res: out_q.put((combo, res))

def _bfs_worker_init(agent_sols_in, game_in, tol_rank_in, tol_nonempty_in, D_box_in, e_box_in):
    global _G_AGENT_SOLS, _G_GAME, _G_TOL_RANK, _G_TOL_NONEMPTY, _G_D_BOX, _G_E_BOX
    _G_AGENT_SOLS = agent_sols_in
    _G_GAME = game_in
    _G_TOL_RANK = tol_rank_in
    _G_TOL_NONEMPTY = tol_nonempty_in
    _G_D_BOX = D_box_in
    _G_E_BOX = e_box_in

def _bfs_process_combo(combo):
    return _process_combo_kernel(combo, _G_AGENT_SOLS, _G_GAME, _G_TOL_RANK, _G_TOL_NONEMPTY, _G_D_BOX, _G_E_BOX)

def build_gne_solution_facet(
    game: GNEGame, agent_solutions: list[AgentSolution], seed: tuple[int, ...] | None = None,
    tol_rank: float = 1e-8, tol_nonempty: float = 1e-6, verbose: bool = True,
) -> FacetGNEResult:
    N = game.N; n_p = game.n_p; t0 = time.perf_counter()
    n_total = 1
    for s in agent_solutions: n_total *= s.n_cr
    total_nb = sum(len(cr.facet_neighbors) for s in agent_solutions for cr in s.regions)
    use_fallback = (total_nb == 0)
    
    if verbose: print(f"\n[facet_gne] N={N}, n_p={n_p} → {n_total} combinations")
    D_box = np.vstack([np.eye(n_p), -np.eye(n_p)]); e_box = np.concatenate([game.p_ub, -game.p_lb])
    gne_regions = []; n_checked = 0

    if not use_fallback:
        import itertools
        valid_seed = seed
        
        if valid_seed is None:
            # Quick check for origin first
            origin = tuple(0 for _ in range(N))
            if verbose: print(f"    [FACET BFS] Checking origin {origin}...")
            cr_origin = _process_combo_kernel(origin, agent_solutions, game, tol_rank, tol_nonempty, D_box, e_box)
            if cr_origin:
                valid_seed = origin
                gne_regions.append(cr_origin)
                if verbose: print(f"    [FACET BFS] Seed found at Origin!")
            else:
                if verbose: print("    [FACET BFS] Searching for initial seed (Parallel)...")
                in_q = multiprocessing.Queue(); out_q = multiprocessing.Queue()
                num_workers = multiprocessing.cpu_count()
                workers = [multiprocessing.Process(target=_seed_worker, args=(in_q, out_q, agent_solutions, game, tol_rank, tol_nonempty, D_box, e_box)) for _ in range(num_workers)]
                for w in workers: w.start()
                
                cr_index_lists = [list(range(s.n_cr)) for s in agent_solutions]
                combo_gen = itertools.product(*cr_index_lists)
                found_seed_data = None
                
                while found_seed_data is None and n_checked < n_total:
                    for _ in range(1000):
                        try: 
                            c = next(combo_gen); in_q.put(c); n_checked += 1
                        except StopIteration: break
                    time.sleep(0.01)
                    while not out_q.empty():
                        found_seed_data = out_q.get(); break
                    if verbose and n_checked % 2000 == 0:
                        print(f"    [FACET BFS] Seed Search: Checked {n_checked}/{n_total} combinations...", end="\r")
                    if n_checked >= n_total: break
                
                for _ in range(num_workers): in_q.put(None)
                for w in workers: w.terminate()
                if found_seed_data:
                    valid_seed, cr_k = found_seed_data
                    gne_regions.append(cr_k)
                    if verbose: print(f"\n    [FACET BFS] Seed found: {valid_seed} (after {n_checked} checks)")

        if valid_seed is None:
            if verbose: print("\n    [FACET BFS] WARNING: No valid GNE CRs found!")
            return FacetGNEResult(GNESolution([], n_p, N), n_checked, n_total, 1.0, False, time.perf_counter()-t0)

        # Parallel BFS from Seed
        visited = {valid_seed}
        current_level = [valid_seed]
        
        def _facet_adjacent_combos(combo, agent_solutions):
            for i, j_i in enumerate(combo):
                for j_new in agent_solutions[i].regions[j_i].facet_neighbors:
                    yield tuple(j_new if k == i else combo[k] for k in range(len(combo)))

        num_workers = multiprocessing.cpu_count()
        if verbose: print(f"    [FACET BFS] Starting Parallel BFS with {num_workers} workers...")
        pool = multiprocessing.Pool(
            processes=num_workers,
            initializer=_bfs_worker_init,
            initargs=(agent_solutions, game, tol_rank, tol_nonempty, D_box, e_box)
        )
        
        # Add neighbors of seed to the first level (seed already evaluated)
        next_level = []
        for nxt in _facet_adjacent_combos(valid_seed, agent_solutions):
            if nxt not in visited: 
                visited.add(nxt)
                next_level.append(nxt)
        current_level = next_level
        
        while current_level:
            results = pool.map(_bfs_process_combo, current_level)
            n_checked += len(current_level)
            
            next_level = []
            for combo, cr_k in zip(current_level, results):
                if cr_k is not None:
                    gne_regions.append(cr_k)
                    for nxt in _facet_adjacent_combos(combo, agent_solutions):
                        if nxt not in visited:
                            visited.add(nxt)
                            next_level.append(nxt)
                            
            current_level = next_level
            if verbose:
                print(f"    [FACET BFS] Checked {n_checked} total, found {len(gne_regions)} GNE CRs, next level: {len(current_level)}   ", end="\r")
                
        pool.close()
        pool.join()
        if verbose: print()
    else:
        # Exhaustive logic...
        pass

    elapsed = time.perf_counter() - t0
    gne_sol = GNESolution(regions=gne_regions, n_p=n_p, N=N)
    return FacetGNEResult(gne_sol, n_checked, n_total, n_checked/max(n_total,1), use_fallback, elapsed)
