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
#  Part 1 — Parallel Neighbor Detection
# ─────────────────────────────────────────────────────────────────────────────

def _region_neighbor_worker(in_q, out_q):
    while True:
        task = in_q.get()
        if task is None: break
        agent_idx, cr_idx, cr_i_E, cr_i_f, other_crs_data = task
        neighbors = []
        for k, (cr_k_E, cr_k_f) in enumerate(other_crs_data):
            if k == cr_idx: continue
            found = False
            for j in range(len(cr_i_f)):
                e_ij = cr_i_E[j]; f_ij = cr_i_f[j]
                e_norm = e_ij / (np.linalg.norm(e_ij) + 1e-14)
                f_s = f_ij / (np.linalg.norm(e_ij) + 1e-14)
                for l in range(len(cr_k_f)):
                    ek = cr_k_E[l]; nrm = np.linalg.norm(ek) + 1e-14
                    if np.linalg.norm(e_norm + ek/nrm) < 1e-6 and abs(f_s + cr_k_f[l]/nrm) < 1e-6:
                        neighbors.append(k); found = True; break
                if found: break
        out_q.put((agent_idx, cr_idx, neighbors))

def find_all_agent_cr_neighbors(
    agent_solutions: list[AgentSolution],
    method: str = "hyperplane",
    verbose: bool = True,
) -> list[AgentSolution]:
    t0 = time.perf_counter()
    in_q = multiprocessing.Queue(); out_q = multiprocessing.Queue()
    num_workers = multiprocessing.cpu_count()
    workers = [multiprocessing.Process(target=_region_neighbor_worker, args=(in_q, out_q)) for _ in range(num_workers)]
    for w in workers: w.start()
    total_tasks = 0
    for a_idx, s in enumerate(agent_solutions):
        other_crs_data = [(cr.E, cr.f) for cr in s.regions]
        for cr_idx, cr in enumerate(s.regions):
            in_q.put((a_idx, cr_idx, cr.E, cr.f, other_crs_data)); total_tasks += 1
    for _ in range(total_tasks):
        a_idx, cr_idx, neighbors = out_q.get()
        agent_solutions[a_idx].regions[cr_idx].facet_neighbors = neighbors
    for _ in range(num_workers): in_q.put(None)
    for w in workers: w.join()
    if verbose:
        print(f"[facet_gne] Parallel Neighbor Search complete in {time.perf_counter()-t0:.2f}s")
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
    
    # Use 'highs' for better performance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = linprog(c_lp, A_ub=A_lp, b_ub=b_lp, bounds=[(None, None)]*(n_p+1), method='highs')
        
    if res.success and res.x[-1] > -tol_nonempty:
        return GNECriticalRegion(combination=tuple(combo), D=D_full, e=e_full, H_x=eq.H_x, h_x=eq.h_x, Mx=Mx, Mp=Mp, M1=M1, is_unique=eq.is_unique)
    return None

def solve_gne_online(p: np.ndarray, prev_combo: tuple | None, agent_sols: list[AgentSolution], game: GNEGame, tol_rank: float = 1e-8):
    """
    Online search for GNE combination. Checks restricted sets if prev_combo is given,
    then falls back to full search.
    Returns (combo, u_star, combos_checked).
    """
    import itertools
    M = game.N
    combos_checked = 0
    
    # 1. Restricted Search (Hop-based to avoid explosion)
    if prev_combo is not None:
        warm_tup = tuple(prev_combo)
        
        def local_search():
            yield warm_tup
            # 1-hop (change one agent)
            for i in range(M):
                v_warm = warm_tup[i]
                for nbr in agent_sols[i].regions[v_warm].facet_neighbors:
                    nxt = list(warm_tup)
                    nxt[i] = nbr
                    yield tuple(nxt)

        for combo in local_search():
            combos_checked += 1
            Mx, Mp, M1 = _assemble_equilibrium_system(combo, agent_sols, game)
            eq = _solve_equilibrium(Mx, Mp, M1, tol_rank=tol_rank)
            if not eq.solvable: continue
            
            D_crs, e_crs = _project_crs_to_p_space(combo, agent_sols, game, eq.H_x, eq.h_x)
            if np.all(D_crs @ p <= e_crs + 1e-6):
                u_star = eq.H_x @ p + eq.h_x
                return combo, u_star, combos_checked
                
        # If not found within 2 hops, return None immediately to use ADMM 
        # (Full search is too slow for online GNE)
        return None, None, combos_checked
                
    # 2. Full Search
    search_sets = [range(s.n_cr) for s in agent_sols]
    search_iter = itertools.product(*search_sets)
    
    for combo in search_iter:
        combos_checked += 1
        Mx, Mp, M1 = _assemble_equilibrium_system(combo, agent_sols, game)
        eq = _solve_equilibrium(Mx, Mp, M1, tol_rank=tol_rank)
        if not eq.solvable: continue
        
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
