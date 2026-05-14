import time
import warnings
import multiprocessing
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

try:
    import gurobipy as _gp
    from gurobipy import GRB as _GRB
    _HAS_GUROBI = True
except ImportError:
    _HAS_GUROBI = False

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


# ─────────────────────────────────────────────────────────────────────────────
#  Layer 1 — Vectorized LP test (same math, no Python row-loops)
# ─────────────────────────────────────────────────────────────────────────────

def _facet_lp_test_fast(
    E_i: np.ndarray, f_i: np.ndarray,
    E_k: np.ndarray, f_k: np.ndarray,
    j: int,
    tol: float = 1e-6,
) -> bool:
    """
    Vectorized drop-in for `_facet_lp_test`.  Builds A_ub in one numpy block
    operation instead of a Python row-loop — 1.5–2× faster on small LPs where
    matrix construction is a significant fraction of total time.
    """
    n_theta = E_i.shape[1]
    nf_i, nf_k = len(f_i), len(f_k)
    n_var = n_theta + 1
    mask = np.ones(nf_i, dtype=bool); mask[j] = False

    # Block-build A_ub:  [E_i | 0]   (CR_i membership)
    #                    [E_k | 0]   (CR_k membership)
    #                    [E_i[≠j] | 1]  (interior margin for t)
    A_ub = np.empty((nf_i + nf_k + (nf_i - 1), n_var))
    A_ub[:nf_i, :n_theta] = E_i;          A_ub[:nf_i, n_theta] = 0.0
    A_ub[nf_i:nf_i+nf_k, :n_theta] = E_k; A_ub[nf_i:nf_i+nf_k, n_theta] = 0.0
    A_ub[nf_i+nf_k:, :n_theta] = E_i[mask]; A_ub[nf_i+nf_k:, n_theta] = 1.0
    b_ub = np.concatenate([f_i, f_k, f_i[mask]])

    A_eq = np.zeros((1, n_var))
    A_eq[0, :n_theta] = E_i[j]
    b_eq = np.array([f_i[j]])

    c = np.zeros(n_var); c[n_theta] = -1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=[(None, None)] * n_var, method='highs')
    return res.success and (-res.fun) > tol


# ─────────────────────────────────────────────────────────────────────────────
#  Layer 3 — Gurobi LP backend (reuses one Env per worker process)
# ─────────────────────────────────────────────────────────────────────────────

def _facet_lp_test_gurobi(
    env,
    E_i: np.ndarray, f_i: np.ndarray,
    E_k: np.ndarray, f_k: np.ndarray,
    j: int,
    tol: float = 1e-6,
) -> bool:
    """
    Gurobi-backed LP test for facet adjacency.  `env` is a gurobipy.Env
    created once per worker process (passed in from the worker's __init__).
    Uses the matrix addMConstr interface to avoid per-constraint Python overhead.
    Dual simplex (Method=1) is fastest for these dense small LPs.
    """
    import gurobipy as gp
    from gurobipy import GRB

    n_theta = E_i.shape[1]
    nf_i, nf_k = len(f_i), len(f_k)
    n_var = n_theta + 1
    mask = np.ones(nf_i, dtype=bool); mask[j] = False

    A_ub = np.empty((nf_i + nf_k + (nf_i - 1), n_var))
    A_ub[:nf_i, :n_theta] = E_i;          A_ub[:nf_i, n_theta] = 0.0
    A_ub[nf_i:nf_i+nf_k, :n_theta] = E_k; A_ub[nf_i:nf_i+nf_k, n_theta] = 0.0
    A_ub[nf_i+nf_k:, :n_theta] = E_i[mask]; A_ub[nf_i+nf_k:, n_theta] = 1.0
    b_ub = np.concatenate([f_i, f_k, f_i[mask]])

    A_eq = np.zeros((1, n_var))
    A_eq[0, :n_theta] = E_i[j]
    b_eq = np.array([f_i[j]])

    m = gp.Model(env=env)
    x = m.addMVar(n_var, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    m.addMConstr(A_ub, x, GRB.LESS_EQUAL, b_ub)
    m.addMConstr(A_eq, x, GRB.EQUAL,      b_eq)
    obj = np.zeros(n_var); obj[n_theta] = 1.0
    m.setMObjective(None, obj, 0.0, sense=GRB.MAXIMIZE)
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        return m.ObjVal > tol
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  Layer 2 — Cheap Chebyshev-center witness pre-filter
# ─────────────────────────────────────────────────────────────────────────────

def _chebyshev_center(E: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Compute the Chebyshev center of {θ : E θ ≤ f} via
        max r  s.t.  E θ + ‖E_l‖ r ≤ f,  r ≥ 0
    Returns the center θ_c.  Falls back to zero vector on failure.
    """
    n = E.shape[1]
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    A = np.hstack([E, norms])
    c = np.zeros(n + 1); c[-1] = -1.0
    bounds = [(None, None)] * n + [(0.0, None)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = linprog(c, A_ub=A, b_ub=f, bounds=bounds, method='highs')
    if res.success:
        return res.x[:n]
    return np.zeros(n)


def _compute_all_chebyshev_centers(
    agent_solutions: list[AgentSolution],
) -> list[np.ndarray]:
    """
    Return a list (one per agent) of (n_cr × n_theta) arrays of Cheb centers.
    Computed once before Phase 2 begins; cost is O(n_cr) small LPs per agent.
    """
    all_centers = []
    for sol in agent_solutions:
        n_cr = len(sol.regions)
        if n_cr == 0:
            all_centers.append(np.zeros((0, 0)))
            continue
        n_theta = sol.regions[0].E.shape[1]
        centers = np.empty((n_cr, n_theta))
        for k, cr in enumerate(sol.regions):
            centers[k] = _chebyshev_center(cr.E, cr.f)
        all_centers.append(centers)
    return all_centers


def _midpoint_witness(
    E_i: np.ndarray, f_i: np.ndarray, theta_i_c: np.ndarray,
    E_k: np.ndarray, f_k: np.ndarray, theta_k_c: np.ndarray,
    j: int,
    tol: float = 1e-5,
) -> bool:
    """
    Sufficient (one-sided) test for facet adjacency — never produces false
    positives, may miss true adjacencies (falls through to LP in that case).

    Projects the midpoint of the two Chebyshev centers onto the shared
    hyperplane (facet j of CR_i).  If the projection lies strictly inside
    both CRs on all other facets, the intersection is full-dimensional and
    adjacency is confirmed without any LP call.
    """
    e_j = E_i[j]
    e_j_sq = float(e_j @ e_j)
    if e_j_sq < 1e-14:
        return False

    # Project midpoint onto shared hyperplane e_j^T θ = f_i[j]
    theta_mid = 0.5 * (theta_i_c + theta_k_c)
    alpha = (e_j @ theta_mid - float(f_i[j])) / e_j_sq
    theta_p = theta_mid - alpha * e_j

    # CR_i: all facets except j must have strict slack
    viol_i = E_i @ theta_p - f_i
    viol_i[j] = -np.inf          # on the hyperplane by construction — ignore
    if viol_i.max() > -tol:
        return False

    # CR_k: all facets must have strict slack
    if (E_k @ theta_p - f_k).max() > -tol:
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Shared hyperplane finder (extracted from worker for reuse)
# ─────────────────────────────────────────────────────────────────────────────

def _find_shared_hyperplane(
    E_i: np.ndarray, f_i: np.ndarray,
    E_k: np.ndarray, f_k: np.ndarray,
    tol: float = 1e-6,
) -> int | None:
    """
    Return the index j of E_i whose outward normal is anti-parallel to some
    facet of E_k (i.e. they touch the same hyperplane from opposite sides).
    Returns None if no such pair is found.
    """
    norms_i = np.linalg.norm(E_i, axis=1)
    norms_k = np.linalg.norm(E_k, axis=1)
    for j in range(len(f_i)):
        ni = norms_i[j]
        if ni < 1e-14:
            continue
        e_n = E_i[j] / ni
        f_n = f_i[j] / ni
        for l in range(len(f_k)):
            nk = norms_k[l]
            if nk < 1e-14:
                continue
            if (np.linalg.norm(e_n + E_k[l] / nk) < tol and
                    abs(f_n + f_k[l] / nk) < tol):
                return j
    return None


def _facet_lp_refine_worker(in_q, out_q):
    """
    Parallel worker: LP-refine hyperplane-based candidate neighbors.

    Task format (updated):
        (agent_idx, cr_idx, cr_i_E, cr_i_f, cheb_i,
         [(k, cr_k_E, cr_k_f, cheb_k), ...])

    Pipeline per candidate pair:
        1. _find_shared_hyperplane — re-locate facet j (needed for LP)
        2. _midpoint_witness       — cheap sufficient accept (Layer 2, skip LP)
        3. _facet_lp_test_gurobi  — rigorous LP via Gurobi  (Layer 3)
           OR _facet_lp_test_fast — rigorous LP via HiGHS   (Layer 1 fallback)

    One Gurobi Env is created lazily per worker process and reused across all
    LP calls in that worker (amortises license-check latency).
    """
    # Lazy Gurobi env — one per worker process, created on first LP call
    gurobi_env = None

    def _get_gurobi_env():
        nonlocal gurobi_env
        if gurobi_env is not None:
            return gurobi_env
        if not _HAS_GUROBI:
            return None
        try:
            import gurobipy as gp
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.setParam("Threads",    1)
            env.setParam("Method",     1)   # dual simplex — best for small dense LPs
            env.setParam("Presolve",   0)   # skip presolve overhead on tiny problems
            env.start()
            gurobi_env = env
        except Exception:
            gurobi_env = None   # graceful fallback to scipy
        return gurobi_env

    while True:
        task = in_q.get()
        if task is None:
            break

        agent_idx, cr_idx, cr_i_E, cr_i_f, cheb_i, candidate_data = task
        refined = []

        for k, cr_k_E, cr_k_f, cheb_k in candidate_data:
            # Step 1: locate shared hyperplane (needed as j for LP)
            shared_j = _find_shared_hyperplane(cr_i_E, cr_i_f, cr_k_E, cr_k_f)
            if shared_j is None:
                continue

            # Step 2: cheap midpoint-witness accept (Layer 2)
            if _midpoint_witness(cr_i_E, cr_i_f, cheb_i,
                                 cr_k_E, cr_k_f, cheb_k, shared_j):
                refined.append(k)
                continue

            # Step 3: rigorous LP (Layer 3 → gurobi; Layer 1 → fast scipy)
            env = _get_gurobi_env()
            if env is not None:
                ok = _facet_lp_test_gurobi(env, cr_i_E, cr_i_f, cr_k_E, cr_k_f, shared_j)
            else:
                ok = _facet_lp_test_fast(cr_i_E, cr_i_f, cr_k_E, cr_k_f, shared_j)
            if ok:
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
#  Hash-based hyperplane adjacency  (O(N·F) replacement for O(N²·F²) scan)
# ─────────────────────────────────────────────────────────────────────────────

def _find_hyperplane_neighbors_hash(
    agent_solutions: list[AgentSolution],
    decimals: int = 6,
    verbose: bool = True,
) -> int:
    """
    Hash-based hyperplane adjacency detection.

    For each facet (e_ij, f_ij) of every CR we normalize the half-space and
    drop it into a hash bucket keyed by (round(e/‖e‖, d), round(f/‖e‖, d)).
    Two CRs share a hyperplane boundary iff one CR has a facet whose key is
    the *negation* of another's — i.e. the same hyperplane traversed in the
    opposite direction (outward normals are anti-parallel).

    Complexity:  O(N · F) build + O(N · F) lookup
        where N = total CRs and F = avg facets per CR.
    Replaces the prior O(N² · F²) pairwise scan in `_hyperplane_adjacency_worker`.

    Sets `cr.facet_neighbors` in-place on every region. Returns total
    neighbor-pair count.

    The integer-scaled rounding (`* 10**decimals → int64`) avoids any
    float-hash ambiguity. `decimals=6` matches the 1e-6 tolerance used by
    the original pairwise test; over-inclusiveness at rounding cell
    boundaries is harmless because Phase-2 LP refinement filters false
    positives. Single-agent solutions are processed serially — no
    multiprocessing queue, no per-task data duplication.
    """
    t0 = time.perf_counter()
    scale = 10 ** decimals
    total_pairs = 0

    for a_idx, sol in enumerate(agent_solutions):
        n_cr = len(sol.regions)
        bucket: dict[tuple, list[int]] = {}

        # Pass 1: bucket every facet by its signed normalized half-space key.
        for cr_idx, cr in enumerate(sol.regions):
            E = np.asarray(cr.E, dtype=np.float64)
            f = np.asarray(cr.f, dtype=np.float64)
            if E.shape[0] == 0:
                continue
            norms = np.linalg.norm(E, axis=1)
            mask = norms > 1e-14
            if not np.any(mask):
                continue
            inv = np.zeros_like(norms)
            inv[mask] = 1.0 / norms[mask]
            E_int = np.round(E * inv[:, None] * scale).astype(np.int64)
            f_int = np.round(f * inv * scale).astype(np.int64)
            for j in np.flatnonzero(mask):
                key = (E_int[j].tobytes(), int(f_int[j]))
                lst = bucket.get(key)
                if lst is None:
                    bucket[key] = [cr_idx]
                elif lst[-1] != cr_idx:
                    lst.append(cr_idx)

        # Pass 2: for each facet, look up the negated key (opposite-pointing facet).
        neighbors: list[set[int]] = [set() for _ in range(n_cr)]
        for cr_idx, cr in enumerate(sol.regions):
            E = np.asarray(cr.E, dtype=np.float64)
            f = np.asarray(cr.f, dtype=np.float64)
            if E.shape[0] == 0:
                continue
            norms = np.linalg.norm(E, axis=1)
            mask = norms > 1e-14
            if not np.any(mask):
                continue
            inv = np.zeros_like(norms)
            inv[mask] = 1.0 / norms[mask]
            E_neg_int = np.round(-E * inv[:, None] * scale).astype(np.int64)
            f_neg_int = np.round(-f * inv * scale).astype(np.int64)
            for j in np.flatnonzero(mask):
                neg_key = (E_neg_int[j].tobytes(), int(f_neg_int[j]))
                entry = bucket.get(neg_key)
                if entry is None:
                    continue
                for other_idx in entry:
                    if other_idx != cr_idx:
                        neighbors[cr_idx].add(other_idx)

        agent_pairs = 0
        for cr_idx, cr in enumerate(sol.regions):
            cr.facet_neighbors = sorted(neighbors[cr_idx])
            agent_pairs += len(cr.facet_neighbors)
        total_pairs += agent_pairs

        if verbose:
            print(f"  [hash_hp] Agent {a_idx}: {n_cr} CRs → {agent_pairs} neighbor pairs",
                  flush=True)

    if verbose:
        print(f"[hash_hp] Done in {time.perf_counter()-t0:.2f}s | total pairs: {total_pairs}",
              flush=True)
    return total_pairs


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
        # ── Phase 1: Hyperplane adjacency (hash-based, O(N·F)) ───────────────
        if verbose:
            print("[find_neighbors] Phase 1/2: Hyperplane pre-filter (hash-based)...")
        _find_hyperplane_neighbors_hash(agent_solutions, verbose=verbose)
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
        backend = "gurobi" if _HAS_GUROBI else "scipy/highs"
        if verbose:
            print(f"[find_neighbors] Phase 2/2: LP refinement "
                  f"({n_hp} candidates, backend={backend}, witness=on)...")
        t2 = time.perf_counter()

        # Chebyshev centers for the witness pre-filter
        cheb_centers = _compute_all_chebyshev_centers(agent_solutions)

        # Build per-agent fast lookup: cr_idx → (E, f)
        cr_lookup: dict[int, dict[int, tuple]] = {}
        for a_idx, s in enumerate(agent_solutions):
            cr_lookup[a_idx] = {k: (cr.E, cr.f) for k, cr in enumerate(s.regions)}

        n_tasks, _ = _run_worker_pool(
            agent_solutions, _facet_lp_refine_worker,
            task_builder=lambda a_idx, cr_idx, cr, s: (
                a_idx, cr_idx, cr.E, cr.f, cheb_centers[a_idx][cr_idx],
                [(k,
                  cr_lookup[a_idx][k][0], cr_lookup[a_idx][k][1],
                  cheb_centers[a_idx][k])
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
            print("[find_neighbors] method=Hyperplane Adjacency (hash-based, O(N·F))")
        _find_hyperplane_neighbors_hash(agent_solutions, verbose=verbose)

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
    # Snapshot hyperplane-based candidates before LP overwrites them
    hp_neighbors: dict[tuple[int,int], list[int]] = {}
    n_hp = 0
    for a_idx, s in enumerate(agent_solutions):
        for cr_idx, cr in enumerate(s.regions):
            hp_neighbors[(a_idx, cr_idx)] = list(cr.facet_neighbors)
            n_hp += len(cr.facet_neighbors)

    backend = "gurobi" if _HAS_GUROBI else "scipy/highs"
    if verbose:
        print(f"[lp_refine] Refining {n_hp} hyperplane candidates via LP "
              f"(backend={backend}, witness=on)...")

    t0 = time.perf_counter()

    # Layer 2: compute Chebyshev centers once — O(n_cr) small LPs per agent
    if verbose:
        n_crs = sum(len(s.regions) for s in agent_solutions)
        print(f"[lp_refine]   Computing {n_crs} Chebyshev centers...", flush=True)
    t_cheb = time.perf_counter()
    cheb_centers = _compute_all_chebyshev_centers(agent_solutions)
    if verbose:
        print(f"[lp_refine]   Cheb done in {time.perf_counter()-t_cheb:.1f}s", flush=True)

    # Build per-agent fast lookup: cr_idx → (E, f)
    cr_lookup: dict[int, dict[int, tuple]] = {}
    for a_idx, s in enumerate(agent_solutions):
        cr_lookup[a_idx] = {k: (cr.E, cr.f) for k, cr in enumerate(s.regions)}

    _run_worker_pool(
        agent_solutions, _facet_lp_refine_worker,
        task_builder=lambda a_idx, cr_idx, cr, s: (
            a_idx, cr_idx, cr.E, cr.f, cheb_centers[a_idx][cr_idx],
            [(k,
              cr_lookup[a_idx][k][0], cr_lookup[a_idx][k][1],
              cheb_centers[a_idx][k])
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

    # Fast path: check at the centre of the parameter box first.
    # If the polytope contains its centre, it is definitely non-empty — skip LP.
    # e_box = [p_ub; -p_lb], so p_center = 0.5*(p_ub + p_lb).
    n_p = game.n_p
    p_center = 0.5 * (e_box[:n_p] + (-e_box[n_p:]))
    if np.all(D_full @ p_center <= e_full + tol_nonempty):
        return GNECriticalRegion(combination=tuple(combo), D=D_full, e=e_full,
                                 H_x=eq.H_x, h_x=eq.h_x, Mx=Mx, Mp=Mp, M1=M1,
                                 is_unique=eq.is_unique)

    # Slow path: Chebyshev LP to detect non-empty regions that exclude the centre.
    n_full = D_full.shape[1]; nrms = np.linalg.norm(D_full, axis=1, keepdims=True)
    A_lp = np.hstack([D_full, nrms]); b_lp = e_full
    c_lp = np.zeros(n_full + 1); c_lp[-1] = -1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = linprog(c_lp, A_ub=A_lp, b_ub=b_lp, bounds=[(None, None)]*(n_full+1), method='highs')
    if res.success and res.x[-1] > -tol_nonempty:
        return GNECriticalRegion(combination=tuple(combo), D=D_full, e=e_full,
                                 H_x=eq.H_x, h_x=eq.h_x, Mx=Mx, Mp=Mp, M1=M1,
                                 is_unique=eq.is_unique)
    return None

def solve_gne_online(
    p: np.ndarray,
    prev_combo: tuple | None,
    agent_sols: list[AgentSolution],
    game: GNEGame,
    tol_rank: float = 1e-8,
    max_hops: int | None = 1,
    combo_cache: dict | None = None,
):
    """
    Online GNE search via BFS over the combo neighbor graph, with optional
    lazy caching of computed (H_x, h_x) equilibrium maps.

    Two combos are adjacent when they differ in exactly one agent's CR and
    that CR is a FACET neighbor.  BFS explores in order of increasing hop
    distance from prev_combo, so the closest valid combo is returned first.

    Caching behaviour (when combo_cache is supplied):
      * First visit to a combo: solve M_x x* = M_p p + M_1 (fast direct
        solve, no SVD) and store (H_x, h_x).
      * Re-visit: O(1) dict lookup → evaluate x*(p) → membership check.
      * After BFS exhausts, a Tier-2 scan over the CACHE acts as a fallback.

    Validity check (MATLAB-style — no LP, no polytope construction):
      For each agent i: build θ_i = [x_{-i}*(p); p] and check E_i θ_i ≤ f_i.
      This is equivalent to D_crs @ p ≤ e_crs but never requires building the
      p-space polytope — cost is one matvec per agent per combo (~5 µs total).

    Cache miss cost: ~30 µs (linsolve 9×9 + membership check).
    Cache hit cost:  ~5 µs  (matvec + comparison).
    Old cache-miss cost was ~150–300 µs (matrix_rank + SVD + projection).

    Caller is responsible for persisting combo_cache across closed-loop steps
    (typically a single dict shared for the whole simulation run).

    max_hops controls BFS depth: 1, 2, …, or None (all reachable combos).
    If prev_combo is None (cold-start), falls back to full exhaustive search.
    Returns (combo, u_star, combos_checked).
    """
    from collections import deque
    import itertools
    M = game.N
    combos_checked = 0
    use_cache = combo_cache is not None
    tol_mem = 1e-6  # CR membership tolerance

    # ── Pre-build per-agent "other agent" index lists (constant per call) ──
    others_list = [[j for j in range(M) if j != i] for i in range(M)]

    def _membership_ok(combo, x_star):
        """
        Check that x_star is in every agent's CR at the current p.
        θ_i = [x_{-i}*; p],  test E_i θ_i ≤ f_i.
        Pure matvec — no LP, no polytope.
        """
        for i, j_i in enumerate(combo):
            cr = agent_sols[i].regions[j_i]
            x_neg = np.concatenate([x_star[game.x_slice(j)] for j in others_list[i]])
            theta_i = np.concatenate([x_neg, p])
            if np.any(cr.E @ theta_i > cr.f + tol_mem):
                return False
        return True

    def _eval(combo):
        """
        Return (valid, x_star).

        Cache hit  : O(1) lookup + matvec + membership  (~5 µs)
        Cache miss : direct linsolve + membership        (~30 µs)

        Replaces old approach (matrix_rank + SVD + _project_crs_to_p_space
        + D_crs @ p check) which cost ~150–300 µs per miss.
        """
        if use_cache and combo in combo_cache:
            H_x, h_x = combo_cache[combo]
        else:
            Mx, Mp, M1 = _assemble_equilibrium_system(combo, agent_sols, game)
            # Fast path: direct solve (LU, ~5 µs for 9×9).
            # Falls back to SVD-based _solve_equilibrium only on singular Mx.
            try:
                n_x = Mx.shape[0]
                n_p = Mp.shape[1]
                RHS = np.empty((n_x, n_p + 1))
                RHS[:, :n_p] = Mp
                RHS[:, n_p]  = M1
                sol = np.linalg.solve(Mx, RHS)
                H_x = sol[:, :n_p]
                h_x = sol[:, n_p]
            except np.linalg.LinAlgError:
                eq = _solve_equilibrium(Mx, Mp, M1, tol_rank=tol_rank)
                if not eq.solvable:
                    return False, None
                H_x, h_x = eq.H_x, eq.h_x
            if use_cache:
                combo_cache[combo] = (H_x, h_x)

        x_star = H_x @ p + h_x
        if _membership_ok(combo, x_star):
            return True, x_star
        return False, None

    if prev_combo is not None:
        base = tuple(prev_combo)
        seen = {base}
        queue = deque([(base, 0)])

        # ── Tier 1: BFS over the combo neighbor graph ───────────────────────
        while queue:
            combo, depth = queue.popleft()
            combos_checked += 1
            valid, x_star = _eval(combo)
            if valid:
                return combo, x_star, combos_checked

            if max_hops is not None and depth >= max_hops:
                continue

            for i in range(M):
                for nbr in agent_sols[i].regions[combo[i]].facet_neighbors:
                    nxt_t = combo[:i] + (nbr,) + combo[i + 1:]
                    if nxt_t not in seen:
                        seen.add(nxt_t)
                        queue.append((nxt_t, depth + 1))

        # ── Tier 2: scan over CACHED combos not yet visited by BFS ──────────
        # Each check: O(1) dict lookup + matvec + membership — no solve.
        if use_cache:
            for combo in combo_cache:
                if combo in seen:
                    continue
                combos_checked += 1
                H_x, h_x = combo_cache[combo]
                x_star = H_x @ p + h_x
                if _membership_ok(combo, x_star):
                    return combo, x_star, combos_checked

        return None, None, combos_checked

    # Cold-start: full exhaustive search (prev_combo is None)
    for combo in itertools.product(*[range(s.n_cr) for s in agent_sols]):
        combos_checked += 1
        valid, x_star = _eval(combo)
        if valid:
            return combo, x_star, combos_checked

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


# ─────────────────────────────────────────────────────────────────────────────
#  Combo Index — lightweight offline map for M ≥ 4
# ─────────────────────────────────────────────────────────────────────────────

def build_combo_index(
    game: GNEGame,
    agent_solutions: list[AgentSolution],
    seed: tuple[int, ...] | None = None,
    tol_rank: float = 1e-8,
    tol_nonempty: float = 1e-6,
    verbose: bool = True,
) -> tuple[dict, int]:
    """
    Build a lightweight combo index via neighbor-constrained BFS.

    Reuses build_gne_solution_facet (same BFS, same correctness) but discards
    D and e (the p-space polytope constraints) and keeps only:
        combo_index: dict[tuple[int,...] → (H_x, h_x)]

    Storage: ~KB–MB instead of ~GB for M ≥ 4.
    Offline cost: same BFS as build_gne_solution_facet (~minutes for M=4,5).

    Parameters
    ----------
    game             : GNEGame
    agent_solutions  : list[AgentSolution] with facet_neighbors populated
    seed             : optional seed combo (uses origin / parallel search otherwise)
    tol_rank         : rank tolerance for equilibrium solve
    tol_nonempty     : Chebyshev LP threshold for non-empty CR check

    Returns
    -------
    combo_index      : dict[tuple → (H_x np.ndarray, h_x np.ndarray)]
    n_combos_checked : int
    """
    fres = build_gne_solution_facet(
        game, agent_solutions, seed=seed,
        tol_rank=tol_rank, tol_nonempty=tol_nonempty, verbose=verbose,
    )
    combo_index = {
        cr.combination: (cr.H_x.copy(), cr.h_x.copy())
        for cr in fres.gne_sol.regions
    }
    if verbose:
        print(f"[combo_index] {len(combo_index)} feasible combos "
              f"(checked {fres.n_combos_checked} via neighbor-BFS)")
    return combo_index, fres.n_combos_checked


def build_combo_index_lp_from_fh(
    agent_solutions_FLP: list[AgentSolution],
    combo_index_FH: dict,
    seed: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> tuple[dict, int]:
    """
    Derive the FACET-LP combo index from an already-built FACET-H combo index.

    Since FACET-LP neighbors ⊆ FACET-H neighbors, every combo reachable via
    FACET-LP is already in combo_index_FH.  We can therefore build combo_index_FLP
    by doing BFS over FACET-LP neighbor edges and checking membership in
    combo_index_FH — replacing the expensive Chebyshev LP with an O(1) hash lookup.

    Typical speedup: 50–200× vs running a full BFS from scratch.

    Parameters
    ----------
    agent_solutions_FLP : list[AgentSolution] with FACET-LP facet_neighbors
    combo_index_FH      : dict[tuple → (H_x, h_x)] from build_combo_index()
    seed                : starting combo (default: origin tuple of zeros)
    verbose             : print progress

    Returns
    -------
    combo_index_FLP : dict[tuple → (H_x, h_x)]  (subset of combo_index_FH)
    n_checked       : int — combos visited during BFS
    """
    M = len(agent_solutions_FLP)
    t0 = time.perf_counter()

    # Determine seed
    if seed is None:
        seed = tuple(0 for _ in range(M))

    if seed not in combo_index_FH:
        # Try to find any feasible seed reachable via FLP from origin
        if verbose:
            print(f"[lp_from_fh] Origin {seed} not in FH index — "
                  f"scanning FH combos for an FLP-reachable seed...")
        found = False
        for combo in combo_index_FH:
            seed = combo; found = True; break
        if not found:
            if verbose:
                print("[lp_from_fh] combo_index_FH is empty — returning empty index.")
            return {}, 0

    def _lp_neighbors(combo):
        for i, j_i in enumerate(combo):
            for nbr in agent_solutions_FLP[i].regions[j_i].facet_neighbors:
                nxt = list(combo)
                nxt[i] = nbr
                yield tuple(nxt)

    visited = {seed}
    # Only BFS from feasible combos (those in FH); inherit H_x, h_x directly
    queue = deque([seed] if seed in combo_index_FH else [])
    combo_index_FLP = {}
    if seed in combo_index_FH:
        combo_index_FLP[seed] = combo_index_FH[seed]
    n_checked = 0

    while queue:
        combo = queue.popleft()
        n_checked += 1
        for nxt in _lp_neighbors(combo):
            if nxt in visited:
                continue
            visited.add(nxt)
            if nxt in combo_index_FH:          # feasible? → free O(1) lookup
                combo_index_FLP[nxt] = combo_index_FH[nxt]
                queue.append(nxt)              # only expand from feasible combos

        if verbose and n_checked % 10000 == 0:
            print(f"    [lp_from_fh] Visited {n_checked}, "
                  f"found {len(combo_index_FLP)} FLP combos ...", end="\r")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\n[lp_from_fh] Done in {elapsed:.1f}s — "
              f"{len(combo_index_FLP)} FLP combos "
              f"({len(combo_index_FLP)/max(len(combo_index_FH),1)*100:.1f}% of FH) "
              f"from {n_checked} BFS visits")
    return combo_index_FLP, n_checked


def build_gne_solution_lp_from_fh(
    agent_solutions_FLP: list[AgentSolution],
    gne_sol_FH: GNESolution,
    seed: tuple[int, ...] | None = None,
    verbose: bool = True,
) -> tuple[GNESolution, int]:
    """
    Derive the FACET-LP GNESolution from an already-built FACET-H GNESolution.

    Equivalent of build_combo_index_lp_from_fh but returns a full GNESolution
    (with D, e, H_x, h_x per GNE CR) — used for M ≤ OFFLINE_BFS_MAX_M where
    the online lookup uses gne_sol.locate(p) which requires the D, e polytopes.

    Since FACET-LP neighbors ⊆ FACET-H neighbors, every feasible combo reachable
    via FACET-LP BFS is already in gne_sol_FH.  We inherit the full
    GNECriticalRegion objects directly — no Chebyshev LP re-run needed.

    Parameters
    ----------
    agent_solutions_FLP : list[AgentSolution] with FACET-LP facet_neighbors
    gne_sol_FH          : GNESolution from build_gne_solution_facet() with FH maps
    seed                : starting combo (default: origin tuple of zeros)
    verbose             : print progress

    Returns
    -------
    gne_sol_FLP : GNESolution  (subset of gne_sol_FH)
    n_checked   : int — BFS nodes visited
    """
    M = len(agent_solutions_FLP)
    n_p = gne_sol_FH.n_p
    t0 = time.perf_counter()

    # Build a fast lookup: combo tuple → GNECriticalRegion from FH solution
    fh_cr_map = {cr.combination: cr for cr in gne_sol_FH.regions}

    if seed is None:
        seed = tuple(0 for _ in range(M))

    if seed not in fh_cr_map:
        if verbose:
            print(f"[lp_from_fh] Origin {seed} not in FH solution — "
                  f"using first available FH combo as seed...")
        if not fh_cr_map:
            if verbose:
                print("[lp_from_fh] FH solution is empty — returning empty GNESolution.")
            return GNESolution(regions=[], n_p=n_p, N=M), 0
        seed = next(iter(fh_cr_map))

    def _lp_neighbors(combo):
        for i, j_i in enumerate(combo):
            for nbr in agent_solutions_FLP[i].regions[j_i].facet_neighbors:
                nxt = list(combo)
                nxt[i] = nbr
                yield tuple(nxt)

    visited = {seed}
    queue = deque([seed] if seed in fh_cr_map else [])
    flp_regions = []
    if seed in fh_cr_map:
        flp_regions.append(fh_cr_map[seed])
    n_checked = 0

    while queue:
        combo = queue.popleft()
        n_checked += 1
        for nxt in _lp_neighbors(combo):
            if nxt in visited:
                continue
            visited.add(nxt)
            if nxt in fh_cr_map:               # feasible? → free O(1) lookup
                flp_regions.append(fh_cr_map[nxt])
                queue.append(nxt)              # only expand from feasible combos

        if verbose and n_checked % 5000 == 0:
            print(f"    [lp_from_fh] Visited {n_checked}, "
                  f"found {len(flp_regions)} FLP CRs ...", end="\r")

    elapsed = time.perf_counter() - t0
    gne_sol_FLP = GNESolution(regions=flp_regions, n_p=n_p, N=M)
    if verbose:
        print(f"\n[lp_from_fh] Done in {elapsed:.1f}s — "
              f"{len(flp_regions)} FLP GNE CRs "
              f"({len(flp_regions)/max(len(gne_sol_FH.regions),1)*100:.1f}% of FH) "
              f"from {n_checked} BFS visits")
    return gne_sol_FLP, n_checked



def solve_gne_online_ci(
    p: np.ndarray,
    prev_combo: tuple | None,
    agent_sols: list[AgentSolution],
    combo_index: dict,
    game: GNEGame,
    tol: float = 1e-6,
    max_hops: int | None = 2,
    exhaustive_fallback: bool = True,
) -> tuple:
    """
    Online GNE search using the precomputed combo index (O(1) hash lookup).

    Two-tier search (both purely local — zero communication rounds):
      Tier 1: BFS over the combo neighbor graph up to ``max_hops``.
              Two combos are adjacent when they differ in exactly one
              agent's CR and that CR is a FACET neighbor.  Yields combos
              in order of increasing hop distance, so the closest valid
              combo is returned first.
      Tier 2: Exhaustive scan over every entry in ``combo_index`` not
              already visited by BFS.  Catches valid combos that BFS
              missed due to hop limit or graph-disconnection.  Only runs
              if ``exhaustive_fallback=True``.

    Returns (None, None, n_checked) only when no entry in ``combo_index``
    matches the current state — caller should treat this as ADMM fallback.

    Approximate candidate counts (M=4, avg_nb=12 neighbors/CR):
      1-hop:        ~49 candidates
      2-hop:        ~1 440 candidates
      3-hop:        ~17 000 candidates
      None:         all FACET-reachable combos via BFS
      Exhaustive:   all |combo_index| entries (~98 000 for M=4)
    Each check is a single dict lookup (~1 µs).

    Parameters
    ----------
    p                  : current parameter / state vector
    prev_combo         : combo from previous step (None → cold-start)
    agent_sols         : per-agent AgentSolution with facet_neighbors populated
    combo_index        : dict[tuple → (H_x, h_x)] from build_combo_index()
    game               : GNEGame
    tol                : agent CR constraint tolerance
    max_hops           : BFS depth — integer, or None for all reachable
    exhaustive_fallback: if True, scan full combo_index when BFS exhausts

    Returns
    -------
    (combo, x_star, n_checked)  if a valid combo is found
    (None,  None,   n_checked)  if NO entry in combo_index is valid → ADMM
    """
    from collections import deque

    M = game.N
    n_checked = 0

    if prev_combo is None:
        return None, None, 0

    def _check(combo):
        """Return (valid, x_star) for one combo. Assumes combo ∈ combo_index."""
        H_x, h_x = combo_index[combo]
        x_star = H_x @ p + h_x
        for i, j_i in enumerate(combo):
            cr = agent_sols[i].regions[j_i]
            others = [j for j in range(M) if j != i]
            x_neg = np.concatenate([x_star[game.x_slice(j)] for j in others])
            theta_i = np.concatenate([x_neg, p])
            if np.any(cr.E @ theta_i > cr.f + tol):
                return False, x_star
        return True, x_star

    # ── Tier 1: BFS over the combo neighbor graph ────────────────────────────
    base = tuple(prev_combo)
    seen = {base}
    queue = deque([(base, 0)])

    while queue:
        combo, depth = queue.popleft()

        n_checked += 1
        if combo in combo_index:
            valid, x_star = _check(combo)
            if valid:
                return combo, x_star, n_checked

        if max_hops is not None and depth >= max_hops:
            continue

        for i in range(M):
            for nbr in agent_sols[i].regions[combo[i]].facet_neighbors:
                nxt_t = combo[:i] + (nbr,) + combo[i + 1:]
                if nxt_t not in seen:
                    seen.add(nxt_t)
                    queue.append((nxt_t, depth + 1))

    # ── Tier 2: exhaustive scan over the full combo_index ────────────────────
    if exhaustive_fallback:
        for combo in combo_index:
            if combo in seen:
                continue
            n_checked += 1
            valid, x_star = _check(combo)
            if valid:
                return combo, x_star, n_checked

    return None, None, n_checked


# ─────────────────────────────────────────────────────────────────────────────
#  MATLAB V2-style online solver — no BFS, per-agent filter + combo check
# ─────────────────────────────────────────────────────────────────────────────

# Module-level Gurobi env — created once, reused across all online LP calls.
_GUROBI_ONLINE_ENV = None


def _get_online_gurobi_env():
    """Lazy singleton Gurobi env for online Chebyshev LP filter calls."""
    global _GUROBI_ONLINE_ENV
    if _GUROBI_ONLINE_ENV is not None:
        return _GUROBI_ONLINE_ENV
    if not _HAS_GUROBI:
        return None
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads",    1)
        env.setParam("Method",     1)   # dual simplex — fastest for small dense LPs
        env.setParam("Presolve",   0)   # skip presolve on tiny problems
        env.start()
        _GUROBI_ONLINE_ENV = env
    except Exception:
        _GUROBI_ONLINE_ENV = None
    return _GUROBI_ONLINE_ENV


def _chebyshev_feasible(env, A_eff: np.ndarray, b_eff: np.ndarray,
                        tol: float = 1e-6) -> bool:
    """
    Check if {U : A_eff U ≤ b_eff} is non-empty via the Chebyshev LP.

    Matches MATLAB IF_mpDiMPC_V2 online filter exactly.

    Formulation (variables [U (n_u,), r (1,)]):
        max   r
        s.t.  A_eff[l,:] @ U  +  ‖A_eff[l,:]‖  * r  ≤  b_eff[l]   ∀l
              r ≥ 0

    r* > tol  ↔  the polytope has a ball of radius r* fitting inside
              ↔  the CR is genuinely feasible at the current p  → KEEP.
    r* ≤ tol  ↔  empty or measure-zero projection at current p  → DISCARD.

    Uses Gurobi (env given) or scipy/HiGHS (env=None fallback).
    ~10–50 µs per call with Gurobi on the small LPs we see here.
    """
    n_c = A_eff.shape[0]
    if n_c == 0:
        return True   # unconstrained U space → trivially non-empty

    n_u    = A_eff.shape[1]
    norms  = np.linalg.norm(A_eff, axis=1, keepdims=True).clip(min=1e-14)
    A_cheb = np.hstack([A_eff, norms])   # (n_c, n_u + 1)

    if env is not None:
        import gurobipy as gp
        from gurobipy import GRB
        m   = gp.Model(env=env)
        lb  = np.full(n_u + 1, -GRB.INFINITY)
        lb[-1] = 0.0                           # r ≥ 0
        x   = m.addMVar(n_u + 1, lb=lb, ub=GRB.INFINITY)
        m.addMConstr(A_cheb, x, GRB.LESS_EQUAL, b_eff)
        obj = np.zeros(n_u + 1); obj[-1] = 1.0
        m.setMObjective(None, obj, 0.0, sense=GRB.MAXIMIZE)
        m.optimize()
        # OPTIMAL   → check r* > tol (non-empty interior)
        # UNBOUNDED → feasible set is non-empty (and unbounded) → keep
        # INFEASIBLE / other → empty → discard
        if m.Status == GRB.OPTIMAL:
            return m.ObjVal > tol
        return m.Status == GRB.UNBOUNDED
    else:
        c      = np.zeros(n_u + 1); c[-1] = -1.0
        bounds = [(None, None)] * n_u + [(0.0, None)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = linprog(c, A_ub=A_cheb, b_ub=b_eff,
                          bounds=bounds, method='highs')
        # status 3 = unbounded in HiGHS → feasible
        if res.status == 3:
            return True
        return res.success and (-res.fun) > tol


def precompute_point_location_arrays(agent_solutions: list) -> None:
    """
    Precompute padded E/f stacked arrays on each AgentSolution for vectorized
    PointLocation.  Call once after loading or computing agent solutions.

    Sets two attributes on each AgentSolution:
        _E_stack : ndarray (n_cr, max_ineq, n_theta)  — padded E matrices
        _f_stack : ndarray (n_cr, max_ineq)           — padded f vectors
    Padding uses E=0, f=+inf so padded rows produce -inf violations and never
    influence argmin / feasibility checks.

    Memory: ~n_cr × max_ineq × n_theta × 8 bytes per agent
            (≈ 27 MB for 2820 CRs × 80 constraints × 15 params).
    Cost during PointLocation: one batched matmul (n_cr, max_ineq, n_theta)@(n_theta,)
            instead of n_cr Python-loop iterations — 50–200× faster for large n_cr.
    """
    for sol in agent_solutions:
        if not sol.regions:
            continue
        n_cr    = len(sol.regions)
        n_theta = sol.regions[0].E.shape[1]
        max_ineq = max(cr.E.shape[0] for cr in sol.regions)

        E_stack = np.zeros((n_cr, max_ineq, n_theta), dtype=np.float64)
        f_stack = np.full((n_cr, max_ineq), np.inf,  dtype=np.float64)

        for k, cr in enumerate(sol.regions):
            n = cr.E.shape[0]
            E_stack[k, :n, :] = cr.E
            f_stack[k, :n]    = cr.f

        sol._E_stack = E_stack
        sol._f_stack = f_stack


def _locate_cr_fast(agent_sol, theta: np.ndarray, tol: float = 1e-6,
                    hint_idx: int | None = None) -> int:
    """
    Find which CR of agent_sol contains theta (point-location).

    Fast path  — vectorized batched matmul when _E_stack/_f_stack exist:
        violations = max(E_stack @ theta - f_stack, axis=1)  →  shape (n_cr,)
        One numpy call replaces the entire Python for-loop.
        ~50–200× faster than the loop for large n_cr (e.g. M=4 with 2820 CRs).

    Warm-start — checked first if hint_idx is given:
        Tests hint CR and its ~50 facet neighbors before the full scan.
        Returns immediately on a hit (O(n_neighbors) ≈ O(50)).
        Falls through to vectorized scan only on a miss.

    Cold-start (hint_idx=None, no _E_stack): original Python for-loop fallback.
    """
    # ── Warm-start: check hint + its 1-hop neighbors ──────────────────────────
    if hint_idx is not None:
        for idx in [hint_idx] + list(agent_sol.regions[hint_idx].facet_neighbors):
            cr = agent_sol.regions[idx]
            if float(np.max(cr.E @ theta - cr.f)) <= tol:
                return idx

    # ── Vectorized path: one batched matmul over all CRs ─────────────────────
    if hasattr(agent_sol, '_E_stack'):
        # (n_cr, max_ineq, n_theta) @ (n_theta,) → (n_cr, max_ineq)
        # np.max over axis=1 → (n_cr,)  violations per CR
        violations = np.max(agent_sol._E_stack @ theta - agent_sol._f_stack, axis=1)
        feasible = np.where(violations <= tol)[0]
        if len(feasible):
            return int(feasible[0])
        return int(np.argmin(violations))

    # ── Fallback: Python loop (only if stacked arrays not precomputed) ────────
    best_idx, best_viol = 0, np.inf
    for idx, cr in enumerate(agent_sol.regions):
        viol = float(np.max(cr.E @ theta - cr.f))
        if viol <= tol:
            return idx
        if viol < best_viol:
            best_viol, best_idx = viol, idx
    return best_idx


def solve_gne_online_v2(
    p: np.ndarray,
    agent_sols: list[AgentSolution],
    game: GNEGame,
    prev_x_star: np.ndarray | None = None,
    tol: float = 1e-6,
    prev_crs: list[int] | None = None,
    combo_cache: dict | None = None,
) -> tuple:
    """
    MATLAB V2-style online GNE search — scored neighbor search + optional cache.

    Mirrors IF_mpDiMPC_V2 exactly:
      1. PointLocation: for each agent i find current CR using prev_x_star
         as the reference U_{-i}.  When prev_crs is given, warm-starts by
         checking that CR + its facet neighbors first — O(n_neighbors) ≈ O(50)
         instead of O(n_cr) full scan.  Critical for M≥4 (2000+ CRs/agent).
      2. Per-agent filter: from 1-hop facet neighbors, keep only those where
         U_{-i}_prev still satisfies the CR constraints after marginalising p.
         (point check — no LP, just A_eff @ U_{-i}_prev ≤ b_eff)
      3. Enumerate product of per-agent filtered sets (~3×3×3 = 27 combos).
      4. For each combo: direct linsolve + per-agent membership check.

    Cost per step (warm, with prev_crs hint):
      PointLocation: O(n_neighbors) per agent  — warm path ≈ O(50)
      Filter:        O(n_neighbors × n_constraints) per agent
      Combo check:   O(n_x²) linsolve + O(M × n_constraints)  — ~30 µs each
      Typical total: < 2 ms even for M=4 with 2000+ CRs per agent

    Parameters
    ----------
    p            : current state / parameter (n_p,)
    agent_sols   : list of AgentSolution with facet_neighbors populated
    game         : GNEGame
    prev_x_star  : stacked U* from the previous step (n_x_total,).
                   Used as the U_{-i} reference for PointLocation and filter.
                   Pass None at cold-start (uses zeros — seed with ADMM first).
    tol          : CR membership tolerance
    prev_crs     : list of per-agent CR indices from the previous step
                   (i.e. the winning combo from the last solve_gne_online_v2
                   call).  Used as warm-start hints for PointLocation.
                   Pass None at cold-start or after an ADMM fallback.
    combo_cache  : optional dict {tuple → (H_x, h_x)} persisted across steps.
                   When provided:
                     • Every newly validated combo is stored in the cache.
                     • If the scored-sort search (Step B3) finds nothing, a
                       Tier-2 scan over the cache is performed before returning
                       None — saving the ADMM fallback in most cases.
                   Pass an empty dict `{}` from the caller and keep it alive
                   across the full simulation; do NOT reset it after fallbacks.

    Returns
    -------
    (combo, x_star, combos_checked)
    combo and x_star are None when no valid combo found in either scored set
    or cache (caller should ADMM-fallback in that case).
    """
    import itertools
    M = game.N
    others_list = [[j for j in range(M) if j != i] for i in range(M)]

    if prev_x_star is None:
        prev_x_star = np.zeros(game.n_x_total)

    # ── Step 1: PointLocation — find current CR for each agent ────────────
    # Warm-start: pass prev_crs[i] as hint so _locate_cr_fast checks the
    # previous CR + its ~50 neighbors before falling back to a full scan.
    current_crs = []
    for i in range(M):
        U_neg_i = np.concatenate([prev_x_star[game.x_slice(j)] for j in others_list[i]])
        theta_i = np.concatenate([U_neg_i, p])
        hint = prev_crs[i] if (prev_crs is not None and i < len(prev_crs)) else None
        current_crs.append(_locate_cr_fast(agent_sols[i], theta_i, tol, hint_idx=hint))

    # ── Step 2: Per-agent Chebyshev LP filter (exact MATLAB IF_mpDiMPC_V2) ──
    #
    # Mirrors MATLAB lines 41-75 exactly:
    #   for each agent i:
    #     candidates = [located_CR] + its 1-hop facet neighbors
    #     for each candidate j:
    #       b_eff = CR.b - CR.A[:, state_cols] @ p      (marginalize p/state)
    #       A_eff = CR.A[:, u_neg_cols]                  (U_{-i} columns only)
    #       keep j  iff  Chebyshev(A_eff, b_eff)  is feasible (r* > tol)
    #
    # The Chebyshev LP asks: does {U_{-i} : A_eff U_{-i} ≤ b_eff} have any
    # interior at the current p?  Feasible → CR is reachable → keep it.
    # Empty projection → CR cannot contribute to a GNE here → discard it.
    #
    # No artificial cap, no scoring, no hard threshold — identical to MATLAB.
    # Uses Gurobi (10–50 µs per LP) or scipy/HiGHS as fallback.
    gurobi_env = _get_online_gurobi_env()

    feasible_per_agent = []
    for i in range(M):
        n_x_neg = game.n_x_total - game.agents[i].n_x

        candidates = [current_crs[i]] + list(
            agent_sols[i].regions[current_crs[i]].facet_neighbors)

        filtered = []
        for cr_idx in candidates:
            cr    = agent_sols[i].regions[cr_idx]
            b_eff = cr.f - cr.E[:, n_x_neg:] @ p   # marginalize p (state cols)
            A_eff = cr.E[:, :n_x_neg]               # U_{-i} columns only
            if _chebyshev_feasible(gurobi_env, A_eff, b_eff, tol):
                filtered.append(cr_idx)

        # Safety: PointLocation guarantees current CR is always feasible.
        # If LP somehow filtered everything out, fall back to current CR only.
        feasible_per_agent.append(filtered if filtered else [current_crs[i]])

    # ── Step 3+4: Enumerate filtered combos, direct solve + membership ────
    combos_checked = 0
    n_x = game.n_x_total
    n_p = game.n_p

    for combo in itertools.product(*feasible_per_agent):
        combos_checked += 1

        # Fast direct solve (no SVD, no rank check — typical case is full-rank)
        Mx, Mp, M1 = _assemble_equilibrium_system(combo, agent_sols, game)
        try:
            RHS = np.empty((n_x, n_p + 1))
            RHS[:, :n_p] = Mp
            RHS[:, n_p]  = M1
            sol  = np.linalg.solve(Mx, RHS)
            H_x  = sol[:, :n_p]
            h_x  = sol[:, n_p]
        except np.linalg.LinAlgError:
            continue

        x_star = H_x @ p + h_x

        # Per-agent membership check (MATLAB: A_CR * theta ≤ b_CR)
        valid = True
        for i, j_i in enumerate(combo):
            cr    = agent_sols[i].regions[j_i]
            x_neg = np.concatenate([x_star[game.x_slice(j)] for j in others_list[i]])
            theta_i = np.concatenate([x_neg, p])
            if np.any(cr.E @ theta_i > cr.f + tol):
                valid = False
                break

        if valid:
            if combo_cache is not None:
                combo_cache[combo] = (H_x.copy(), h_x.copy())
            return combo, x_star, combos_checked

    # ── Tier-2: scan combo_cache for any valid combo missed by scored search ──
    # Cost: ~5 µs per cached entry (one matvec + membership check, no linsolve).
    # Fires only when the 1-hop scored set misses the correct combo — avoids
    # the ADMM fallback in cases where the trajectory revisits an earlier combo.
    if combo_cache:
        checked_in_b3 = set(itertools.product(*feasible_per_agent))
        for combo, (H_x, h_x) in combo_cache.items():
            if combo in checked_in_b3:
                continue
            combos_checked += 1
            x_star = H_x @ p + h_x
            valid = True
            for i, j_i in enumerate(combo):
                cr = agent_sols[i].regions[j_i]
                x_neg = np.concatenate([x_star[game.x_slice(j)] for j in others_list[i]])
                theta_i = np.concatenate([x_neg, p])
                if np.any(cr.E @ theta_i > cr.f + tol):
                    valid = False
                    break
            if valid:
                return combo, x_star, combos_checked

    return None, None, combos_checked

