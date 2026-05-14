# mpGNE-PPOPT — Project Handoff

> **Multi-Parametric Generalized Nash Equilibrium via PPOPT**
> ACC 2026 / CCE Case Study — Explicit Distributed MPC for Multi-Agent Systems

---

## 1. Project Overview

This framework solves **multi-agent GNE (Generalized Nash Equilibrium)** problems for distributed
MPC systems with coupled constraints. Each agent solves its own multi-parametric QP (mpQP) offline,
producing a piecewise-affine solution map. The core contribution is the **FACET** method — a
neighbor-based approach that assembles agent-level critical regions into joint GNE maps, enabling
sub-millisecond online control.

### Key Innovation: FACET-V2 Online Solver (Exact MATLAB IF_mpDiMPC_V2)

For M > OFFLINE_BFS_MAX_M (state_bounds: M≥3, l_max: M≥4), an explicit p-space GNE map is
infeasible. The **V2 online solver** (`solve_gne_online_v2`) replicates MATLAB's IF_mpDiMPC_V2
line-for-line:

| MATLAB IF_mpDiMPC_V2_3s.m | Python solve_gne_online_v2 |
|---|---|
| `PointLocation(Solution_i, theta_i)` | Vectorized batched matmul over all CRs |
| `temp_i = Solution_i(temp_i).boundary1` | `[current_cr] + facet_neighbors` |
| `b = CR.b - CR.A(:,state_cols)*p` | `b_eff = cr.f - cr.E[:, n_x_neg:] @ p` |
| `A = CR.A(:, u_cols)` | `A_eff = cr.E[:, :n_x_neg]` |
| `[x,~] = Chebyshev(A, b)` | `_chebyshev_feasible(gurobi_env, A_eff, b_eff)` |
| `if all(~isnan(x)) → keep` | `if feasible → keep CR` |
| `simul_solutions_3s(...)` | `itertools.product(*feasible_per_agent)` + linsolve |
| Fallback to `I_mpDiMPC_3s` | ADMM fallback (25 iters) |

**Three additional Python improvements beyond base MATLAB:**
1. **Vectorized PointLocation** — `precompute_point_location_arrays` stacks all CRs into a padded
   3-D array. One batched `numpy` matmul replaces an O(n_cr) Python loop (50–200× faster for M=4).
2. **Warm hint (`prev_crs`)** — checks the previous step's CR and its neighbors first; returns
   early in O(n_neighbors) ≈ O(50) without touching the full batched matmul.
3. **Combo cache** — persistent `{combo → (H_x, h_x)}` dict across steps. When the 1-hop search
   fails, a Tier-2 cache scan (~5 µs/entry, ~50 entries) fires before ADMM fallback.

---

## 2. Repository Structure

```
mpgne_ppopt/
├── mpgne/                          # Core library
│   ├── __init__.py
│   ├── plant.py                    # Plant / Subsystem dynamical models
│   ├── plant_gen.py                # Random plant generation
│   ├── game.py                     # Agent, GNEGame data structures
│   ├── mpc_builder.py              # Builds GNEGame from Plant
│   ├── mp_solver.py                # Per-agent mpQP solve via PPOPT
│   ├── cr_store.py                 # AgentCR, AgentSolution, GNECriticalRegion, GNESolution
│   ├── gne_combiner.py             # Equilibrium system assembly + projection
│   ├── facet_gne.py                # ★ FACET neighbor detection + V2 online solver
│   ├── admm_solver.py              # ADMM iterative baseline
│   ├── proj_grad_solver.py         # Jacobi Best-Response baseline
│   ├── impdimc_solver.py           # Implicit DiMPC (ImpGNE) baseline
│   └── centralized_solver.py       # Centralized SLSQP (reference)
│
├── tests/
│   ├── full_case_study.py          # ★ Main benchmark script (Tables 1–4, plots)
│   ├── bench_state_bounds.py       # ★ Quick benchmark for state_bounds M=2,3
│   ├── full_case_study_data_state_bounds/  # state_bounds checkpoints
│   ├── full_case_study_data_l_max/         # l_max checkpoints
│   ├── bench_state_bounds_ckpt/            # state_bounds M=2,3 quick-bench checkpoints
│   └── test_*.py                           # Unit tests
│
├── demo_single_run.py
├── requirements.txt
└── Handoff.md
```

---

## 3. Complete Pipeline — Step by Step

This section is the single authoritative reference for every step, both offline and online.
All steps are listed in execution order. **Nothing is skipped.**

---

### OFFLINE STEPS (done once per plant, results checkpointed)

---

#### Step 1 — Plant Generation
**Function:** `make_random_plants(M, N, seed=...)`
**File:** `mpgne/plant_gen.py`

Generates N random stable LTI plants each with M coupled subsystems.
Per subsystem: nx_i=2 states, nu_i=1 input, Np=3 prediction horizon.
Stability enforced by resampling A_i until spectral radius < 1.
State bounds x_lb ∈ [-100,-10], x_ub ∈ [10,100]; input bounds u_lb ∈ [-5,-1], u_ub ∈ [1,5].

**Same for l_max and state_bounds** — plant generation is coupling-agnostic.

---

#### Step 2 — Game Construction
**Function:** `make_gne_game_from_plant(plant, coupling_mode=...)`
**File:** `mpgne/mpc_builder.py`

Expands the plant over the prediction horizon (Np=3) and builds a `GNEGame` object.
Computes prediction matrices Φ_x and Γ_j, local cost matrices Q_i/R_i/P_i, and encodes
the coupling constraint into each agent's `Agent` object.

**THIS IS WHERE l_max AND state_bounds DIVERGE:**

| | l_max | state_bounds |
|---|---|---|
| Coupling | Σ u_{j,k} ≤ L_max per step | x_lb ≤ Φ_x p + Σ Γ_j U_j ≤ x_ub |
| Stored in | `Agent.C`, `GNEGame.d`, `S_coup` | `Agent.Gamma_self`, `M_theta`, `x_lb_rep`, `x_ub_rep` |
| Type | Standard Nash (shared constraint) | Generalized Nash (feasible set depends on others) |

---

#### Step 3 — Per-Agent mpQP Solve
**Function:** `solve_all_agents_mp(game, algorithm=...)`
**File:** `mpgne/mp_solver.py`

For each agent i, builds and solves a multi-parametric QP using PPOPT:

```
min_{U_i}  ½ U_i^T Q_i U_i  +  (H_i θ_i)^T U_i  +  c_i^T U_i
s.t.       G_i U_i  ≤  b_i  +  F_i θ_i
           θ_i ∈ [θ_min, θ_max]   (box on parameter space)
```

where θ_i = [U_{-i}; p] is the parameter vector (other agents' decisions + state).

PPOPT returns a piecewise-affine solution: for each Critical Region (CR) j,
the optimal control is `U_i*(θ_i) = A_j θ_i + b_j` when `E_j θ_i ≤ f_j`.

**THIS IS THE BIGGEST DIFFERENCE between l_max and state_bounds:**

| | l_max | state_bounds |
|---|---|---|
| Constraints per agent | ~9 (6 input + 3 coupling) | ~42 (6 input + 36 state bounds) |
| Max CRs ≈ C(n_con, n_dec) | C(9,3)=84 → ~100 CRs | C(42,3)=11480 → ~2000 CRs |
| Offline solve time | ~1-5 min/agent | ~5-30 min/agent |

The 36 state bound constraints (2 × Np × nx_total = 2×3×6=36) all depend on θ_i
through the dense M_θ matrix — this is why state_bounds has 20× more CRs.

Saves: `M{M}_plant{idx}_agent_sols_base.pkl`

---

#### Step 4a — FACET-H Neighbor Detection (Hyperplane Adjacency)
**Function:** `find_all_agent_cr_neighbors(agent_sols, method="hyperplane_adjacency")`
**File:** `mpgne/facet_gne.py`

For each agent i, detects which pairs of CRs share a common facet boundary.

**Algorithm (hash-based, O(N×F)):**
1. For each CR, for each facet: normalize the half-space (e/‖e‖, f/‖e‖) → integer key
2. Two CRs are hyperplane-adjacent if one has a facet whose key is the NEGATION of the other's
   (anti-parallel normals = shared hyperplane)
3. Sets `cr.facet_neighbors` for every CR

Over-inclusive filter (necessary but not sufficient for true facet sharing).
Equivalent to MATLAB's `boundaryExplorerNew`.

**l_max:** ~100 CRs/agent, avg ~5-15 hyperplane neighbors/CR.
**state_bounds:** ~2000 CRs/agent, avg ~50-80 hyperplane neighbors/CR.

Saves: `M{M}_plant{idx}_agent_sols_FH.pkl`

---

#### Step 4b — FACET-LP Neighbor Refinement
**Function:** `refine_neighbors_with_lp(agent_sols_FH)`
**File:** `mpgne/facet_gne.py`

Refines the over-inclusive hyperplane neighbor list to true facet neighbors via LP.

**Algorithm (per candidate pair from Step 4a):**
1. Find the shared hyperplane index j (anti-parallel normal check)
2. Chebyshev midpoint witness test (fast: no LP, ~1 µs) — if passes, confirmed adjacent
3. If inconclusive: LP test — maximize slack on shared facet while satisfying both CRs
   (t* > ε means full-dimensional intersection = true facet)
4. Uses Gurobi (fast), falls back to scipy HiGHS

Result: `cr.facet_neighbors` replaced with LP-verified exact facet neighbors (~10/CR for state_bounds).
FACET-LP produces compact neighbor sets → fewer Chebyshev LPs per online step → faster online.

Saves: `M{M}_plant{idx}_agent_sols_FLP.pkl`

---

#### Step 5 — Offline BFS Explicit GNE Map (M ≤ OFFLINE_BFS_MAX_M only)
**Function:** `build_gne_solution_facet(game, agent_sols_FH)`
**File:** `mpgne/facet_gne.py`

Builds the complete explicit p-space GNE map by BFS over the combo neighbor graph.
Each node is a combo (j_1,...,j_M) — one CR index per agent.
Two combos are adjacent if they differ in exactly one agent's CR and that CR is a facet neighbor.

**Algorithm per combo:**
1. `_assemble_equilibrium_system`: build Mx, Mp, M1 from agents' affine maps
2. `_solve_equilibrium`: solve for H_x, h_x (direct `np.linalg.solve`, falls back to SVD)
3. `_project_crs_to_p_space`: substitute x*(p) into CR constraints → D p ≤ e
4. **Fast check:** if D p_center ≤ e at the box center → include without LP
5. If fast check fails: Chebyshev LP to verify {p: D p ≤ e} is non-empty
6. Store valid `GNECriticalRegion(D, e, H_x, h_x)`

**CRITICAL DIFFERENCE:**

| | l_max | state_bounds |
|---|---|---|
| CRs/agent | ~100 | ~2000 |
| Total combos (M=3) | 100³=1M | 2000³=8B |
| BFS-reachable combos (M=3) | ~2000-5000 | ~125,000 |
| Offline BFS time (M=3) | ~5-30 min | ~15 min (but result is 1 GB!) |
| Feasible for M=2? | Yes | Yes (~1416 GNE CRs, ~3.8 MB) |
| Feasible for M=3? | Yes | **No** — 125K GNE CRs = 1 GB file |

**OFFLINE_BFS_MAX_M:**
- `l_max`: 3 (M=3 explicit map is feasible)
- `state_bounds`: **2** — never raise above 2 for state_bounds

Saves: `M{M}_plant{idx}_facet_sol_FH.pkl`

---

#### Step 5b — FACET-LP Explicit Map from FH (M ≤ OFFLINE_BFS_MAX_M only)
**Function:** `build_gne_solution_lp_from_fh(agent_sols_FLP, gne_sol_FH)`
**File:** `mpgne/facet_gne.py`

BFS over FACET-LP neighbor edges + O(1) hash lookup in existing FH map.
Since FLP neighbors ⊆ FH neighbors, every FLP-reachable combo is already in gne_sol_FH.
Speedup: 50–200× vs building FLP map independently.

Saves: `M{M}_plant{idx}_facet_sol_FLP.pkl`

---

### ONLINE STEPS (per time step k, during closed-loop simulation)

---

#### MODE A — Explicit Map Lookup (M ≤ OFFLINE_BFS_MAX_M)

Used when `facet_sol` is provided (from Step 5).

**Per step k:**
```
p = x(k)
cr_idx = facet_sol.locate(p)     ← scan stored CRs: find j where D_j p ≤ e_j
```

- **Hit:** `U* = H_x_j @ p + h_x_j` → `u_i = U*[offset_i : offset_i + nu_i]`
  - iters[k] = 1
- **Miss (state outside map):** ADMM fallback (FALLBACK_ITERS=25)
  - iters[k] = 1 + n_admm_iter

**Data transfer = 1 per step** (when no fallback). Same for l_max and state_bounds.

---

#### MODE B — V2 Online Solver (M > OFFLINE_BFS_MAX_M)

**Function:** `solve_gne_online_v2(p, agent_sols, game, prev_x_star, prev_crs, combo_cache)`
**File:** `mpgne/facet_gne.py` (lines ~1600–1790)

Mirrors MATLAB `IF_mpDiMPC_V2_3s.m` exactly for both FACET-H and FACET-LP.

---

**One-time setup before simulation loop:**

`precompute_point_location_arrays(agent_sols)` — called once in `_run_facet_sim`.
Stacks each agent's CRs into padded arrays:
- `sol._E_stack` : shape `(n_cr, max_ineq, n_theta)` — all E matrices, zero-padded
- `sol._f_stack` : shape `(n_cr, max_ineq)` — all f vectors, padded with +∞

Cost: one-time ~0.1s. Enables vectorized PointLocation via a single `numpy` batched matmul.

---

**Cold start (k=0 only):**
```
ADMM 500 iters → prev_x_star = res.x_stacked
```
Gives a realistic reference point for Sub-step B1. ADMM cost excluded from FACET timing.

---

**Sub-step B1 — PointLocation (per agent, independently)**

Repeated for each agent i = 0, 1, …, M−1:

```
U_{-i} = prev_x_star[ slices of all agents j ≠ i ]
θ_i    = [ U_{-i} ;  p ]                    ← (n_x_neg + n_p,) vector

# Warm path: check prev_crs[i] and its neighbors first (~50 checks, early return)
if prev_crs is not None:
    for idx in [prev_crs[i]] + facet_neighbors(prev_crs[i]):
        if max(E_idx @ θ_i - f_idx) ≤ tol: return idx

# Vectorized full scan (fires only on cold-start or large state jump):
violations = max( _E_stack @ θ_i  -  _f_stack,  axis=1 )   ← one batched matmul
current_crs[i] = argmin(violations)  or  first index where violations ≤ tol
```

Result: `current_crs[i]` — the CR index for agent i at the current θ_i.

---

**Sub-step B2 — Chebyshev LP Filter (per agent, matches MATLAB exactly)**

Repeated for each agent i = 0, 1, …, M−1:

```
candidates = [ current_crs[i] ] + facet_neighbors( current_crs[i] )
             # FACET-H: ~50 candidates   FACET-LP: ~10 candidates

filtered[i] = []
for cr_idx in candidates:
    b_eff = cr.f  -  cr.E[:, n_x_neg:] @ p     ← marginalize state/p out
    A_eff = cr.E[:, :n_x_neg]                   ← only U_{-i} constraints

    # Chebyshev LP (Gurobi ~10-50 µs; scipy HiGHS fallback):
    # max  r
    # s.t. A_eff U  +  ‖A_eff_rows‖ * r  ≤  b_eff
    #      r ≥ 0
    # r* > tol → polytope {U_{-i}: A_eff U ≤ b_eff} has non-empty interior → KEEP
    # r* ≤ tol or infeasible → projected CR is empty at current p → DISCARD
    # UNBOUNDED  → feasible set is non-empty (no upper bound on U) → KEEP
    if _chebyshev_feasible(gurobi_env, A_eff, b_eff):
        filtered[i].append(cr_idx)

# Safety: current CR always passes (PointLocation guarantees it).
feasible_per_agent[i] = filtered[i]  if filtered[i]  else [current_crs[i]]
```

This is the exact filter from MATLAB's `Chebyshev(A, b)` call (lines 47-50 / 59-63 / 71-74 of IF_mpDiMPC_V2_3s.m). Applied to ALL neighbors with no cap, no truncation.

---

**Sub-step B3 — Combo Enumeration and Verification**

```
for combo in itertools.product(*feasible_per_agent):
    combos_checked += 1

    # Assemble and solve equilibrium system (9×9 for M=3)
    Mx, Mp, M1 = _assemble_equilibrium_system(combo, agent_sols, game)
    sol  = np.linalg.solve(Mx, [Mp | M1])    ← LU factorization ~5 µs
    H_x  = sol[:, :n_p]
    h_x  = sol[:, n_p]
    x_star = H_x @ p + h_x                   ← evaluate at current p

    # Per-agent membership check
    valid = True
    for i, j_i in combo:
        θ_i = [ x_star_{-i} ;  p ]
        if max(E_{j_i} @ θ_i - f_{j_i}) > tol:
            valid = False; break

    if valid:
        combo_cache[combo] = (H_x, h_x)      ← store for Tier-2 scan
        return combo, x_star, combos_checked
```

In the warm case (system near equilibrium) the first combo in the product is correct
and `combos_checked = 1`.

---

**Tier-2 — Combo Cache Scan (fires only when B3 found nothing)**

```
# combo_cache = {tuple → (H_x, h_x)}, persisted across all steps of the run
for combo, (H_x, h_x) in combo_cache.items():
    if combo already checked in B3: skip
    combos_checked += 1
    x_star = H_x @ p + h_x                   ← no solve needed, ~5 µs
    # per-agent membership check (same as B3)
    if valid: return combo, x_star, combos_checked
```

The cache grows naturally: after ~20 steps it covers all combos the trajectory will ever visit.
Tier-2 fires only when the 1-hop Chebyshev-filtered set misses — rare for smooth trajectories.

---

**ADMM Fallback (last resort — fires only when both B3 and Tier-2 fail)**

```
ADMM 25 iters, tol=1e-3
prev_x_star = res.x_stacked    ← re-seed for next step
prev_crs    = None             ← reset warm hint
combo_cache NOT reset          ← historical combos still available
iters[k]    = 1 + n_admm_iter
fallbacks  += 1
```

---

**After successful V2 step:**
```
prev_x_star = U_star            ← carry forward as reference for next step
prev_crs    = list(combo)       ← warm hint for next PointLocation
iters[k]    = combos_checked    ← honest data transfer count
u_k[i]      = U_star[offset_i : offset_i + nu_i]
x(k+1)      = plant.step(x(k), u_k)
```

---

## 4. l_max vs state_bounds — Complete Comparison

| Step | l_max | state_bounds | Same algorithm? |
|---|---|---|---|
| Plant generation | identical | identical | ✓ |
| Game construction | shared constraint (C, d, S_coup) | generalized (Gamma, M_theta, bounds) | ✗ |
| Per-agent mpQP | ~9 constraints → ~100 CRs | ~42 constraints → ~2000 CRs | ✓ (same PPOPT call) |
| FACET-H detection | ~5-15 neighbors/CR | ~50-80 neighbors/CR | ✓ (same hash algorithm) |
| FACET-LP refinement | ~5-10 neighbors/CR | ~10 neighbors/CR | ✓ |
| Offline BFS (M=3) | ~2000 GNE CRs, feasible | 125K GNE CRs, 1 GB — **not viable** | ✓ (same BFS, different scale) |
| OFFLINE_BFS_MAX_M | 3 | **2** (never raise for state_bounds) | ✗ |
| Online MODE A | M ≤ 3 | M ≤ 2 | ✓ |
| Online MODE B V2 | M ≥ 4 | M ≥ 3 | ✓ (same algorithm) |
| B1 PointLocation cost | O(100) per agent | O(2000) per agent → vectorized essential | ✓ |
| B2 Chebyshev LP calls/step | ~10×M (compact FH) | ~80×M (FH) or ~10×M (FLP) | ✓ |
| Data transfer (MODE A) | 1/step | 1/step | ✓ |
| Data transfer (MODE B, typical) | ~1/step | ~1/step | ✓ |

**Root cause of scale difference:** state_bounds adds 2×Np×nx_total = 36 parametric constraints
per agent (upper + lower state bounds over the full horizon). Each parametric constraint creates
more CR boundaries → C(42,3)≈11k vs C(9,3)=84 possible CR configurations → ~20× more CRs.

---

## 5. Key Functions Reference

### `facet_gne.py`

| Function | Purpose | Step |
|---|---|---|
| `find_all_agent_cr_neighbors()` | Hash-based hyperplane adjacency detection | Step 4a |
| `refine_neighbors_with_lp()` | LP-refine to exact facet neighbors | Step 4b |
| `build_gne_solution_facet()` | BFS → full GNESolution (offline) | Step 5 |
| `build_gne_solution_lp_from_fh()` | Derive FLP map from FH (fast, no re-BFS) | Step 5b |
| `precompute_point_location_arrays()` | ★ Stack all CRs' E/f into padded arrays for vectorized PointLocation | Before online loop |
| `_get_online_gurobi_env()` | Lazy Gurobi env singleton for online LP calls | B2 filter |
| `_chebyshev_feasible()` | ★ Chebyshev LP feasibility check — exact MATLAB `Chebyshev(A,b)` | B2 filter |
| `_locate_cr_fast()` | Warm-hint + vectorized batched matmul point-location | B1 |
| `solve_gne_online_v2()` | ★ Full MATLAB V2 online solver (B1+B2+B3+Tier-2) | MODE B online |
| `_process_combo_kernel()` | Per-combo check for offline BFS | Step 5 |
| `solve_gne_online()` | (kept for reference) Old BFS + lazy cache approach | — |
| `build_combo_index()` | (preserved, unused) Full offline combo dict | — |
| `solve_gne_online_ci()` | (preserved, unused) Combo-index online search | — |

### `full_case_study.py`

| Function | Purpose |
|---|---|
| `run_one_plant()` | Full pipeline: Steps 1-5 + all simulations |
| `_run_admm_sim()` | ADMM closed-loop (baseline) |
| `_run_br_sim()` | Jacobi Best-Response (baseline) |
| `_run_impgne_sim()` | Implicit DiMPC / ImpGNE (baseline) |
| `_run_facet_sim()` | FACET-H/LP: MODE A (explicit) or MODE B (V2 online + cache) |
| `_run_explicit_sim()` | Explicit centralized map lookup (M ≤ OFFLINE_BFS_MAX_M) |
| `_generate_reports()` | Print Tables 1–4 + save boxplot figures |

---

## 6. Configuration — `full_case_study.py`

| Variable | Default | Notes |
|---|---|---|
| `N_PLANTS` | 3 | Random plants per M value |
| `T_SIM` | 100 | Simulation horizon |
| `M_LIST` | [2, 3, 4] | Agent counts to benchmark |
| `COUPLING_MODE` | `"state_bounds"` | `"l_max"` or `"state_bounds"` |
| `OFFLINE_BFS_MAX_M` | 2 | Explicit GNE map for M ≤ this. **Keep at 2 for state_bounds.** For l_max, can be 3. |
| `ALGO` | `combinatorial_parallel_exp` | PPOPT mpQP algorithm |
| `ADMM_RHO` | 1.0 | ADMM penalty (used in `admm_solve` calls) |
| `ADMM_ITERS` | 2000 | Iterations for ADMM benchmark method |
| `ADMM_TOL` | 1e-4 | ADMM convergence tolerance |
| `FALLBACK_ITERS` | 25 | ADMM iterations for V2 fallback (re-seed only, loose) |
| `FALLBACK_TOL` | 1e-3 | Fallback tolerance |
| `BR_ITERS` | 2000 | Jacobi BR max iterations |
| `IMP_ITERS` | 200 | ImpGNE max iterations |
| `SEED` | 250 | Random seed for plant generation |
| `L_MAX` | 2.5 | Aggregate input limit (l_max mode only) |

**Removed variables** (previously present, now deleted):
- `FACETH_MAX_HOPS` — was passed to `_run_facet_sim` but ignored by V2. Removed.
- `FACETLP_MAX_HOPS` — same reason. Removed.

---

## 7. How to Run

### Full Benchmark (state_bounds, M=2,3,4)
```bash
cd mpgne_ppopt
python tests/full_case_study.py
```
COUPLING_MODE = "state_bounds", OFFLINE_BFS_MAX_M = 2.
- M=2: explicit map + all 5 baselines
- M=3,4: V2 online (Chebyshev LP filter + combo cache) + all 5 baselines

Offline (mpQP + neighbors) loads from checkpoints if already computed. Online sim re-runs
if `results.pkl` is missing. To re-run online only: delete `M{M}_plant{idx}_results.pkl`.

### Quick Benchmark (state_bounds, M=2,3)
```bash
python tests/bench_state_bounds.py
```
Loads cached CRs and neighbor maps. Runs ADMM vs FACET-V2 and prints timing + DT.

### l_max benchmark
Set `COUPLING_MODE = "l_max"` in `full_case_study.py`. Can raise `OFFLINE_BFS_MAX_M = 3`.

### Report only (from existing checkpoints)
```python
_generate_reports(all_results, coupling_label, CKPT_DIR)
```

---

## 8. Checkpoint System

All intermediate results saved to `tests/full_case_study_data_{COUPLING_MODE}/`:

```
M{M}_plant{idx:03d}_agent_sols_base.pkl   # Step 3: per-agent mpQP solutions
M{M}_plant{idx:03d}_agent_sols_FH.pkl     # Step 4a: + FACET-H neighbor maps
M{M}_plant{idx:03d}_agent_sols_FLP.pkl    # Step 4b: + FACET-LP neighbor maps
M{M}_plant{idx:03d}_facet_sol_FH.pkl      # Step 5:  explicit GNE map (M ≤ OFFLINE_BFS_MAX_M only)
M{M}_plant{idx:03d}_facet_sol_FLP.pkl     # Step 5b: FLP GNE map (M ≤ OFFLINE_BFS_MAX_M only)
M{M}_plant{idx:03d}_results.pkl           # Simulation results (all methods)
```

To re-run a specific step: delete that step's `.pkl`. The script detects missing checkpoints
and recomputes only what is needed. Deleting only `results.pkl` is the fastest re-run option.

**Important:** `_E_stack` and `_f_stack` attributes (for vectorized PointLocation) are NOT stored
in the `.pkl` files — they are recomputed by `precompute_point_location_arrays()` at the start
of each `_run_facet_sim` call. This is correct behavior (fast, ~0.1s, no storage overhead).

---

## 9. Output Tables and Metrics

| Table | Content |
|---|---|
| **Table 1** | Average online solution time per step (ms) — 6 methods |
| **Table 2** | FACET fallback instances (ADMM fallbacks out of T_SIM) |
| **Table 3** | Average critical regions per agent, by M |
| **Table 4** | Total data transfer (communication rounds) over T_SIM — all 6 methods |

**Data transfer definition:**
- ADMM/BR/ImpGNE: `n_iter` per step (each iteration = one agent-to-agent exchange)
- Explicit map (MODE A): always 1 per step
- FACET V2 (MODE B): `combos_checked` per step — honest count including Tier-2 cache checks.
  Typical value = 1 for smooth trajectories; higher only on fallback steps.

**Excess DT above T_SIM** = fallbacks × ADMM_iters_per_fallback (~10 iters at tol=1e-3).

Saved figures: `crs_boxplot.png`, `times_boxplot.png`, `data_transfer_boxplot.png`

---

## 10. Known Issues and Status

### Working
- M=2 state_bounds: explicit map, DT=1, 0 fallbacks ✓
- M=3 state_bounds: V2 online (Chebyshev LP filter + cache), DT≈1 ✓
- M=4 state_bounds: V2 online, vectorized PointLocation essential ✓
- l_max M=2,3 (explicit map): working from saved checkpoints ✓
- l_max M=4+ (V2 online): working ✓
- Data transfer count is honest (`combos_checked`) ✓
- PointLocation is vectorized — no Python loop for M=4 with 2800 CRs/agent ✓
- Chebyshev LP filter matches MATLAB exactly for both FACET-H and FACET-LP ✓
- Combo cache prevents ADMM fallbacks on trajectory revisits ✓

### Known Limitations
- **FACET-H online is slower than FACET-LP** for large M (state_bounds):
  FACET-H has ~80 neighbors/CR → ~80×M Chebyshev LPs per step (each ~10-50 µs with Gurobi).
  For M=4: ~320 LPs × 30 µs ≈ 10 ms/step. FACET-LP has ~10 neighbors/CR → ~40 LPs ≈ 1.2 ms/step.
  This is correct MATLAB behavior — MATLAB also uses LP-verified (FACET-LP style) neighbors.
- **M=3 state_bounds explicit map** is infeasible (125K GNE CRs = 1 GB).
  `OFFLINE_BFS_MAX_M` must remain 2 for state_bounds.
- **Gurobi required for best performance.** scipy HiGHS fallback is correct but slower
  (~300-500 µs per LP vs ~10-50 µs for Gurobi).

### Preserved (unused, not removed)
- `solve_gne_online()`: old BFS + lazy cache version, kept in `facet_gne.py`.
- `build_combo_index()`, `solve_gne_online_ci()`: full combo index methods.

---

## 11. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | 2.4.4 | Core numerical, vectorized PointLocation batched matmul |
| `scipy` | 1.17.1 | `linprog` for LP tests (HiGHS fallback), `solve_discrete_are` |
| `matplotlib` | 3.10.8 | Plotting |
| `ppopt` | 1.6.12 | Multi-parametric QP solver (offline mpQP) |
| `gurobipy` | 13.0.1 | ★ Online Chebyshev LP filter + offline FACET-LP refinement. Falls back to scipy HiGHS if unavailable but is significantly slower. |
| `osqp` | — | QP solver used inside ADMM/BR/ImpGNE baselines |
