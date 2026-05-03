"""
plant_gen.py — Random stable controllable plant generator.

Matches the case study setup in:
  Saini, Brahmbhatt et al. (C&ChE 2025), Section 4
  Brahmbhatt et al. (ACC 2026), Section IV-A
  MATLAB reference: Step1_ModelCreater_3s.m

Per-subsystem parameters (fixed, matching MATLAB Step1_ModelCreater_3s.m):
  nx_i = 2 states,  nu_i = 1 input
  A_i  elements ~ Uniform[-1, 1], resampled until spectral radius < 1 (stable)
  B_{i,j} elements ~ Uniform[-1, 1]  (all j, full coupling)
  State bounds: lb ~ Uniform[-100, -10]^nx_total  (MATLAB: rand*90 - 100)
               ub ~ Uniform[ 10, 100]^nx_total  (MATLAB: rand*90 + 10)
  Input bounds: lb ~ Uniform[-5,  -1]  (MATLAB: rand*4 - 5)
               ub ~ Uniform[ 1,   5]  (MATLAB: rand*4 + 1)
"""

from __future__ import annotations
import numpy as np
from .plant import Subsystem, Plant


# ─────────────────────────────────────────────────────────────────────────────
#  Single random plant
# ─────────────────────────────────────────────────────────────────────────────

def make_random_plant(
    M: int,
    Np: int = 3,
    nx_i: int = 2,
    nu_i: int = 1,
    rho_min: float = 0.0,
    rho_max: float = 1.0,
    rng: np.random.Generator | None = None,
) -> Plant:
    """
    Generate one random stable plant with M coupled subsystems.

    Matches MATLAB Step1_ModelCreater_3s.m exactly:
      - A_i resampled until all eigenvalues strictly inside unit circle (|eig| < 1)
      - B is (nx_i, nu_i*M) full coupling matrix; split into per-controller columns
      - Bounds match MATLAB: Xmin in [-100,-10], Xmax in [10,100],
        Umin in [-5,-1], Umax in [1,5]

    Parameters
    ----------
    M       : number of subsystems
    Np      : prediction horizon (MATLAB: OH=NC=DH=3)
    nx_i    : states per subsystem (paper: 2)
    nu_i    : inputs per subsystem (paper: 1)
    rho_max : unused (A is resampled until stable, matching MATLAB while loop)
    rng     : numpy Generator  (created internally if None)
    """
    if rng is None:
        rng = np.random.default_rng()

    subsystems = []
    for i in range(M):

        # ── A_i: resample until spectral radius in [rho_min, rho_max) ────────
        # rho_min > 0 forces slow (near-marginal) dynamics for hard benchmarks.
        # Default rho_min=0.0 matches original MATLAB: while any(|eig|>=1) ... end
        A_i = rng.uniform(-1.0, 1.0, (nx_i, nx_i))
        rho = np.max(np.abs(np.linalg.eigvals(A_i)))
        while not (rho_min <= rho < rho_max):
            A_i = rng.uniform(-1.0, 1.0, (nx_i, nx_i))
            rho = np.max(np.abs(np.linalg.eigvals(A_i)))

        # ── B: full (nx_i, nu_i*M) matrix, split per controller column ───────
        # Matches MATLAB: B = rand(nx, nu*np)*2-1;  then model.B = aux(:,i)
        B_full = rng.uniform(-1, 1, (nx_i, nu_i * M))   # (2, M)
        B = {j: B_full[:, j*nu_i:(j+1)*nu_i] for j in range(M)}

        # ── Bounds matching MATLAB exactly ──────────────────────────────────
        # MATLAB: Xmax = rand(1,nx*np)*90 + 10  → Uniform[10, 100]
        # MATLAB: Xmin = rand(1,nx*np)*90 - 100 → Uniform[-100, -10]
        # MATLAB: Umax = rand(1,nu*np)*4  + 1   → Uniform[1, 5]
        # MATLAB: Umin = rand(1,nu*np)*4  - 5   → Uniform[-5, -1]
        x_lb = rng.uniform(-100.0, -10.0, nx_i)
        x_ub = rng.uniform(  10.0, 100.0, nx_i)
        u_lb = rng.uniform(  -5.0,  -1.0, nu_i)
        u_ub = rng.uniform(   1.0,   5.0, nu_i)

        subsystems.append(Subsystem(
            index=i, A=A_i, B=B,
            x_lb=x_lb, x_ub=x_ub,
            u_lb=u_lb, u_ub=u_ub,
        ))

    return Plant(subsystems=subsystems, Np=Np)


# ─────────────────────────────────────────────────────────────────────────────
#  Batch generator
# ─────────────────────────────────────────────────────────────────────────────

def make_random_plants(
    M: int,
    N: int,
    Np: int = 3,
    nx_i: int = 2,
    nu_i: int = 1,
    rho_min: float = 0.0,
    rho_max: float = 1.0,
    seed: int = 2025,
) -> list[Plant]:
    """
    Generate N random plants for the case study.

    Parameters
    ----------
    M       : number of subsystems
    N       : number of plants  (paper: 100)
    rho_min : minimum spectral radius of A_i (0.0 = no lower bound)
    rho_max : maximum spectral radius of A_i (< 1.0 guarantees stability)
    seed    : fixed seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    return [make_random_plant(M, Np, nx_i, nu_i, rho_min, rho_max, rng) for _ in range(N)]


# ─────────────────────────────────────────────────────────────────────────────
#  Random initial condition (inside bounds, near origin)
# ─────────────────────────────────────────────────────────────────────────────

def make_ic(plant: Plant, scale: float = 0.1, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample a random initial state inside the state bounds.

    Uses x0 = Uniform[scale * x_lb, scale * x_ub] per subsystem so the IC
    is well inside the feasible region.  The same IC is used for all methods.
    """
    if rng is None:
        rng = np.random.default_rng()

    parts = []
    for s in plant.subsystems:
        # scale down from bounds: lb*scale..ub*scale
        lo = np.minimum(s.x_lb * scale, s.x_ub * scale)
        hi = np.maximum(s.x_lb * scale, s.x_ub * scale)
        parts.append(rng.uniform(lo, hi))
    return np.concatenate(parts)