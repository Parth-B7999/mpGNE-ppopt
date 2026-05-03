"""
plant.py — Subsystem and Plant data structures.

Paper reference: Saini, Brahmbhatt et al. (C&ChE 2025), Sec. 2.3, Eq. (7)-(10)
  x_i(k+1) = A_i x_i(k) + sum_j B_{i,j} u_j(k)

The Plant assembles all subsystems into the global model:
  x(k+1) = A x(k) + sum_j B_j u_j(k)
where A = block_diag(A_1, ..., A_M), B_j = [B_{1,j}; ...; B_{M,j}]
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Subsystem:
    """
    Represents one subsystem i in a network of M coupled subsystems.

    State dynamics:  x_i(k+1) = A_i x_i(k) + sum_{j=1}^{M} B_{i,j} u_j(k)

    Attributes
    ----------
    index : int
        Zero-based subsystem index i.
    A : ndarray, shape (nx_i, nx_i)
        Local state matrix A_i.
    B : dict[int, ndarray]
        Coupling input matrices.  B[j] is B_{i,j}, shape (nx_i, nu_j).
        j is 0-based.  B[i] is the self-coupling (direct input) matrix.
    x_lb, x_ub : ndarray, shape (nx_i,)
        State lower/upper bounds.
    u_lb, u_ub : ndarray, shape (nu_i,)
        Input lower/upper bounds for u_i (this subsystem's own input).
    """

    index: int
    A: np.ndarray
    B: dict[int, np.ndarray]
    x_lb: np.ndarray
    x_ub: np.ndarray
    u_lb: np.ndarray
    u_ub: np.ndarray

    @property
    def nx(self) -> int:
        return self.A.shape[0]

    @property
    def nu(self) -> int:
        return self.B[self.index].shape[1]

    def step(self, x_i: np.ndarray, u_all: dict[int, np.ndarray]) -> np.ndarray:
        """Simulate one step:  x_i(k+1) = A_i x_i + sum_j B_{i,j} u_j."""
        x_next = self.A @ x_i
        for j, B_ij in self.B.items():
            x_next = x_next + B_ij @ u_all[j]
        return x_next


@dataclass
class Plant:
    """
    Full interconnected plant of M coupled subsystems.

    Assembles:
        x(k+1) = A x(k) + sum_j B_j u_j(k)
    where A = block_diag(A_1,...,A_M), B_j = stack of B_{i,j} over i.

    Attributes
    ----------
    subsystems : list[Subsystem]
        Length M, indexed 0 to M-1.
    Np : int
        Prediction horizon (same for all controllers, as in the paper).
    """

    subsystems: list[Subsystem]
    Np: int = 3

    @property
    def M(self) -> int:
        return len(self.subsystems)

    @property
    def nx(self) -> int:
        return sum(s.nx for s in self.subsystems)

    @property
    def nu(self) -> int:
        return sum(s.nu for s in self.subsystems)

    @property
    def A(self) -> np.ndarray:
        """Global block-diagonal state matrix."""
        return np.block([[np.zeros((si.nx, sj.nx)) if i != j else si.A
                          for j, sj in enumerate(self.subsystems)]
                         for i, si in enumerate(self.subsystems)])

    def B_j(self, j: int) -> np.ndarray:
        """Global input matrix for controller j: B_j = [B_{1,j}; ...; B_{M,j}]."""
        return np.vstack([s.B.get(j, np.zeros((s.nx, self.subsystems[j].nu)))
                          for s in self.subsystems])

    def step(self, x: np.ndarray, u: dict[int, np.ndarray]) -> np.ndarray:
        """Simulate one global step."""
        x_next = self.A @ x
        for j in range(self.M):
            x_next = x_next + self.B_j(j) @ u[j]
        return x_next

    def state_slice(self, i: int) -> slice:
        """Index slice for subsystem i's states within the global x vector."""
        start = sum(s.nx for s in self.subsystems[:i])
        return slice(start, start + self.subsystems[i].nx)


# ---------------------------------------------------------------------------
# Factory: exact sample plant from ACC 2026 paper, M=2 case (Eq. 16)
# ---------------------------------------------------------------------------

def make_acc2026_plant() -> Plant:
    """
    Two-subsystem plant from:
        Brahmbhatt et al., ACC 2026, Section IV-A, Eq. (15)-(16).

    x1 in R^2, x2 in R^2, u1 in R^1, u2 in R^1.
    State bounds randomly chosen in [-100,-10] (lb) and [10,100] (ub).
    Input bounds randomly chosen in [-5,-1] (lb) and [1,5] (ub).
    """
    A1 = np.array([[0.1645, 0.7399],
                   [0.0815, -0.4704]])
    B11 = np.array([[-0.3639],
                    [-0.7616]])
    B12 = np.array([[0.8797],
                    [0.2911]])

    A2 = np.array([[-0.0411, 0.0894],
                   [0.2786,  0.2946]])
    B21 = np.array([[0.0878],
                    [0.4421]])
    B22 = np.array([[0.0450],
                    [0.9874]])

    sub0 = Subsystem(
        index=0,
        A=A1,
        B={0: B11, 1: B12},
        x_lb=np.array([-63.5878, -59.6464]),
        x_ub=np.array([29.6809, 19.5218]),
        u_lb=np.array([-1.2686]),
        u_ub=np.array([3.5116]),
    )

    sub1 = Subsystem(
        index=1,
        A=A2,
        B={0: B21, 1: B22},
        x_lb=np.array([-67.0765, -31.2846]),
        x_ub=np.array([19.8728, 15.7232]),
        u_lb=np.array([-1.1090]),
        u_ub=np.array([4.0879]),
    )

    return Plant(subsystems=[sub0, sub1], Np=3)
