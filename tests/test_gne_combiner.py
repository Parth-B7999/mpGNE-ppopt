"""
Tests for mpgne/gne_combiner.py

Fast tests (no PPOPT, hand-crafted CRs):  run normally
Slow tests (full PPOPT solve + combine):  marked @pytest.mark.slow

Run fast only:  python -m pytest tests/test_gne_combiner.py -v -m "not slow"
Run all:        python -m pytest tests/test_gne_combiner.py -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpgne.game import GNEGame, Agent, make_random_game
from mpgne.cr_store import AgentCR, AgentSolution, GNESolution
from mpgne.gne_combiner import (
    _assemble_equilibrium_system,
    _solve_equilibrium,
    _project_crs_to_p_space,
    _cr_nonempty,
    build_gne_solution,
    verify_gne_at_p,
)
from ppopt.mp_solvers.solve_mpqp import mpqp_algorithm


# ---------------------------------------------------------------------------
# Minimal hand-crafted 2-agent game for structural tests
# ---------------------------------------------------------------------------
#
# N=2, n_x_i=1 each, n_p=1, n_coupling=1
# Agent 0 best response:  x_0* = 0.5 * x_1 + 0.2 * p + 1.0   (arbitrary affine)
# Agent 1 best response:  x_1* = 0.3 * x_0 + 0.1 * p - 0.5
#
# θ_0 = [x_1; p]  →  A_0 = [[0.5, 0.2]], b_0 = [1.0]
# θ_1 = [x_0; p]  →  A_1 = [[0.3, 0.1]], b_1 = [-0.5]
#
# M_x x* = M_p p + M_1:
#   [ 1  -0.5] [x_0]   [0.2] p + [1.0]
#   [-0.3  1 ] [x_1] = [0.1]     [-0.5]
#
# Solve: M_x^{-1} = 1/(1-0.15) * [[1, 0.5],[0.3, 1]] = 1/0.85 * ...
#   det(M_x) = 1 - 0.15 = 0.85

def _make_hand_game():
    """Return a tiny 2-agent game with known equilibrium structure."""
    n_p = 1

    def make_agent(idx, n_p):
        return Agent(
            index=idx, n_x=1,
            Q=np.array([[1.0]]),
            c=np.zeros(1),
            F=np.zeros((1, n_p)),
            C=np.ones((1, 1)),
            A_loc=np.vstack([np.eye(1), -np.eye(1)]),
            b_loc=10.0 * np.ones(2),
            S_loc=np.zeros((2, n_p)),
        )

    agents = [make_agent(i, n_p) for i in range(2)]
    game = GNEGame(
        agents=agents,
        d=np.array([10.0]),
        S_coup=np.zeros((1, n_p)),
        p_lb=np.array([-5.0]),
        p_ub=np.array([5.0]),
    )
    return game


def _make_hand_cr(A_row, b_val, E_rows, f_vals, index=0):
    """1-D AgentCR with given affine law and CR constraints."""
    return AgentCR(
        E=np.array(E_rows, dtype=float),
        f=np.array(f_vals, dtype=float),
        A=np.array([A_row], dtype=float),   # (1, n_theta)
        b=np.array([b_val], dtype=float),
        index=index,
    )


def _make_hand_agent_solutions():
    """
    Build AgentSolutions for the hand game.
    Single CR per agent (covers the full parameter space for simplicity).

    Agent 0: x_0* = 0.5*x_1 + 0.2*p + 1.0
             θ_0 = [x_1; p],  A_0 = [[0.5, 0.2]], b_0 = 1.0
             CR: -10 ≤ x_1 ≤ 10 and -5 ≤ p ≤ 5  (box on θ_0)

    Agent 1: x_1* = 0.3*x_0 + 0.1*p - 0.5
             θ_1 = [x_0; p],  A_1 = [[0.3, 0.1]], b_1 = -0.5
             CR: same box
    """
    # θ = [x_{-i}; p] ∈ R^2
    E_box = np.vstack([np.eye(2), -np.eye(2)])    # (4, 2)
    f_box = np.array([10., 5., 10., 5.])

    cr0 = _make_hand_cr(A_row=[0.5, 0.2], b_val=1.0,
                        E_rows=E_box, f_vals=f_box, index=0)
    cr1 = _make_hand_cr(A_row=[0.3, 0.1], b_val=-0.5,
                        E_rows=E_box, f_vals=f_box, index=0)

    sol0 = AgentSolution(agent_index=0, n_x_i=1, n_theta_i=2, regions=[cr0])
    sol1 = AgentSolution(agent_index=1, n_x_i=1, n_theta_i=2, regions=[cr1])
    return [sol0, sol1]


# ---------------------------------------------------------------------------
# _assemble_equilibrium_system
# ---------------------------------------------------------------------------

class TestAssembleEquilibriumSystem:

    def test_Mx_shape(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        Mx, Mp, M1 = _assemble_equilibrium_system((0, 0), sols, game)
        assert Mx.shape == (2, 2)

    def test_Mp_shape(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        Mx, Mp, M1 = _assemble_equilibrium_system((0, 0), sols, game)
        assert Mp.shape == (2, 1)

    def test_M1_shape(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        Mx, Mp, M1 = _assemble_equilibrium_system((0, 0), sols, game)
        assert M1.shape == (2,)

    def test_Mx_diagonal_is_identity(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        Mx, _, _ = _assemble_equilibrium_system((0, 0), sols, game)
        # Diagonal blocks should be identity
        np.testing.assert_allclose(Mx[0, 0], 1.0)
        np.testing.assert_allclose(Mx[1, 1], 1.0)

    def test_Mx_off_diagonal_correct(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        Mx, _, _ = _assemble_equilibrium_system((0, 0), sols, game)
        # Agent 0 law: x_0* = 0.5 x_1 + ... → Mx[0,1] = -0.5
        np.testing.assert_allclose(Mx[0, 1], -0.5)
        # Agent 1 law: x_1* = 0.3 x_0 + ... → Mx[1,0] = -0.3
        np.testing.assert_allclose(Mx[1, 0], -0.3)

    def test_Mp_correct(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        _, Mp, _ = _assemble_equilibrium_system((0, 0), sols, game)
        np.testing.assert_allclose(Mp[0, 0], 0.2)
        np.testing.assert_allclose(Mp[1, 0], 0.1)

    def test_M1_correct(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        _, _, M1 = _assemble_equilibrium_system((0, 0), sols, game)
        np.testing.assert_allclose(M1[0],  1.0)
        np.testing.assert_allclose(M1[1], -0.5)

    def test_Mx_det_nonzero(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        Mx, _, _ = _assemble_equilibrium_system((0, 0), sols, game)
        # det = 1 - 0.5*0.3 = 0.85
        np.testing.assert_allclose(np.linalg.det(Mx), 0.85, atol=1e-10)


# ---------------------------------------------------------------------------
# _solve_equilibrium
# ---------------------------------------------------------------------------

class TestSolveEquilibrium:

    def _hand_matrices(self):
        Mx = np.array([[1.0, -0.5], [-0.3, 1.0]])
        Mp = np.array([[0.2], [0.1]])
        M1 = np.array([1.0, -0.5])
        return Mx, Mp, M1

    def test_unique_flag(self):
        Mx, Mp, M1 = self._hand_matrices()
        eq = _solve_equilibrium(Mx, Mp, M1)
        assert eq.is_unique is True
        assert eq.solvable is True

    def test_H_x_shape(self):
        Mx, Mp, M1 = self._hand_matrices()
        eq = _solve_equilibrium(Mx, Mp, M1)
        assert eq.H_x.shape == (2, 1)

    def test_h_x_shape(self):
        Mx, Mp, M1 = self._hand_matrices()
        eq = _solve_equilibrium(Mx, Mp, M1)
        assert eq.h_x.shape == (2,)

    def test_equilibrium_residual_unique(self):
        """M_x (H_x p + h_x) should equal M_p p + M_1 for any p."""
        Mx, Mp, M1 = self._hand_matrices()
        eq = _solve_equilibrium(Mx, Mp, M1)
        for p_val in [-3.0, 0.0, 2.5]:
            p = np.array([p_val])
            x_star = eq.H_x @ p + eq.h_x
            lhs = Mx @ x_star
            rhs = Mp @ p + M1
            np.testing.assert_allclose(lhs, rhs, atol=1e-10)

    def test_singular_solvable(self):
        """Rank-1 M_x with compatible RHS → solvable, not unique."""
        Mx = np.array([[1.0, -1.0], [-1.0, 1.0]])   # rank 1
        Mp = np.array([[1.0], [-1.0]])               # in range of M_x
        M1 = np.array([0.5, -0.5])
        eq = _solve_equilibrium(Mx, Mp, M1)
        assert eq.solvable is True
        assert eq.is_unique is False

    def test_singular_insolvable(self):
        """Rank-1 M_x with incompatible RHS → insolvable."""
        Mx = np.array([[1.0, -1.0], [-1.0, 1.0]])   # rank 1
        Mp = np.array([[1.0], [1.0]])                # NOT in range of M_x
        M1 = np.array([0.0, 0.0])
        eq = _solve_equilibrium(Mx, Mp, M1)
        assert eq.solvable is False

    def test_minorm_residual(self):
        """Min-norm solution satisfies M_x x* = M_p p + M_1 when solvable."""
        Mx = np.array([[1.0, -1.0], [-1.0, 1.0]])
        Mp = np.array([[1.0], [-1.0]])
        M1 = np.array([0.5, -0.5])
        eq = _solve_equilibrium(Mx, Mp, M1)
        assert eq.solvable
        p = np.array([1.0])
        x = eq.H_x @ p + eq.h_x
        np.testing.assert_allclose(Mx @ x, Mp @ p + M1, atol=1e-9)


# ---------------------------------------------------------------------------
# _project_crs_to_p_space
# ---------------------------------------------------------------------------

class TestProjectCRsToPSpace:

    def test_D_shape(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        Mx, Mp, M1 = _assemble_equilibrium_system((0, 0), sols, game)
        eq = _solve_equilibrium(Mx, Mp, M1)
        D, e = _project_crs_to_p_space((0, 0), sols, game, eq.H_x, eq.h_x)
        # 2 agents × 4 CR rows each = 8 rows; n_p=1
        assert D.shape == (8, 1)

    def test_e_shape(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        Mx, Mp, M1 = _assemble_equilibrium_system((0, 0), sols, game)
        eq = _solve_equilibrium(Mx, Mp, M1)
        D, e = _project_crs_to_p_space((0, 0), sols, game, eq.H_x, eq.h_x)
        assert e.shape == (8,)

    def test_p_zero_satisfies_projected_cr(self):
        """x*(0) substituted back should satisfy all CR constraints."""
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        Mx, Mp, M1 = _assemble_equilibrium_system((0, 0), sols, game)
        eq = _solve_equilibrium(Mx, Mp, M1)
        D, e = _project_crs_to_p_space((0, 0), sols, game, eq.H_x, eq.h_x)
        p = np.array([0.0])
        assert np.all(D @ p <= e + 1e-8)


# ---------------------------------------------------------------------------
# _cr_nonempty
# ---------------------------------------------------------------------------

class TestCrNonempty:

    def test_simple_feasible(self):
        D = np.array([[1.0], [-1.0]])
        e = np.array([5.0, 5.0])     # -5 ≤ p ≤ 5
        assert _cr_nonempty(D, e) is True

    def test_infeasible(self):
        D = np.array([[1.0], [-1.0]])
        e = np.array([-6.0, -6.0])  # p ≤ -6 and p ≥ 6 → impossible
        assert _cr_nonempty(D, e) is False

    def test_single_point(self):
        D = np.array([[1.0], [-1.0]])
        e = np.array([3.0, -3.0])    # p ≤ 3 and p ≥ 3 → p = 3 only
        # Chebyshev radius = 0 → r* ≈ 0, above -tol
        assert _cr_nonempty(D, e, tol=1e-4) is True

    def test_2d_feasible(self):
        D = np.vstack([np.eye(2), -np.eye(2)])
        e = 5.0 * np.ones(4)
        assert _cr_nonempty(D, e) is True

    def test_2d_infeasible(self):
        D = np.vstack([np.eye(2), -np.eye(2)])
        e = np.array([-1., -1., -1., -1.])   # all bounds negative → impossible
        assert _cr_nonempty(D, e) is False


# ---------------------------------------------------------------------------
# build_gne_solution (full pipeline, fast: hand-crafted CRs)
# ---------------------------------------------------------------------------

class TestBuildGneSolutionFast:

    def test_returns_gne_solution(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        gne = build_gne_solution(game, sols, verbose=False)
        assert isinstance(gne, GNESolution)

    def test_at_least_one_cr(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        gne = build_gne_solution(game, sols, verbose=False)
        assert gne.n_cr >= 1

    def test_gne_cr_has_correct_n_p(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        gne = build_gne_solution(game, sols, verbose=False)
        assert gne.n_p == game.n_p

    def test_equilibrium_residual_small(self):
        """For any p inside a GNE CR, the equilibrium residual must be ~0."""
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        gne = build_gne_solution(game, sols, verbose=False)
        assert gne.n_cr >= 1
        p = np.array([0.0])
        k = gne.locate(p)
        if k is not None:
            assert gne[k].residual(p) < 1e-8

    def test_x_star_satisfies_equilibrium_equations(self):
        """M_x x*(p) == M_p p + M_1 for a p inside the GNE CR."""
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        gne = build_gne_solution(game, sols, verbose=False)
        if gne.n_cr == 0:
            pytest.skip("No GNE CRs found for hand game")
        cr = gne[0]
        p = np.array([1.0])
        x = cr.evaluate(p)
        lhs = cr.Mx @ x
        rhs = cr.Mp @ p + cr.M1
        np.testing.assert_allclose(lhs, rhs, atol=1e-8)

    def test_summary_contains_n_cr(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        gne = build_gne_solution(game, sols, verbose=False)
        s = gne.summary()
        assert str(gne.n_cr) in s


# ---------------------------------------------------------------------------
# verify_gne_at_p (fast, hand-crafted)
# ---------------------------------------------------------------------------

class TestVerifyGneAtP:

    def test_found_inside_cr(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        gne = build_gne_solution(game, sols, verbose=False)
        p = np.array([0.0])
        k = gne.locate(p)
        if k is None:
            pytest.skip("p=0 not in any GNE CR for hand game")
        result = verify_gne_at_p(p, gne, game, sols, verbose=False)
        assert result['found'] is True

    def test_not_found_outside(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        gne = build_gne_solution(game, sols, verbose=False)
        p = np.array([999.0])
        result = verify_gne_at_p(p, gne, game, sols, verbose=False)
        assert result['found'] is False

    def test_residual_small_when_found(self):
        game = _make_hand_game()
        sols = _make_hand_agent_solutions()
        gne = build_gne_solution(game, sols, verbose=False)
        p = np.array([0.0])
        if gne.locate(p) is None:
            pytest.skip("p=0 not in any CR")
        result = verify_gne_at_p(p, gne, game, sols, verbose=False)
        if result['found']:
            assert result['residual'] < 1e-8


# ---------------------------------------------------------------------------
# Full end-to-end with PPOPT (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestEndToEnd:

    @pytest.fixture
    def solved_game2(self):
        from mpgne.mp_solver import solve_all_agents_mp
        game = make_random_game(N=2, n_x=1, n_p=1, n_coupling=1,
                                x_bound=5.0, p_bound=3.0, seed=0)
        sols = solve_all_agents_mp(game,
                                   algorithm=mpqp_algorithm.combinatorial,
                                   verbose=False)
        return game, sols

    def test_build_returns_gne_solution(self, solved_game2):
        game, sols = solved_game2
        gne = build_gne_solution(game, sols, verbose=False)
        assert isinstance(gne, GNESolution)

    def test_at_least_one_gne_cr(self, solved_game2):
        game, sols = solved_game2
        gne = build_gne_solution(game, sols, verbose=False)
        assert gne.n_cr >= 1

    def test_residual_small_for_found_p(self, solved_game2):
        """For p inside a GNE CR, equilibrium residual < 1e-6."""
        game, sols = solved_game2
        gne = build_gne_solution(game, sols, verbose=False)
        p = np.zeros(game.n_p)
        k = gne.locate(p)
        if k is None:
            pytest.skip("p=0 not in any GNE CR for this seed")
        assert gne[k].residual(p) < 1e-6

    def test_x_star_feasible(self, solved_game2):
        """x*(p) must satisfy all game constraints."""
        game, sols = solved_game2
        gne = build_gne_solution(game, sols, verbose=False)
        p = np.zeros(game.n_p)
        k = gne.locate(p)
        if k is None:
            pytest.skip("p=0 not in any GNE CR for this seed")
        x_star = gne[k].evaluate(p)
        assert game.all_feasible(x_star, p, tol=1e-5)

    def test_verify_gne_at_p(self, solved_game2):
        """verify_gne_at_p returns found=True for p=0 (if in a CR)."""
        game, sols = solved_game2
        gne = build_gne_solution(game, sols, verbose=False)
        p = np.zeros(game.n_p)
        result = verify_gne_at_p(p, gne, game, sols, verbose=False)
        if result['found']:
            assert result['residual'] < 1e-6
            assert result['feasible'] is True

    def test_combination_covers_all_agents(self, solved_game2):
        """Every GNE CR's combination has exactly N entries."""
        game, sols = solved_game2
        gne = build_gne_solution(game, sols, verbose=False)
        for cr in gne.regions:
            assert len(cr.combination) == game.N
