"""
Tests for mpgne/mp_solver.py

Fast tests (matrix shapes, no PPOPT):  marked normally
Slow tests (actual PPOPT solve):       marked with @pytest.mark.slow

Run fast only:   python -m pytest tests/test_mp_solver.py -v -m "not slow"
Run all:         python -m pytest tests/test_mp_solver.py -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpgne.game import make_random_game
from mpgne.plant_gen import make_random_plants
from mpgne.mpc_builder import make_gne_game_from_plant, default_local_weights
from mpgne.cr_store import AgentSolution
from mpgne.mp_solver import (
    build_cost_matrices,
    build_constraint_matrices,
    build_parameter_space,
    _extract_box_bounds,
    solve_agent_mp,
    solve_all_agents_mp,
)
from ppopt.mp_solvers.solve_mpqp import mpqp_algorithm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def game2():
    """N=2, n_x=1, n_p=1, n_coupling=1 — simplest possible game."""
    return make_random_game(N=2, n_x=1, n_p=1, n_coupling=1,
                            x_bound=5.0, p_bound=3.0, seed=0)


@pytest.fixture
def game3():
    """N=3, n_x=2, n_p=2, n_coupling=1."""
    return make_random_game(N=3, n_x=2, n_p=2, n_coupling=1,
                            x_bound=5.0, p_bound=5.0, seed=7)


@pytest.fixture
def plant2():
    """M=2 random plant, Np=3."""
    return make_random_plants(2, 1, Np=3, seed=11)[0]


@pytest.fixture
def game_state_bounds(plant2):
    """MPC game with state-bounds coupling (default, ACC 2026)."""
    Q, R, P = default_local_weights(plant2)
    return make_gne_game_from_plant(plant2, coupling_mode="state_bounds",
                                    Q_list=Q, R_list=R, P_list=P)


@pytest.fixture
def game_lmax(plant2):
    """MPC game with aggregate-input L_max coupling (earlier formulation)."""
    Q, R, P = default_local_weights(plant2)
    return make_gne_game_from_plant(plant2, coupling_mode="l_max", L_max=5.0,
                                    Q_list=Q, R_list=R, P_list=P)


# ---------------------------------------------------------------------------
# _extract_box_bounds
# ---------------------------------------------------------------------------

class TestExtractBoxBounds:

    def test_symmetric_box(self, game2):
        ai = game2.agents[0]
        lb, ub = _extract_box_bounds(ai)
        assert lb.shape == (ai.n_x,)
        assert ub.shape == (ai.n_x,)
        np.testing.assert_allclose(ub,  5.0 * np.ones(ai.n_x))
        np.testing.assert_allclose(lb, -5.0 * np.ones(ai.n_x))

    def test_bounds_lb_le_ub(self, game3):
        for a in game3.agents:
            lb, ub = _extract_box_bounds(a)
            assert np.all(lb <= ub)

    def test_fallback_warns(self):
        from mpgne.game import Agent
        bad_agent = Agent(
            index=0, n_x=2,
            Q=np.eye(2), c=np.zeros(2), F=np.zeros((2, 1)),
            C=np.ones((1, 2)),
            A_loc=np.random.randn(3, 2),   # non-box structure
            b_loc=np.ones(3),
            S_loc=np.zeros((3, 1)),
        )
        with pytest.warns(UserWarning, match="box structure"):
            lb, ub = _extract_box_bounds(bad_agent)
        np.testing.assert_allclose(lb, -1e3 * np.ones(2))
        np.testing.assert_allclose(ub,  1e3 * np.ones(2))


# ---------------------------------------------------------------------------
# build_cost_matrices — shapes and values
# ---------------------------------------------------------------------------

class TestBuildCostMatrices:

    def test_Q_shape(self, game2):
        Q, H, c = build_cost_matrices(game2, 0)
        assert Q.shape == (1, 1)

    def test_H_shape_game2(self, game2):
        # n_theta = (N-1)*n_x + n_p = 1*1 + 1 = 2
        _, H, _ = build_cost_matrices(game2, 0)
        assert H.shape == (1, 2)

    def test_H_shape_game3(self, game3):
        # agent 0: n_x_i=2, n_x_neg=(3-1)*2=4, n_p=2 → n_theta=6
        _, H, _ = build_cost_matrices(game3, 0)
        assert H.shape == (2, 6)

    def test_c_shape(self, game2):
        _, _, c = build_cost_matrices(game2, 0)
        assert c.shape == (1,)

    def test_Q_is_pd(self, game3):
        for i in range(game3.N):
            Q, _, _ = build_cost_matrices(game3, i)
            assert np.linalg.eigvalsh(Q).min() > 0

    def test_H_x_neg_block_is_zero(self, game2):
        # F_i = 0 in make_random_game, so H should be all zeros
        _, H, _ = build_cost_matrices(game2, 0)
        np.testing.assert_allclose(H, np.zeros((1, 2)))

    def test_H_p_block_equals_F_i(self):
        """When F_i is non-zero, H's p-block matches it."""
        g = make_random_game(N=2, n_x=1, n_p=2, seed=5)
        # Manually set F_i
        g.agents[0].F = np.array([[1.5, -0.3]])
        _, H, _ = build_cost_matrices(g, 0)
        # n_theta=3 (1 from x_{-i}, 2 from p); H[:,1:] should be F_i
        np.testing.assert_allclose(H[:, 1:], [[1.5, -0.3]])


# ---------------------------------------------------------------------------
# build_constraint_matrices — shapes and correctness
# ---------------------------------------------------------------------------

class TestBuildConstraintMatrices:

    def test_G_shape_game2(self, game2):
        # n_loc = 2*n_x = 2, n_coupling = 1 → total rows = 3
        G, b, F = build_constraint_matrices(game2, 0)
        assert G.shape == (3, 1)

    def test_b_shape_game2(self, game2):
        G, b, F = build_constraint_matrices(game2, 0)
        assert b.shape == (3,)

    def test_F_shape_game2(self, game2):
        # n_theta = 2 (1 x_{-i} + 1 p)
        G, b, F = build_constraint_matrices(game2, 0)
        assert F.shape == (3, 2)

    def test_G_shape_game3(self, game3):
        # n_loc=4, n_coupling=1 → 5 rows; n_x_i=2
        G, b, F = build_constraint_matrices(game3, 1)
        assert G.shape == (5, 2)

    def test_F_shape_game3(self, game3):
        # n_theta_i = 4 + 2 = 6
        G, b, F = build_constraint_matrices(game3, 1)
        assert F.shape == (5, 6)

    def test_G_local_block_is_A_loc(self, game2):
        ai = game2.agents[0]
        G, _, _ = build_constraint_matrices(game2, 0)
        np.testing.assert_array_equal(G[:ai.n_loc], ai.A_loc)

    def test_G_coupling_block_is_C_i(self, game2):
        ai = game2.agents[0]
        G, _, _ = build_constraint_matrices(game2, 0)
        np.testing.assert_array_equal(G[ai.n_loc:], ai.C)

    def test_b_coupling_block_is_d(self, game2):
        ai = game2.agents[0]
        _, b, _ = build_constraint_matrices(game2, 0)
        np.testing.assert_array_equal(b[ai.n_loc:], game2.d)

    def test_F_coupling_x_neg_block_is_neg_C_neg(self, game2):
        # C_{-i} for agent 0 = C_1 (only one other agent)
        C1 = game2.agents[1].C       # (1, 1)
        _, _, F = build_constraint_matrices(game2, 0)
        n_loc = game2.agents[0].n_loc
        # F[n_loc, 0] = -C1[0,0]  (x_{-i} block)
        np.testing.assert_allclose(F[n_loc:, :1], -C1)

    def test_coupling_feasible_at_zero(self, game2):
        """x_i=0 with θ_i=0 satisfies constraints (coupling RHS = d > 0)."""
        G, b, F = build_constraint_matrices(game2, 0)
        x_i   = np.zeros(game2.agents[0].n_x)
        theta = np.zeros(2)
        lhs = G @ x_i
        rhs = b + F @ theta
        assert np.all(lhs <= rhs + 1e-10)


# ---------------------------------------------------------------------------
# build_parameter_space — shapes and bounds
# ---------------------------------------------------------------------------

class TestBuildParameterSpace:

    def test_A_t_shape_game2(self, game2):
        # n_theta = 2 → A_t is (4, 2)
        A_t, b_t = build_parameter_space(game2, 0)
        assert A_t.shape == (4, 2)

    def test_b_t_shape_game2(self, game2):
        A_t, b_t = build_parameter_space(game2, 0)
        assert b_t.shape == (4, 1)

    def test_A_t_is_pm_identity(self, game3):
        n_theta = (game3.N - 1) * 2 + 2   # 4 + 2 = 6
        A_t, _ = build_parameter_space(game3, 0)
        assert A_t.shape == (2 * n_theta, n_theta)
        np.testing.assert_array_equal(A_t[:n_theta],  np.eye(n_theta))
        np.testing.assert_array_equal(A_t[n_theta:], -np.eye(n_theta))

    def test_theta_zero_satisfies_param_space(self, game2):
        A_t, b_t = build_parameter_space(game2, 0)
        theta = np.zeros(2)
        assert np.all(A_t @ theta <= b_t.ravel() + 1e-10)

    def test_p_bounds_reflected(self, game2):
        # p ∈ [-3, 3] in game2; last column of A_t should enforce this
        A_t, b_t = build_parameter_space(game2, 0)
        n_theta = 2
        # Row 1 (index 1): +I row for p → b_t[1] = p_ub = 3
        assert abs(b_t[1, 0] - 3.0) < 1e-10
        # Row 3 (index 3): -I row for p → b_t[3] = -p_lb = 3
        assert abs(b_t[3, 0] - 3.0) < 1e-10


# ---------------------------------------------------------------------------
# Full PPOPT solve (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSolveAgentMp:

    def test_returns_agent_solution(self, game2):
        sol = solve_agent_mp(game2, 0,
                             algorithm=mpqp_algorithm.combinatorial,
                             verbose=False)
        assert isinstance(sol, AgentSolution)
        assert sol.agent_index == 0

    def test_at_least_one_cr(self, game2):
        sol = solve_agent_mp(game2, 0,
                             algorithm=mpqp_algorithm.combinatorial,
                             verbose=False)
        assert sol.n_cr >= 1

    def test_n_x_i_correct(self, game2):
        sol = solve_agent_mp(game2, 0,
                             algorithm=mpqp_algorithm.combinatorial,
                             verbose=False)
        assert sol.n_x_i == game2.agents[0].n_x

    def test_n_theta_i_correct(self, game2):
        sol = solve_agent_mp(game2, 0,
                             algorithm=mpqp_algorithm.combinatorial,
                             verbose=False)
        expected = (game2.n_x_total - game2.agents[0].n_x) + game2.n_p
        assert sol.n_theta_i == expected

    def test_best_response_feasible(self, game2):
        """x_i* from a CR lookup must satisfy all agent i constraints."""
        sol = solve_agent_mp(game2, 0,
                             algorithm=mpqp_algorithm.combinatorial,
                             verbose=False)
        G, b, F = build_constraint_matrices(game2, 0)
        # Pick a θ inside the first CR
        cr0 = sol.regions[0]
        # Find a feasible interior point: use the Chebyshev center approach
        # (simple: try θ=0 and check)
        theta = np.zeros(sol.n_theta_i)
        v = sol.locate(theta)
        if v is None:
            pytest.skip("θ=0 not in any CR for this seed")
        x_i = sol.regions[v].evaluate(theta)
        lhs = G @ x_i
        rhs = b + F @ theta
        assert np.all(lhs <= rhs + 1e-6), "Best response violates constraints"

    def test_both_agents_solved(self, game2):
        sols = solve_all_agents_mp(game2,
                                   algorithm=mpqp_algorithm.combinatorial,
                                   verbose=False)
        assert len(sols) == 2
        assert all(s.n_cr >= 1 for s in sols)

    def test_agent_indices_match(self, game2):
        sols = solve_all_agents_mp(game2,
                                   algorithm=mpqp_algorithm.combinatorial,
                                   verbose=False)
        for i, s in enumerate(sols):
            assert s.agent_index == i

    def test_vector_game_solves(self, game3):
        """N=3, n_x=2 — verify all three agents get CRs."""
        sols = solve_all_agents_mp(game3,
                                   algorithm=mpqp_algorithm.combinatorial,
                                   verbose=False)
        assert len(sols) == game3.N
        for s in sols:
            assert s.n_cr >= 1


# ---------------------------------------------------------------------------
# Coupling-mode constraint structure (fast — no PPOPT solve)
# ---------------------------------------------------------------------------

class TestCouplingModeConstraints:
    """Verify that build_constraint_matrices produces the right structure
    for both coupling_mode="state_bounds" and coupling_mode="l_max"."""

    def test_state_bounds_no_coupling_block(self, game_state_bounds):
        """state_bounds game has n_coupling=0; no coupling rows in G."""
        assert game_state_bounds.n_coupling == 0
        for i in range(game_state_bounds.N):
            ai = game_state_bounds.agents[i]
            assert ai.C is None
            assert ai.has_state_constraints

    def test_state_bounds_G_has_state_rows(self, game_state_bounds, plant2):
        """G must include ±Gamma_i rows for each agent (2*Np*nx rows)."""
        Np, nx = plant2.Np, plant2.nx
        for i in range(game_state_bounds.N):
            ai = game_state_bounds.agents[i]
            n_loc = ai.n_loc                        # 2*Np*nu_i
            n_state = 2 * Np * nx                   # ±Gamma_i rows
            G, b, F = build_constraint_matrices(game_state_bounds, i)
            assert G.shape[0] == n_loc + n_state

    def test_lmax_has_coupling_block(self, game_lmax, plant2):
        """l_max game has n_coupling=Np; each agent has C != None."""
        assert game_lmax.n_coupling == plant2.Np
        for i in range(game_lmax.N):
            ai = game_lmax.agents[i]
            assert ai.C is not None
            assert not ai.has_state_constraints

    def test_lmax_G_has_coupling_rows(self, game_lmax, plant2):
        """G must include Np coupling rows (one per prediction step)."""
        Np = plant2.Np
        for i in range(game_lmax.N):
            ai = game_lmax.agents[i]
            n_loc = ai.n_loc                        # 2*Np*nu_i
            G, b, F = build_constraint_matrices(game_lmax, i)
            assert G.shape[0] == n_loc + Np

    def test_lmax_d_shape(self, game_lmax, plant2):
        """Coupling RHS d must have length Np."""
        assert game_lmax.d.shape == (plant2.Np,)
        assert np.all(game_lmax.d == 5.0)          # L_max=5.0 in fixture

    def test_state_bounds_F_has_M_theta_rows(self, game_state_bounds, plant2):
        """F parametric block for state rows must have shape (2*Np*nx, n_theta)."""
        Np, nx = plant2.Np, plant2.nx
        for i in range(game_state_bounds.N):
            ai = game_state_bounds.agents[i]
            n_loc = ai.n_loc
            _, _, F = build_constraint_matrices(game_state_bounds, i)
            n_state_rows = 2 * Np * nx
            assert F[n_loc: n_loc + n_state_rows].shape[0] == n_state_rows

    def test_invalid_coupling_mode_raises(self, plant2):
        Q, R, P = default_local_weights(plant2)
        with pytest.raises(ValueError, match="coupling_mode"):
            make_gne_game_from_plant(plant2, coupling_mode="bad",
                                     Q_list=Q, R_list=R, P_list=P)


# ---------------------------------------------------------------------------
# Full PPOPT solve with plant-based games (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSolveAgentMpPlant:

    def test_state_bounds_solves(self, game_state_bounds):
        """state_bounds MPC game: all agents get at least one CR."""
        sols = solve_all_agents_mp(game_state_bounds,
                                   algorithm=mpqp_algorithm.combinatorial,
                                   verbose=False)
        assert len(sols) == game_state_bounds.N
        assert all(s.n_cr >= 1 for s in sols)

    def test_lmax_solves(self, game_lmax):
        """l_max MPC game: all agents get at least one CR."""
        sols = solve_all_agents_mp(game_lmax,
                                   algorithm=mpqp_algorithm.combinatorial,
                                   verbose=False)
        assert len(sols) == game_lmax.N
        assert all(s.n_cr >= 1 for s in sols)
