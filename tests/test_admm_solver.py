"""
Tests for mpgne/admm_solver.py

Fast tests  (no PPOPT, hand-crafted or tiny game):  run normally
Slow tests  (ADMM vs explicit GNE from gne_combiner): @pytest.mark.slow

Run fast only:  python -m pytest tests/test_admm_solver.py -v -m "not slow"
Run all:        python -m pytest tests/test_admm_solver.py -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpgne.game import Agent, GNEGame, make_random_game
from mpgne.admm_solver import (
    ADMMResult,
    _solve_agent_xupdate,
    _z_update,
    _lambda_update,
    _compute_residuals,
    admm_solve,
    admm_solve_grid,
)


# ---------------------------------------------------------------------------
# Minimal hand-crafted 2-agent game (same as test_gne_combiner.py)
# ---------------------------------------------------------------------------

def _make_hand_game(n_p=1):
    def _agent(idx):
        return Agent(
            index=idx, n_x=1,
            Q=np.array([[2.0]]),        # PD
            c=np.zeros(1),
            F=np.zeros((1, n_p)),
            C=np.ones((1, 1)),          # coupling: sum x_i ≤ d
            A_loc=np.vstack([np.eye(1), -np.eye(1)]),
            b_loc=10.0 * np.ones(2),
            S_loc=np.zeros((2, n_p)),
        )
    agents = [_agent(i) for i in range(2)]
    return GNEGame(
        agents=agents,
        d=np.array([3.0]),              # x_0 + x_1 ≤ 3
        S_coup=np.zeros((1, n_p)),
        p_lb=-5.0 * np.ones(n_p),
        p_ub= 5.0 * np.ones(n_p),
    )


def _make_unconstrained_game(n_p=1):
    """Coupling constraint is loose — each agent's unconstrained min satisfies it."""
    def _agent(idx):
        return Agent(
            index=idx, n_x=1,
            Q=np.array([[2.0]]),
            c=np.array([1.0]),          # unconstrained min at x=-0.5
            F=np.zeros((1, n_p)),
            C=np.ones((1, 1)),
            A_loc=np.vstack([np.eye(1), -np.eye(1)]),
            b_loc=10.0 * np.ones(2),
            S_loc=np.zeros((2, n_p)),
        )
    agents = [_agent(i) for i in range(2)]
    return GNEGame(
        agents=agents,
        d=np.array([20.0]),             # very loose: x_0+x_1 ≤ 20
        S_coup=np.zeros((1, n_p)),
        p_lb=-5.0 * np.ones(n_p),
        p_ub= 5.0 * np.ones(n_p),
    )


# ---------------------------------------------------------------------------
# ADMMResult dataclass
# ---------------------------------------------------------------------------

class TestADMMResult:

    def test_x_stacked_shape(self):
        x_sol = [np.array([1.0]), np.array([2.0])]
        res = ADMMResult(
            x_sol=x_sol, z_sol=[np.zeros(1)]*2,
            lambda_sol=[np.zeros(1)]*2,
            n_iter=10, converged=True,
            primal_res=1e-5, dual_res=1e-5, coupling_violation=0.0,
        )
        np.testing.assert_array_equal(res.x_stacked, [1.0, 2.0])

    def test_fields_accessible(self):
        res = ADMMResult(
            x_sol=[np.zeros(1)], z_sol=[np.zeros(1)],
            lambda_sol=[np.zeros(1)],
            n_iter=5, converged=False,
            primal_res=1.0, dual_res=0.5, coupling_violation=0.1,
            primal_hist=[1.0, 0.5], dual_hist=[0.5, 0.1],
            solve_time=0.01,
        )
        assert res.n_iter == 5
        assert res.converged is False
        assert len(res.primal_hist) == 2


# ---------------------------------------------------------------------------
# _solve_agent_xupdate
# ---------------------------------------------------------------------------

class TestSolveAgentXUpdate:

    def test_returns_correct_shape(self):
        game = _make_hand_game()
        x = _solve_agent_xupdate(game, 0, np.zeros(1),
                                  z_i=np.zeros(1), lambda_i=np.zeros(1), rho=1.0)
        assert x.shape == (1,)

    def test_unconstrained_optimal(self):
        """With no coupling penalty and loose constraints, x_i* = -Q^{-1} c."""
        game = _make_unconstrained_game()
        p = np.zeros(1)
        # J_i = x^2 + x → min at x = -0.5
        x = _solve_agent_xupdate(game, 0, p, z_i=np.zeros(1),
                                  lambda_i=np.zeros(1), rho=1e-6)   # tiny rho
        np.testing.assert_allclose(x, [-0.5], atol=1e-4)

    def test_satisfies_local_constraints(self):
        game = _make_hand_game()
        p = np.zeros(1)
        x = _solve_agent_xupdate(game, 0, p, z_i=np.zeros(1),
                                  lambda_i=np.zeros(1), rho=1.0)
        ai = game.agents[0]
        rhs = ai.b_loc + ai.S_loc @ p
        assert np.all(ai.A_loc @ x <= rhs + 1e-6)

    def test_warm_start_accepted(self):
        game = _make_hand_game()
        p = np.zeros(1)
        x0 = np.array([1.0])
        x = _solve_agent_xupdate(game, 0, p, z_i=np.zeros(1),
                                  lambda_i=np.zeros(1), rho=1.0, x0=x0)
        assert x.shape == (1,)


# ---------------------------------------------------------------------------
# _z_update
# ---------------------------------------------------------------------------

class TestZUpdate:

    def test_returns_list_of_correct_length(self):
        game = _make_hand_game()
        x = [np.zeros(1), np.zeros(1)]
        lam = [np.zeros(1), np.zeros(1)]
        z = _z_update(game, np.zeros(1), x, lam, rho=1.0)
        assert len(z) == 2

    def test_z_shape(self):
        game = _make_hand_game()
        x = [np.zeros(1), np.zeros(1)]
        lam = [np.zeros(1), np.zeros(1)]
        z = _z_update(game, np.zeros(1), x, lam, rho=1.0)
        assert z[0].shape == (1,)

    def test_no_excess_unconstrained(self):
        """When agg < rhs, z_i = C_i x_i + λ/ρ (no shift, Boyd ADMM sign)."""
        game = _make_hand_game()
        p = np.zeros(1)
        x = [np.array([-2.0]), np.array([-2.0])]   # sum_unc = -4 < 3 = rhs
        lam = [np.zeros(1), np.zeros(1)]
        z = _z_update(game, p, x, lam, rho=1.0)
        # z_unc = C x + λ/ρ = -2 + 0 = -2
        np.testing.assert_allclose(z[0], [-2.0], atol=1e-12)
        np.testing.assert_allclose(z[1], [-2.0], atol=1e-12)

    def test_excess_shifts_uniformly(self):
        """When sum z_unc > rhs, each z_i shifts by -excess/N."""
        game = _make_hand_game()   # rhs = 3
        p = np.zeros(1)
        x = [np.array([3.0]), np.array([3.0])]   # z_unc = 3, sum = 6 > 3
        lam = [np.zeros(1), np.zeros(1)]
        z = _z_update(game, p, x, lam, rho=1.0)
        # excess = 6 - 3 = 3, shift = 3/2 = 1.5 → z_i = 1.5
        np.testing.assert_allclose(sum(z), [3.0], atol=1e-10)
        np.testing.assert_allclose(z[0], [1.5], atol=1e-10)

    def test_coupling_satisfied_after_z_update(self):
        """After z-update, Σ z_i ≤ d + S_coup p."""
        game = _make_hand_game()
        p = np.zeros(1)
        # z_unc = C x + λ/ρ = 5 + 0 = 5 each; sum = 10 > 3
        x = [np.array([5.0]), np.array([5.0])]
        lam = [np.zeros(1), np.zeros(1)]
        z = _z_update(game, p, x, lam, rho=1.0)
        agg = sum(z)
        rhs = game.d + game.S_coup @ p
        assert np.all(agg <= rhs + 1e-10)


# ---------------------------------------------------------------------------
# _lambda_update
# ---------------------------------------------------------------------------

class TestLambdaUpdate:

    def test_lambda_update_zero_residual(self):
        """If C_i x_i = z_i, λ stays unchanged."""
        game = _make_hand_game()
        x = [np.array([1.5]), np.array([1.5])]
        z = [game.agents[i].C @ x[i] for i in range(2)]   # C_i x_i = z_i
        lam = [np.array([0.5]), np.array([0.3])]
        new_lam = _lambda_update(game, x, z, lam, rho=2.0)
        np.testing.assert_allclose(new_lam[0], lam[0], atol=1e-12)
        np.testing.assert_allclose(new_lam[1], lam[1], atol=1e-12)

    def test_lambda_update_nonzero_residual(self):
        """λ_i^{k+1} = λ_i^k + ρ (C_i x_i - z_i)  (sign-correct Boyd formula)."""
        game = _make_hand_game()
        x   = [np.array([2.0]), np.array([1.0])]
        z   = [np.array([1.0]), np.array([1.0])]   # residuals: [1, 0]
        lam = [np.array([0.5]), np.array([0.3])]
        new_lam = _lambda_update(game, x, z, lam, rho=2.0)
        np.testing.assert_allclose(new_lam[0], [0.5 + 2*(2-1)], atol=1e-12)  # 2.5
        np.testing.assert_allclose(new_lam[1], [0.3 + 2*(1-1)], atol=1e-12)  # 0.3


# ---------------------------------------------------------------------------
# _compute_residuals
# ---------------------------------------------------------------------------

class TestComputeResiduals:

    def test_zero_when_converged(self):
        game = _make_hand_game()
        x = [np.array([1.5]), np.array([1.5])]
        z = [game.agents[i].C @ x[i] for i in range(2)]
        z_prev = z
        r, s = _compute_residuals(game, x, z, z_prev, rho=1.0)
        assert abs(r) < 1e-12
        assert abs(s) < 1e-12

    def test_nonzero_residual(self):
        game = _make_hand_game()
        x = [np.array([2.0]), np.array([1.0])]
        z = [np.array([1.0]), np.array([1.0])]
        z_prev = [np.array([0.5]), np.array([0.5])]
        r, s = _compute_residuals(game, x, z, z_prev, rho=2.0)
        assert r > 0
        assert s > 0


# ---------------------------------------------------------------------------
# admm_solve — convergence and feasibility
# ---------------------------------------------------------------------------

class TestAdmmSolve:

    def test_returns_admm_result(self):
        game = _make_hand_game()
        res = admm_solve(game, np.zeros(1), max_iter=200, verbose=False)
        assert isinstance(res, ADMMResult)

    def test_x_sol_length(self):
        game = _make_hand_game()
        res = admm_solve(game, np.zeros(1), max_iter=200, verbose=False)
        assert len(res.x_sol) == game.N

    def test_x_sol_shapes(self):
        game = _make_hand_game()
        res = admm_solve(game, np.zeros(1), max_iter=200, verbose=False)
        for i, x_i in enumerate(res.x_sol):
            assert x_i.shape == (game.agents[i].n_x,)

    def test_converges(self):
        game = _make_hand_game()
        res = admm_solve(game, np.zeros(1), rho=1.0, max_iter=500, tol=1e-5)
        assert res.converged, f"Did not converge after {res.n_iter} iters"

    def test_coupling_satisfied(self):
        game = _make_hand_game()
        p = np.zeros(1)
        res = admm_solve(game, p, max_iter=500, tol=1e-5)
        assert res.coupling_violation < 1e-4

    def test_local_constraints_satisfied(self):
        game = _make_hand_game()
        p = np.zeros(1)
        res = admm_solve(game, p, max_iter=500, tol=1e-5)
        for i, x_i in enumerate(res.x_sol):
            ai = game.agents[i]
            rhs = ai.b_loc + ai.S_loc @ p
            assert np.all(ai.A_loc @ x_i <= rhs + 1e-4)

    def test_primal_hist_length(self):
        game = _make_hand_game()
        res = admm_solve(game, np.zeros(1), max_iter=100, tol=1e-10)
        assert len(res.primal_hist) == res.n_iter

    def test_primal_hist_decreasing_trend(self):
        game = _make_hand_game()
        # start from x=5 to ensure a non-trivial initial residual
        x_init = [np.array([5.0]), np.array([5.0])]
        res = admm_solve(game, np.zeros(1), max_iter=300, tol=1e-8, x_init=x_init)
        hist = res.primal_hist
        assert hist[-1] < hist[0]

    def test_unconstrained_game_matches_local_min(self):
        """When coupling is loose, x_i* = unconstrained minimum -Q^{-1}c."""
        game = _make_unconstrained_game()
        p = np.zeros(1)
        res = admm_solve(game, p, rho=0.1, max_iter=1000, tol=1e-6)
        # Unconstrained min for each agent: 2x + 1 = 0 → x = -0.5
        for x_i in res.x_sol:
            np.testing.assert_allclose(x_i, [-0.5], atol=1e-3)

    def test_different_p_gives_different_solution(self):
        """Parametric GNE: different p → different x*."""
        game = make_random_game(N=2, n_x=1, n_p=1, n_coupling=1, seed=1)
        # Set F_i non-zero so p affects the solution
        game.agents[0].F = np.array([[1.0]])
        game.agents[1].F = np.array([[0.5]])
        p1 = np.array([-2.0])
        p2 = np.array([ 2.0])
        res1 = admm_solve(game, p1, max_iter=500, tol=1e-5)
        res2 = admm_solve(game, p2, max_iter=500, tol=1e-5)
        assert not np.allclose(res1.x_stacked, res2.x_stacked, atol=1e-3)

    def test_warm_start_fewer_iters(self):
        """Warm-starting from solution should need ≤ iters than cold start."""
        game = _make_hand_game()
        p = np.zeros(1)
        cold = admm_solve(game, p, max_iter=500, tol=1e-6)
        warm = admm_solve(game, p, max_iter=500, tol=1e-6, x_init=cold.x_sol)
        assert warm.n_iter <= cold.n_iter


# ---------------------------------------------------------------------------
# admm_solve_grid
# ---------------------------------------------------------------------------

class TestAdmmSolveGrid:

    def test_returns_list_correct_length(self):
        game = _make_hand_game()
        p_grid = np.linspace(-2, 2, 5).reshape(-1, 1)
        results = admm_solve_grid(game, p_grid, max_iter=300, tol=1e-4)
        assert len(results) == 5

    def test_all_results_are_admm_result(self):
        game = _make_hand_game()
        p_grid = np.linspace(-1, 1, 3).reshape(-1, 1)
        results = admm_solve_grid(game, p_grid, max_iter=300, tol=1e-4)
        for r in results:
            assert isinstance(r, ADMMResult)

    def test_coupling_satisfied_across_grid(self):
        game = _make_hand_game()
        p_grid = np.linspace(-1, 1, 5).reshape(-1, 1)
        results = admm_solve_grid(game, p_grid, max_iter=500, tol=1e-5)
        for r in results:
            assert r.coupling_violation < 1e-3


# ---------------------------------------------------------------------------
# Slow: compare ADMM to explicit GNE from gne_combiner
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestAdmmVsExplicit:

    @pytest.fixture
    def game_and_gne(self):
        from mpgne.mp_solver import solve_all_agents_mp
        from mpgne.gne_combiner import build_gne_solution
        from ppopt.mp_solvers.solve_mpqp import mpqp_algorithm

        game = make_random_game(N=2, n_x=1, n_p=1, n_coupling=1,
                                x_bound=5.0, p_bound=3.0, seed=0)
        sols = solve_all_agents_mp(game, algorithm=mpqp_algorithm.combinatorial,
                                   verbose=False)
        gne = build_gne_solution(game, sols, verbose=False)
        return game, gne

    def test_admm_matches_explicit_at_p0(self, game_and_gne):
        """At p=0, ADMM solution should match explicit GNE within tolerance."""
        game, gne = game_and_gne
        p = np.zeros(game.n_p)
        k = gne.locate(p)
        if k is None:
            pytest.skip("p=0 not in any GNE CR")

        x_explicit = gne[k].evaluate(p)
        res = admm_solve(game, p, rho=1.0, max_iter=1000, tol=1e-6)
        assert res.converged
        np.testing.assert_allclose(res.x_stacked, x_explicit, atol=1e-3)

    def test_admm_matches_explicit_on_grid(self, game_and_gne):
        """Over a grid of p values, ADMM and explicit agree where CR exists."""
        game, gne = game_and_gne
        p_vals = np.linspace(-2, 2, 10)
        matches = 0
        for pv in p_vals:
            p = np.array([pv])
            k = gne.locate(p)
            if k is None:
                continue
            x_exp = gne[k].evaluate(p)
            res = admm_solve(game, p, rho=1.0, max_iter=1000, tol=1e-6)
            if res.converged:
                if np.allclose(res.x_stacked, x_exp, atol=1e-2):
                    matches += 1
        assert matches >= 3, f"Only {matches}/10 grid points matched explicit"


# ---------------------------------------------------------------------------
# ADMM on plant-based MPC games — both coupling modes
# ---------------------------------------------------------------------------

def _make_mpc_game(coupling_mode: str, L_max: float = 5.0):
    """Small M=2 MPC game with the requested coupling formulation."""
    from mpgne.plant_gen import make_random_plants
    from mpgne.mpc_builder import make_gne_game_from_plant, default_local_weights
    plant = make_random_plants(2, 1, seed=77)[0]
    Q, R, P = default_local_weights(plant)
    game = make_gne_game_from_plant(plant, coupling_mode=coupling_mode, L_max=L_max,
                                    Q_list=Q, R_list=R, P_list=P)
    # Return plant too so caller can build a valid p inside state bounds
    return plant, game


class TestAdmmPlantCouplingModes:

    def test_state_bounds_converges(self):
        """ADMM must converge on a state_bounds MPC game."""
        plant, game = _make_mpc_game("state_bounds")
        p = np.concatenate([s.x_lb * 0.1 for s in plant.subsystems])
        res = admm_solve(game, p, rho=1.0, max_iter=2000, tol=1e-4, verbose=False)
        assert res.converged, f"ADMM did not converge (state_bounds): {res.n_iter} iters"

    def test_lmax_converges(self):
        """ADMM must converge on an l_max MPC game."""
        plant, game = _make_mpc_game("l_max", L_max=5.0)
        p = np.concatenate([s.x_lb * 0.1 for s in plant.subsystems])
        res = admm_solve(game, p, rho=1.0, max_iter=2000, tol=1e-4, verbose=False)
        assert res.converged, f"ADMM did not converge (l_max): {res.n_iter} iters"

    def test_state_bounds_local_constraints_satisfied(self):
        """All local input-bound constraints are met at the ADMM solution."""
        plant, game = _make_mpc_game("state_bounds")
        p = np.concatenate([s.x_lb * 0.05 for s in plant.subsystems])
        res = admm_solve(game, p, rho=1.0, max_iter=2000, tol=1e-4, verbose=False)
        for i, ai in enumerate(game.agents):
            x_i = res.x_sol[i]
            rhs = ai.b_loc + ai.S_loc @ p
            assert np.all(ai.A_loc @ x_i <= rhs + 1e-4), \
                f"Agent {i} violates local constraints (state_bounds)"

    def test_lmax_coupling_satisfied(self):
        """Aggregate-input coupling Σ C_i x_i ≤ d is met at the ADMM solution."""
        plant, game = _make_mpc_game("l_max", L_max=5.0)
        p = np.concatenate([s.x_lb * 0.05 for s in plant.subsystems])
        res = admm_solve(game, p, rho=1.0, max_iter=2000, tol=1e-4, verbose=False)
        assert game.coupling_feasible(res.x_stacked, p, tol=1e-3), \
            "L_max coupling constraint violated at ADMM solution"

    def test_mode_affects_n_coupling(self):
        """state_bounds → n_coupling=0; l_max → n_coupling=Np."""
        plant, g_sb = _make_mpc_game("state_bounds")
        _,     g_lm = _make_mpc_game("l_max")
        assert g_sb.n_coupling == 0
        assert g_lm.n_coupling == plant.Np
