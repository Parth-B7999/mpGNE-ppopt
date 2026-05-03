"""
Tests for mpgne/game.py — Agent, GNEGame, make_random_game.

Run with:
    cd mpgne_ppopt
    python -m pytest tests/test_game.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpgne.game import make_random_game


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_game():
    """2-agent, scalar decision, scalar parameter, scalar coupling."""
    return make_random_game(N=2, n_x=1, n_p=1, n_coupling=1, seed=0)


@pytest.fixture
def vector_game():
    """3-agent, 2D decision, 2D parameter, 2 coupling constraints."""
    return make_random_game(N=3, n_x=2, n_p=2, n_coupling=2, seed=42)


# ---------------------------------------------------------------------------
# Agent shape tests
# ---------------------------------------------------------------------------

class TestAgentShapes:

    def test_Q_is_square_and_pd(self, simple_game):
        for a in simple_game.agents:
            assert a.Q.shape == (a.n_x, a.n_x)
            eigvals = np.linalg.eigvalsh(a.Q)
            assert eigvals.min() > 0, "Q must be positive definite"

    def test_c_shape(self, simple_game):
        for a in simple_game.agents:
            assert a.c.shape == (a.n_x,)

    def test_F_shape(self, simple_game):
        for a in simple_game.agents:
            assert a.F.shape == (a.n_x, simple_game.n_p)

    def test_C_shape(self, simple_game):
        for a in simple_game.agents:
            assert a.C.shape == (simple_game.n_coupling, a.n_x)

    def test_A_loc_shape(self, simple_game):
        for a in simple_game.agents:
            assert a.A_loc.shape == (a.n_loc, a.n_x)
            assert a.b_loc.shape == (a.n_loc,)
            assert a.S_loc.shape == (a.n_loc, simple_game.n_p)

    def test_properties_consistent(self, vector_game):
        for a in vector_game.agents:
            assert a.n_coupling == vector_game.n_coupling
            assert a.n_p == vector_game.n_p


# ---------------------------------------------------------------------------
# GNEGame structure tests
# ---------------------------------------------------------------------------

class TestGNEGameStructure:

    def test_N_agents(self, simple_game):
        assert simple_game.N == 2

    def test_n_p(self, simple_game):
        assert simple_game.n_p == 1

    def test_n_coupling(self, simple_game):
        assert simple_game.n_coupling == 1

    def test_n_x_total(self, simple_game):
        # 2 agents × 1 decision each = 2
        assert simple_game.n_x_total == 2

    def test_x_slice_non_overlapping(self, vector_game):
        slices = [vector_game.x_slice(i) for i in range(vector_game.N)]
        indices = []
        for s in slices:
            indices.extend(range(s.start, s.stop))
        assert len(indices) == len(set(indices)), "x slices must not overlap"
        assert len(indices) == vector_game.n_x_total

    def test_x_slice_covers_all(self, vector_game):
        total = 0
        for i in range(vector_game.N):
            s = vector_game.x_slice(i)
            total += s.stop - s.start
        assert total == vector_game.n_x_total

    def test_d_shape(self, simple_game):
        assert simple_game.d.shape == (simple_game.n_coupling,)

    def test_S_coup_shape(self, vector_game):
        assert vector_game.S_coup.shape == (vector_game.n_coupling, vector_game.n_p)

    def test_p_bounds_shape(self, vector_game):
        assert vector_game.p_lb.shape == (vector_game.n_p,)
        assert vector_game.p_ub.shape == (vector_game.n_p,)
        assert np.all(vector_game.p_lb <= vector_game.p_ub)


# ---------------------------------------------------------------------------
# Feasibility and cost tests
# ---------------------------------------------------------------------------

class TestFeasibilityAndCost:

    def test_zero_is_locally_feasible(self, simple_game):
        p = np.zeros(simple_game.n_p)
        for a in simple_game.agents:
            x_i = np.zeros(a.n_x)
            assert a.local_feasible(x_i, p), "x=0 must satisfy box constraints"

    def test_zero_x_coupling_feasible(self, simple_game):
        x = np.zeros(simple_game.n_x_total)
        p = np.zeros(simple_game.n_p)
        assert simple_game.coupling_feasible(x, p), "x=0 must satisfy coupling"

    def test_all_feasible_at_zero(self, vector_game):
        x = np.zeros(vector_game.n_x_total)
        p = np.zeros(vector_game.n_p)
        assert vector_game.all_feasible(x, p)

    def test_infeasible_outside_box(self, simple_game):
        p = np.zeros(simple_game.n_p)
        x_i = np.array([999.0])  # way outside box bound of 10
        assert not simple_game.agents[0].local_feasible(x_i, p)

    def test_cost_is_scalar(self, vector_game):
        p = np.zeros(vector_game.n_p)
        x = np.zeros(vector_game.n_x_total)
        cost = vector_game.total_cost(x, p)
        assert isinstance(cost, float)

    def test_cost_zero_at_zero_when_c_is_zero(self, simple_game):
        # c_i = 0, x_i = 0 => J_i = 0
        p = np.zeros(simple_game.n_p)
        x = np.zeros(simple_game.n_x_total)
        assert abs(simple_game.total_cost(x, p)) < 1e-12

    def test_coupling_lhs_shape(self, vector_game):
        x = np.zeros(vector_game.n_x_total)
        lhs = vector_game.coupling_lhs(x)
        assert lhs.shape == (vector_game.n_coupling,)


# ---------------------------------------------------------------------------
# make_random_game reproducibility
# ---------------------------------------------------------------------------

class TestRandomGame:

    def test_same_seed_reproducible(self):
        g1 = make_random_game(N=3, n_x=2, n_p=2, seed=7)
        g2 = make_random_game(N=3, n_x=2, n_p=2, seed=7)
        for a1, a2 in zip(g1.agents, g2.agents):
            np.testing.assert_array_equal(a1.Q, a2.Q)

    def test_different_seeds_differ(self):
        g1 = make_random_game(N=2, n_x=1, seed=1)
        g2 = make_random_game(N=2, n_x=1, seed=2)
        # Q matrices should differ
        assert not np.allclose(g1.agents[0].Q, g2.agents[0].Q)

    def test_large_game_valid(self):
        g = make_random_game(N=5, n_x=3, n_p=4, n_coupling=2, seed=99)
        assert g.N == 5
        assert g.n_x_total == 15
        assert g.n_p == 4
        assert g.n_coupling == 2
        for a in g.agents:
            assert np.linalg.eigvalsh(a.Q).min() > 0

    def test_agent_indices_correct(self):
        g = make_random_game(N=4, n_x=1, seed=0)
        for i, a in enumerate(g.agents):
            assert a.index == i

    def test_coupling_sum_correct(self):
        # With default equal weights C_i = (1/N)*ones, sum_i C_i x_i = mean(x)
        g = make_random_game(N=2, n_x=1, n_coupling=1, coupling_scale=1.0, seed=0)
        x = np.array([4.0, 6.0])   # sum = 10, coupling lhs = 0.5*4 + 0.5*6 = 5
        lhs = g.coupling_lhs(x)
        np.testing.assert_allclose(lhs, [5.0], atol=1e-12)
