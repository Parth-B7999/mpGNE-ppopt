"""
Tests for mpgne/cr_store.py

Run with:
    cd mpgne_ppopt
    python -m pytest tests/test_cr_store.py -v
"""

import numpy as np
import pytest
import pickle
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpgne.cr_store import (
    AgentCR, AgentSolution,
    GNECriticalRegion, GNESolution,
    agent_solution_from_ppopt,
    save_agent_solutions, load_agent_solutions,
    save_gne_solution, load_gne_solution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent_cr(n_theta=3, n_x=1, index=0):
    """1-D box CR: -5 <= theta <= 5 for each dim; solution x* = theta[0]."""
    E = np.vstack([np.eye(n_theta), -np.eye(n_theta)])  # (2n_theta, n_theta)
    f = 5.0 * np.ones(2 * n_theta)
    A = np.zeros((n_x, n_theta))
    A[0, 0] = 1.0  # x* = theta_0
    b = np.zeros(n_x)
    return AgentCR(E=E, f=f, A=A, b=b, index=index)


def make_agent_solution(n_cr=3, n_theta=3, n_x=1, agent_index=0):
    """AgentSolution with n_cr non-overlapping 1-D CRs."""
    regions = []
    edges = np.linspace(-10, 10, n_cr + 1)
    for v in range(n_cr):
        lo, hi = edges[v], edges[v + 1]
        E = np.array([[1.0], [-1.0]])                 # 1-D, n_theta=1
        f = np.array([hi, -lo])
        A = np.eye(n_x, 1)
        b = np.zeros(n_x)
        regions.append(AgentCR(E=E, f=f, A=A, b=b, index=v))
    return AgentSolution(
        agent_index=agent_index,
        n_x_i=n_x,
        n_theta_i=1,
        regions=regions,
    )


def make_gne_cr(n_p=2, n_x_total=2, combination=(0, 1)):
    """A simple GNE CR: p-box [-5,5]^2, x*(p) = 0.5*p."""
    D = np.vstack([np.eye(n_p), -np.eye(n_p)])
    e = 5.0 * np.ones(2 * n_p)
    H_x = 0.5 * np.eye(n_x_total, n_p)
    h_x = np.zeros(n_x_total)
    Mx  = np.eye(n_x_total)
    Mp  = 0.5 * np.eye(n_x_total, n_p)
    M1  = np.zeros(n_x_total)
    return GNECriticalRegion(
        combination=combination, D=D, e=e,
        H_x=H_x, h_x=h_x,
        Mx=Mx, Mp=Mp, M1=M1, is_unique=True,
    )


# ---------------------------------------------------------------------------
# AgentCR tests
# ---------------------------------------------------------------------------

class TestAgentCR:

    def test_shapes_after_init(self):
        cr = make_agent_cr(n_theta=3, n_x=2)
        assert cr.E.shape == (6, 3)
        assert cr.f.shape == (6,)
        assert cr.A.shape == (2, 3)
        assert cr.b.shape == (2,)

    def test_contains_inside(self):
        cr = make_agent_cr(n_theta=2)
        assert cr.contains(np.array([1.0, -2.0]))

    def test_contains_outside(self):
        cr = make_agent_cr(n_theta=2)
        assert not cr.contains(np.array([6.0, 0.0]))  # 6 > 5

    def test_contains_on_boundary(self):
        cr = make_agent_cr(n_theta=1)
        assert cr.contains(np.array([5.0]))    # exactly on boundary

    def test_evaluate_shape(self):
        cr = make_agent_cr(n_theta=3, n_x=1)
        result = cr.evaluate(np.array([2.0, 1.0, 0.5]))
        assert result.shape == (1,)

    def test_evaluate_value(self):
        cr = make_agent_cr(n_theta=3, n_x=1)
        # A[0,0]=1, rest 0; b=0 → x* = theta[0]
        result = cr.evaluate(np.array([3.14, 0.0, 0.0]))
        np.testing.assert_allclose(result, [3.14])

    def test_properties(self):
        cr = make_agent_cr(n_theta=4, n_x=2)
        assert cr.n_theta == 4
        assert cr.n_x == 2
        assert cr.n_ineq == 8

    def test_facet_neighbors_default_empty(self):
        cr = make_agent_cr()
        assert cr.facet_neighbors == []


# ---------------------------------------------------------------------------
# AgentSolution tests
# ---------------------------------------------------------------------------

class TestAgentSolution:

    def test_len(self):
        sol = make_agent_solution(n_cr=4)
        assert len(sol) == 4

    def test_indexing(self):
        sol = make_agent_solution(n_cr=3)
        assert isinstance(sol[0], AgentCR)
        assert sol[0].index == 0

    def test_n_cr(self):
        sol = make_agent_solution(n_cr=5)
        assert sol.n_cr == 5

    def test_locate_finds_correct_cr(self):
        sol = make_agent_solution(n_cr=3)  # CRs cover (-10,-3.33), (-3.33,3.33), (3.33,10)
        v = sol.locate(np.array([0.0]))    # should be CR 1 (middle)
        assert v == 1

    def test_locate_returns_none_outside(self):
        sol = make_agent_solution(n_cr=3)
        v = sol.locate(np.array([999.0]))
        assert v is None

    def test_evaluate_returns_array(self):
        sol = make_agent_solution(n_cr=3, n_x=1)
        result = sol.evaluate(np.array([0.0]))
        assert result is not None
        assert result.shape == (1,)

    def test_evaluate_none_outside(self):
        sol = make_agent_solution(n_cr=2)
        assert sol.evaluate(np.array([999.0])) is None

    def test_cr_indices_sequential(self):
        sol = make_agent_solution(n_cr=4)
        for v, cr in enumerate(sol.regions):
            assert cr.index == v


# ---------------------------------------------------------------------------
# GNECriticalRegion tests
# ---------------------------------------------------------------------------

class TestGNECriticalRegion:

    def test_shapes(self):
        cr = make_gne_cr(n_p=2, n_x_total=2)
        assert cr.D.shape == (4, 2)
        assert cr.e.shape == (4,)
        assert cr.H_x.shape == (2, 2)
        assert cr.h_x.shape == (2,)

    def test_contains_inside(self):
        cr = make_gne_cr()
        assert cr.contains(np.array([1.0, 2.0]))

    def test_contains_outside(self):
        cr = make_gne_cr()
        assert not cr.contains(np.array([10.0, 10.0]))

    def test_evaluate_shape(self):
        cr = make_gne_cr(n_p=2, n_x_total=2)
        x = cr.evaluate(np.array([2.0, 4.0]))
        assert x.shape == (2,)

    def test_evaluate_value(self):
        cr = make_gne_cr(n_p=2, n_x_total=2)
        # H_x = 0.5*I, h_x = 0 → x* = 0.5*p
        x = cr.evaluate(np.array([2.0, 4.0]))
        np.testing.assert_allclose(x, [1.0, 2.0])

    def test_residual_near_zero(self):
        cr = make_gne_cr(n_p=2, n_x_total=2)
        # Mx=I, Mp=0.5*I, M1=0 → residual = ||x* - 0.5*p|| = ||0.5p - 0.5p|| = 0
        p = np.array([3.0, -1.0])
        assert cr.residual(p) < 1e-10

    def test_combination_stored(self):
        cr = make_gne_cr(combination=(2, 0, 1))
        assert cr.combination == (2, 0, 1)

    def test_properties(self):
        cr = make_gne_cr(n_p=3, n_x_total=4)
        assert cr.n_p == 3
        assert cr.n_x_total == 4

    def test_is_unique_flag(self):
        cr = make_gne_cr()
        assert cr.is_unique is True


# ---------------------------------------------------------------------------
# GNESolution tests
# ---------------------------------------------------------------------------

class TestGNESolution:

    @pytest.fixture
    def sol(self):
        cr0 = make_gne_cr(n_p=1, n_x_total=2, combination=(0, 0))
        # CR0: -5 <= p <= 5, x* = 0.5*[p, p]
        cr1 = GNECriticalRegion(
            combination=(0, 1),
            D=np.array([[1.0], [-1.0]]),
            e=np.array([10.0, 0.0]),   # 0 <= p <= 10
            H_x=np.array([[1.0], [0.5]]),
            h_x=np.zeros(2),
            Mx=np.eye(2), Mp=np.array([[1.0], [0.5]]), M1=np.zeros(2),
            is_unique=True,
        )
        return GNESolution(regions=[cr0, cr1], n_p=1, N=2)

    def test_len(self, sol):
        assert len(sol) == 2

    def test_n_cr(self, sol):
        assert sol.n_cr == 2

    def test_n_unique(self, sol):
        assert sol.n_unique == 2

    def test_locate_first_match(self, sol):
        # p=2 is in both CR0 (-5..5) and CR1 (0..10)
        k = sol.locate(np.array([2.0]))
        assert k == 0   # first match

    def test_locate_all(self, sol):
        ks = sol.locate_all(np.array([2.0]))
        assert 0 in ks and 1 in ks

    def test_locate_none_outside(self, sol):
        assert sol.locate(np.array([20.0])) is None

    def test_evaluate_returns_array(self, sol):
        x = sol.evaluate(np.array([2.0]))
        assert x is not None
        assert x.shape == (2,)

    def test_evaluate_none_outside(self, sol):
        assert sol.evaluate(np.array([100.0])) is None

    def test_summary_string(self, sol):
        s = sol.summary()
        assert "GNESolution" in s
        assert "unique" in s


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_load_agent_solutions(self, tmp_path):
        sols = [make_agent_solution(n_cr=2, agent_index=i) for i in range(3)]
        path = str(tmp_path / "agents.pkl")
        save_agent_solutions(sols, path)
        loaded = load_agent_solutions(path)
        assert len(loaded) == 3
        assert loaded[1].agent_index == 1
        assert loaded[0].n_cr == 2

    def test_save_load_gne_solution(self, tmp_path):
        gne = GNESolution(
            regions=[make_gne_cr(n_p=2, n_x_total=2)],
            n_p=2, N=2,
        )
        path = str(tmp_path / "gne.pkl")
        save_gne_solution(gne, path)
        loaded = load_gne_solution(path)
        assert loaded.n_cr == 1
        assert loaded.n_p == 2

    def test_roundtrip_values_preserved(self, tmp_path):
        cr = make_gne_cr(n_p=2, n_x_total=2, combination=(3, 7))
        gne = GNESolution(regions=[cr], n_p=2, N=2)
        path = str(tmp_path / "gne2.pkl")
        save_gne_solution(gne, path)
        loaded = load_gne_solution(path)
        np.testing.assert_array_equal(
            loaded[0].H_x, cr.H_x
        )
        assert loaded[0].combination == (3, 7)
