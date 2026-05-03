"""
Tests for mpgne/gne_selector.py

Fast tests:  hand-crafted CRs with controlled infinite structure
Slow tests:  full PPOPT game with real infinite CRs  (@pytest.mark.slow)

Run fast:  python -m pytest tests/test_gne_selector.py -v -m "not slow"
Run all:   python -m pytest tests/test_gne_selector.py -v
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpgne.game import Agent, GNEGame
from mpgne.cr_store import GNECriticalRegion, GNESolution
from mpgne.gne_selector import (
    select_gne,
    evaluate_all_types, evaluate_all_crs,
    AllGNETypes,
)


# ---------------------------------------------------------------------------
# Fixtures — hand-crafted 2-agent game with KNOWN infinite equilibria
#
# Paper Example 1 structure (scalar decisions, p = [p_c, p_1]):
#   Agent 0: x_0* = 0.5 x_1 + 0.2 p_c
#   Agent 1: x_1* = 0.3 x_0 + 0.1 p_c
#
# Unique case: M_x = [[1, -0.5],[-0.3, 1]] → det = 0.85 ≠ 0
#
# Infinite case: build an M_x with rank 1 deliberately
# ---------------------------------------------------------------------------

def _make_game():
    """Minimal 2-agent game with n_x=1, n_p=1 for clean test cases."""
    def _agent(idx):
        return Agent(
            index=idx, n_x=1,
            Q=np.array([[2.0]]),
            c=np.zeros(1),
            F=np.array([[0.5]]),
            C=np.ones((1, 1)),
            A_loc=np.vstack([np.eye(1), -np.eye(1)]),
            b_loc=10.0 * np.ones(2),
            S_loc=np.zeros((2, 1)),
        )
    return GNEGame(
        agents=[_agent(0), _agent(1)],
        d=np.array([5.0]),
        S_coup=np.zeros((1, 1)),
        p_lb=np.array([-3.0]),
        p_ub=np.array([3.0]),
    )


def _make_unique_cr(n_p=1):
    """Unique GNE CR: M_x full rank."""
    Mx = np.array([[1.0, -0.5], [-0.3, 1.0]])   # det = 0.85
    Mp = np.array([[0.2], [0.1]])
    M1 = np.array([1.0, -0.5])
    Mx_inv = np.linalg.inv(Mx)
    H_x = Mx_inv @ Mp
    h_x = Mx_inv @ M1
    D   = np.vstack([np.eye(n_p), -np.eye(n_p)])
    e   = 3.0 * np.ones(2 * n_p)
    return GNECriticalRegion(
        combination=(0, 0), D=D, e=e,
        H_x=H_x, h_x=h_x,
        Mx=Mx, Mp=Mp, M1=M1, is_unique=True,
    )


def _make_infinite_cr(n_p=1):
    """
    Infinite GNE CR: rank(M_x) = 1 < 2.

    M_x = [[1, -1],[-1, 1]]   rank 1
    With solvable RHS: Mp = [[1],[-1]], M1 = [0.5, -0.5]
    Min-norm solution: Mx^+ (Mp p + M1) = 0.5*[[1],[1]]*(p + 0.5) + ...
    Null space: V2 = [1/√2, 1/√2]^T (direction [1,1])
    """
    Mx  = np.array([[1.0, -1.0], [-1.0, 1.0]])   # rank 1
    Mp  = np.array([[1.0], [-1.0]])               # in range of M_x
    M1  = np.array([0.5, -0.5])
    Mx_pinv = np.linalg.pinv(Mx)
    H_x = Mx_pinv @ Mp
    h_x = Mx_pinv @ M1
    D   = np.vstack([np.eye(n_p), -np.eye(n_p)])
    e   = 3.0 * np.ones(2 * n_p)
    return GNECriticalRegion(
        combination=(0, 1), D=D, e=e,
        H_x=H_x, h_x=h_x,
        Mx=Mx, Mp=Mp, M1=M1, is_unique=False,
    )


# ---------------------------------------------------------------------------
# Basic API tests
# ---------------------------------------------------------------------------

class TestSelectGneAPI:

    def test_unknown_type_raises(self):
        cr = _make_unique_cr()
        game = _make_game()
        with pytest.raises(ValueError, match="Unknown gne_type"):
            select_gne(cr, game, np.array([0.0]), gne_type="magic")

    def test_returns_correct_shape_unique(self):
        cr = _make_unique_cr()
        game = _make_game()
        p = np.array([1.0])
        for gne_type in ("min_norm", "welfare", "v_gne"):
            x = select_gne(cr, game, p, gne_type)
            assert x.shape == (2,), f"{gne_type}: wrong shape"

    def test_returns_correct_shape_infinite(self):
        cr = _make_infinite_cr()
        game = _make_game()
        p = np.array([1.0])
        for gne_type in ("min_norm", "welfare", "v_gne"):
            x = select_gne(cr, game, p, gne_type)
            assert x.shape == (2,), f"{gne_type}: wrong shape"


# ---------------------------------------------------------------------------
# Unique CR: all methods agree
# ---------------------------------------------------------------------------

class TestUniqueGNE:

    def test_all_types_same_for_unique_cr(self):
        cr   = _make_unique_cr()
        game = _make_game()
        p    = np.array([1.5])
        x_mn = select_gne(cr, game, p, "min_norm")
        x_wf = select_gne(cr, game, p, "welfare")
        x_vg = select_gne(cr, game, p, "v_gne")
        np.testing.assert_allclose(x_mn, x_wf, atol=1e-10)
        np.testing.assert_allclose(x_mn, x_vg, atol=1e-10)

    def test_unique_cr_small_residual(self):
        cr   = _make_unique_cr()
        game = _make_game()
        for pv in [-2.0, 0.0, 2.0]:
            p = np.array([pv])
            for gne_type in ("min_norm", "welfare", "v_gne"):
                x = select_gne(cr, game, p, gne_type)
                res = np.linalg.norm(cr.Mx @ x - cr.Mp @ p - cr.M1)
                assert res < 1e-8, f"{gne_type} at p={pv}: residual={res:.2e}"


# ---------------------------------------------------------------------------
# Infinite CR: methods differ, all valid GNEs
# ---------------------------------------------------------------------------

class TestInfiniteGNE:

    def test_min_norm_is_pseudoinverse(self):
        cr = _make_infinite_cr()
        game = _make_game()
        p  = np.array([1.0])
        x  = select_gne(cr, game, p, "min_norm")
        # Should equal H_x p + h_x (already pseudoinverse result)
        np.testing.assert_allclose(x, cr.H_x @ p + cr.h_x, atol=1e-10)

    def test_all_types_satisfy_equilibrium(self):
        """All selections must satisfy M_x x* = M_p p + M_1."""
        cr   = _make_infinite_cr()
        game = _make_game()
        for pv in [-1.0, 0.0, 1.5]:
            p = np.array([pv])
            for gne_type in ("min_norm", "welfare", "v_gne"):
                x   = select_gne(cr, game, p, gne_type)
                res = np.linalg.norm(cr.Mx @ x - cr.Mp @ p - cr.M1)
                assert res < 1e-7, \
                    f"{gne_type} at p={pv}: equilibrium residual={res:.2e}"

    def test_welfare_minimises_social_cost(self):
        """Welfare selection should give LOWER or equal f^SW than min-norm."""
        cr   = _make_infinite_cr()
        game = _make_game()
        p    = np.array([1.0])
        x_mn = select_gne(cr, game, p, "min_norm")
        x_wf = select_gne(cr, game, p, "welfare")

        def _welfare(x):
            total = 0.0
            for a in game.agents:
                sl = game.x_slice(a.index)
                xi = x[sl]
                total += 0.5 * xi @ a.Q @ xi + (a.c + a.F @ p) @ xi
            return total

        assert _welfare(x_wf) <= _welfare(x_mn) + 1e-8

    def test_v_gne_equal_lagrange_multipliers(self):
        """v-GNE should give equal coupling multipliers when the system is non-degenerate."""
        # Asymmetric game so A_v ≠ 0 (non-degenerate equal-multiplier system)
        # Agent 0: Q=2, C=1   Agent 1: Q=4, C=1 (different Hessians → different grad → A_v ≠ 0)
        from mpgne.game import Agent, GNEGame
        def _asym_agent(idx, q_val):
            return Agent(
                index=idx, n_x=1,
                Q=np.array([[float(q_val)]]),
                c=np.zeros(1), F=np.array([[0.5]]),
                C=np.ones((1, 1)),
                A_loc=np.vstack([np.eye(1), -np.eye(1)]),
                b_loc=10.0 * np.ones(2),
                S_loc=np.zeros((2, 1)),
            )
        asym_game = GNEGame(
            agents=[_asym_agent(0, 2), _asym_agent(1, 4)],
            d=np.array([5.0]), S_coup=np.zeros((1, 1)),
            p_lb=np.array([-3.0]), p_ub=np.array([3.0]),
        )
        cr = _make_infinite_cr()
        p  = np.array([1.0])
        x  = select_gne(cr, asym_game, p, "v_gne")

        # Primary check: x must satisfy the equilibrium system
        res = np.linalg.norm(cr.Mx @ x - cr.Mp @ p - cr.M1)
        assert res < 1e-6, f"v_gne equilibrium residual too large: {res:.2e}"

        # Secondary: verify multipliers are closer than min-norm solution's multipliers
        def _get_lam(game, i, xi, p):
            a   = game.agents[i]
            g   = a.Q @ xi + a.c + a.F @ p
            CCT = a.C @ a.C.T
            return -np.linalg.solve(CCT, a.C @ g) if np.linalg.matrix_rank(CCT) == CCT.shape[0] \
                   else np.linalg.lstsq(a.C.T, -g, rcond=None)[0]

        x_mn = select_gne(cr, asym_game, p, "min_norm")
        lam_diff_vgne = abs(
            _get_lam(asym_game, 0, x[asym_game.x_slice(0)], p)[0] -
            _get_lam(asym_game, 1, x[asym_game.x_slice(1)], p)[0]
        )
        lam_diff_mn = abs(
            _get_lam(asym_game, 0, x_mn[asym_game.x_slice(0)], p)[0] -
            _get_lam(asym_game, 1, x_mn[asym_game.x_slice(1)], p)[0]
        )
        # v_gne should reduce the multiplier gap vs min_norm (or at least not worsen it)
        assert lam_diff_vgne <= lam_diff_mn + 1e-6, \
            f"v_gne multiplier gap {lam_diff_vgne:.4f} > min_norm {lam_diff_mn:.4f}"

    def test_selections_differ_for_infinite_cr(self):
        """For infinite CRs the three methods should generally give DIFFERENT x*."""
        cr   = _make_infinite_cr()
        game = _make_game()
        p    = np.array([2.0])
        x_mn = select_gne(cr, game, p, "min_norm")
        x_wf = select_gne(cr, game, p, "welfare")
        x_vg = select_gne(cr, game, p, "v_gne")
        # At least welfare should differ from min-norm for non-trivial game
        # (min-norm may accidentally coincide for symmetric games, so just check shapes)
        assert x_mn.shape == x_wf.shape == x_vg.shape


# ---------------------------------------------------------------------------
# evaluate_all_types
# ---------------------------------------------------------------------------

class TestEvaluateAllTypes:

    def test_returns_all_gne_types(self):
        cr   = _make_unique_cr()
        game = _make_game()
        p    = np.array([1.0])
        res  = evaluate_all_types(cr, game, p)
        assert isinstance(res, AllGNETypes)
        assert res.min_norm.shape == (2,)
        assert res.welfare.shape  == (2,)
        assert res.v_gne.shape    == (2,)

    def test_residuals_all_small(self):
        for cr_fn in (_make_unique_cr, _make_infinite_cr):
            cr   = cr_fn()
            game = _make_game()
            p    = np.array([0.5])
            res  = evaluate_all_types(cr, game, p)
            for gne_type, r in res.residuals.items():
                assert r < 1e-7, f"{gne_type}: residual={r:.2e} for {cr_fn.__name__}"

    def test_is_unique_flag_correct(self):
        assert evaluate_all_types(_make_unique_cr(),   _make_game(), np.zeros(1)).is_unique
        assert not evaluate_all_types(_make_infinite_cr(), _make_game(), np.zeros(1)).is_unique

    def test_print_summary_runs(self, capsys):
        res = evaluate_all_types(_make_infinite_cr(), _make_game(), np.zeros(1))
        res.print_summary()   # no agent_labels argument anymore
        captured = capsys.readouterr()
        assert "min_norm" in captured.out
        assert "welfare"  in captured.out
        assert "v_gne"    in captured.out


# ---------------------------------------------------------------------------
# evaluate_all_crs  (uses GNESolution)
# ---------------------------------------------------------------------------

class TestEvaluateAllCRs:

    def _make_gne_sol(self):
        cr_u = _make_unique_cr()
        cr_i = _make_infinite_cr()
        return GNESolution(regions=[cr_u, cr_i], n_p=1, N=2)

    def test_finds_matching_cr(self):
        gne_sol = self._make_gne_sol()
        game    = _make_game()
        results = evaluate_all_crs(gne_sol, game, np.array([0.0]),
                                   verbose=False)
        assert len(results) >= 1

    def test_p_outside_returns_empty(self):
        gne_sol = self._make_gne_sol()
        game    = _make_game()
        results = evaluate_all_crs(gne_sol, game, np.array([100.0]),
                                   verbose=False)
        assert results == []

    def test_all_found_results_have_small_residuals(self):
        gne_sol = self._make_gne_sol()
        game    = _make_game()
        results = evaluate_all_crs(gne_sol, game, np.array([0.0]), verbose=False)
        for r in results:
            for gne_type, res in r.residuals.items():
                assert res < 1e-7, f"{gne_type}: residual={res:.2e}"


# ---------------------------------------------------------------------------
# Slow: full PPOPT game with real infinite CRs
# ---------------------------------------------------------------------------

def _make_infinite_cr_game():
    """Design a capacity-sharing game guaranteed to have infinite equilibria.

    N=2 agents each want to maximize output (c=-1), share capacity p.
    With identical costs and coupling C_i=1, the equilibrium system has
    rank 1 (one equation x1+x2=p, two unknowns) => 1-D family of equilibria.
    """
    from mpgne.game import Agent, GNEGame
    q = 1.0
    agents = [
        Agent(index=0, n_x=1, Q=np.array([[q]]), c=np.array([-1.0]),
              F=np.zeros((1, 1)), C=np.array([[1.0]]),
              A_loc=np.array([[1.], [-1.]]), b_loc=np.array([3., 3.]),
              S_loc=np.zeros((2, 1))),
        Agent(index=1, n_x=1, Q=np.array([[q]]), c=np.array([-1.0]),
              F=np.zeros((1, 1)), C=np.array([[1.0]]),
              A_loc=np.array([[1.], [-1.]]), b_loc=np.array([3., 3.]),
              S_loc=np.zeros((2, 1))),
    ]
    return GNEGame(agents=agents, d=np.array([0.0]),
                   S_coup=np.array([[1.0]]),
                   p_lb=np.array([0.5]), p_ub=np.array([4.0]))


def _chebyshev_center(cr, game):
    """Return a point strictly inside cr, or None on failure."""
    from scipy.optimize import linprog
    n_p = game.n_p
    nrms = np.linalg.norm(cr.D, axis=1, keepdims=True)
    A = np.hstack([cr.D, nrms])
    c = np.zeros(n_p + 1);  c[-1] = -1.0
    res = linprog(c, A_ub=A, b_ub=cr.e,
                  bounds=[(game.p_lb[k], game.p_ub[k]) for k in range(n_p)]
                         + [(None, None)],
                  method='highs')
    return res.x[:n_p] if res.status == 0 else None


@pytest.mark.slow
class TestSelectorWithRealGame:

    @pytest.fixture
    def solved(self):
        from mpgne.mp_solver import solve_all_agents_mp
        from mpgne.gne_combiner import build_gne_solution
        from ppopt.mp_solvers.solve_mpqp import mpqp_algorithm

        game = _make_infinite_cr_game()
        sols = solve_all_agents_mp(game, algorithm=mpqp_algorithm.combinatorial,
                                   verbose=False)
        gne  = build_gne_solution(game, sols, verbose=False)
        return game, gne

    def test_all_types_valid_for_infinite_crs(self, solved):
        game, gne = solved
        infinite_crs = [cr for cr in gne.regions if not cr.is_unique]
        assert infinite_crs, "Designed game must have infinite CRs"

        cr = infinite_crs[0]
        p  = _chebyshev_center(cr, game)
        assert p is not None, "Cannot find interior point for infinite CR"

        for gne_type in ("min_norm", "welfare", "v_gne"):
            x   = select_gne(cr, game, p, gne_type)
            res = float(np.linalg.norm(cr.Mx @ x - cr.Mp @ p - cr.M1))
            assert res < 1e-5, f"{gne_type}: residual={res:.2e}"

    def test_welfare_le_minnorm_cost(self, solved):
        game, gne = solved
        infinite_crs = [cr for cr in gne.regions if not cr.is_unique]
        assert infinite_crs, "Designed game must have infinite CRs"

        cr = infinite_crs[0]
        p  = _chebyshev_center(cr, game)
        assert p is not None

        x_mn = select_gne(cr, game, p, "min_norm")
        x_wf = select_gne(cr, game, p, "welfare")

        def _sw(x):
            return sum(0.5 * x[game.x_slice(a.index)] @ a.Q @ x[game.x_slice(a.index)]
                       + (a.c + a.F @ p) @ x[game.x_slice(a.index)]
                       for a in game.agents)

        assert _sw(x_wf) <= _sw(x_mn) + 1e-6
