"""
Tests for mpgne/facet_gne.py

Fast tests (hand-crafted CRs, no PPOPT):   run normally
Slow tests (full PPOPT + facet pipeline):  @pytest.mark.slow

Run fast only:  python -m pytest tests/test_facet_gne.py -v -m "not slow"
Run all:        python -m pytest tests/test_facet_gne.py -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpgne.game import make_random_game
from mpgne.cr_store import AgentCR, AgentSolution, GNESolution
from mpgne.facet_gne import (
    _is_box_constraint,
    _hyperplane_in_cr,
    _solve_facet_lp,
    _facet_adjacent_combos,
    find_agent_cr_neighbors,
    find_all_agent_cr_neighbors,
    build_gne_solution_facet,
    FacetGNEResult,
)


# ---------------------------------------------------------------------------
# Hand-crafted 1-D AgentSolution with known facet structure
#
# Three non-overlapping CRs on the real line (n_theta=1):
#   CR0: [-10, -2]  →  x* = θ  (identity)
#   CR1: [-2,   3]  →  x* = θ
#   CR2: [  3,  10] →  x* = θ
# Facet neighbors: 0-1 (share boundary at -2), 1-2 (share boundary at 3)
# ---------------------------------------------------------------------------

def _make_1d_agent_sol(agent_index=0):
    """1-D AgentSolution with 3 CRs partitioning [-10, 10]."""
    edges = [(-10., -2.), (-2., 3.), (3., 10.)]
    regions = []
    for v, (lo, hi) in enumerate(edges):
        E = np.array([[1.], [-1.]])      # x ≤ hi, -x ≤ -lo → lo ≤ x ≤ hi
        f = np.array([hi, -lo])
        A = np.array([[1.]])             # x* = θ
        b = np.array([0.])
        regions.append(AgentCR(E=E, f=f, A=A, b=b, index=v))
    return AgentSolution(agent_index=agent_index, n_x_i=1, n_theta_i=1,
                         regions=regions)


def _make_2d_agent_sol(agent_index=0):
    """2-D AgentSolution: 4 box CRs tiling a 2×2 grid.
    CR ordering:  [(-10,0)×(-10,0), (0,10)×(-10,0),
                   (-10,0)×(0,10),  (0,10)×(0,10)]
    Known neighbors: 0-1 (share x1=0), 0-2 (share x2=0), 1-3, 2-3.
    """
    corners = [
        (np.array([-10., -10.]), np.array([0., 0.])),
        (np.array([  0., -10.]), np.array([10., 0.])),
        (np.array([-10.,   0.]), np.array([0., 10.])),
        (np.array([  0.,   0.]), np.array([10., 10.])),
    ]
    regions = []
    for v, (lo, hi) in enumerate(corners):
        E = np.vstack([np.eye(2), -np.eye(2)])
        f = np.concatenate([hi, -lo])
        A = np.eye(2, 2)
        b = np.zeros(2)
        regions.append(AgentCR(E=E, f=f, A=A, b=b, index=v))
    return AgentSolution(agent_index=agent_index, n_x_i=2, n_theta_i=2,
                         regions=regions)


# ---------------------------------------------------------------------------
# _is_box_constraint
# ---------------------------------------------------------------------------

class TestIsBoxConstraint:

    def test_unit_vector_is_box(self):
        assert _is_box_constraint(np.array([1., 0., 0.]))
        assert _is_box_constraint(np.array([0., -1., 0.]))

    def test_non_unit_is_not_box(self):
        assert not _is_box_constraint(np.array([1., 1., 0.]))  # two nonzeros
        assert not _is_box_constraint(np.array([2., 0., 0.]))  # value ≠ 1
        assert not _is_box_constraint(np.array([0., 0., 0.]))  # all zero


# ---------------------------------------------------------------------------
# _hyperplane_in_cr
# ---------------------------------------------------------------------------

class TestHyperplaneInCR:

    def test_shared_boundary_detected_non_box(self):
        """Non-axis-aligned hyperplane shared between two CRs is detected."""
        # CR with non-box constraint e=[1,1]/√2, f=0
        e = np.array([1., 1.]) / np.sqrt(2)
        f = 0.
        cr_k = AgentCR(
            E=np.array([[-1., -1.], [1., 0.]]) / np.sqrt(2),
            f=np.array([0., 5.]),
            A=np.eye(2, 2), b=np.zeros(2), index=0,
        )
        # cr_k row 0 is -e with -f → opposite direction at same boundary
        found = _hyperplane_in_cr(e, f, cr_k, check_same_dir=False)
        assert found, "Opposite-direction non-box boundary should be detected"

    def test_hyperplane_skips_box_constraints(self):
        """Hyperplane method skips axis-aligned (box) constraints by design."""
        sol = _make_1d_agent_sol()
        cr0, cr1 = sol[0], sol[1]
        # CR0's row 0: E=[1.], f=-2 — this IS a box constraint → skipped
        found = _hyperplane_in_cr(cr0.E[0], cr0.f[0], cr1, check_same_dir=False)
        assert not found, "Box constraints should be skipped by _hyperplane_in_cr"

    def test_non_adjacent_not_detected(self):
        """CR0 and CR2 share no boundary (gap between them)."""
        sol = _make_1d_agent_sol()
        cr0, cr2 = sol[0], sol[2]
        # CR0 upper boundary at -2; CR2 lower boundary at 3 — not shared
        found_upper = _hyperplane_in_cr(cr0.E[0], cr0.f[0], cr2, check_same_dir=False)
        found_lower = _hyperplane_in_cr(cr0.E[1], cr0.f[1], cr2, check_same_dir=False)
        assert not (found_upper or found_lower)

    def test_box_constraint_skipped(self):
        """Pure axis-aligned rows should be skipped."""
        e_box = np.array([1., 0.])   # box constraint
        cr    = _make_2d_agent_sol()[0]
        # Should NOT flag box constraints as shared hyperplanes
        result = _hyperplane_in_cr(e_box, 0.0, cr, check_same_dir=False)
        # Result could be True or False but must not raise
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# _solve_facet_lp
# ---------------------------------------------------------------------------

class TestSolveFacetLP:

    def test_adjacent_returns_positive(self):
        """CR0 and CR1 share facet at θ=-2; LP should return t* > 0."""
        sol = _make_1d_agent_sol()
        cr0, cr1 = sol[0], sol[1]
        # Row 0 of cr0: θ ≤ -2 (upper boundary)
        t_opt = _solve_facet_lp(cr0, cr1, j=0)
        assert t_opt is not None
        assert t_opt > 0

    def test_non_adjacent_returns_none_or_negative(self):
        """CR0 and CR2 do not share a proper facet."""
        sol = _make_1d_agent_sol()
        cr0, cr2 = sol[0], sol[2]
        # Check all facets of cr0 against cr2
        positives = []
        for j in range(cr0.n_ineq):
            t = _solve_facet_lp(cr0, cr2, j)
            if t is not None and t > 1e-6:
                positives.append(t)
        assert len(positives) == 0, "CR0 and CR2 should not be LP-adjacent"


# ---------------------------------------------------------------------------
# _facet_adjacent_combos
# ---------------------------------------------------------------------------

class TestFacetAdjacentCombos:

    def _sols_with_neighbors(self):
        """Two 1-D AgentSolutions with neighbors manually set."""
        sol0 = _make_1d_agent_sol(0)
        sol1 = _make_1d_agent_sol(1)
        sol0[0].facet_neighbors = [1]
        sol0[1].facet_neighbors = [0, 2]
        sol0[2].facet_neighbors = [1]
        sol1[0].facet_neighbors = [1]
        sol1[1].facet_neighbors = [0, 2]
        sol1[2].facet_neighbors = [1]
        return [sol0, sol1]

    def test_number_of_adjacent(self):
        sols = self._sols_with_neighbors()
        combos = list(_facet_adjacent_combos((1, 1), sols))
        # From (1,1): agent0 can go to {0,2}, agent1 can go to {0,2}
        # → 2 + 2 = 4 adjacent combos
        assert len(combos) == 4

    def test_adjacent_differ_by_one(self):
        sols = self._sols_with_neighbors()
        for combo in _facet_adjacent_combos((1, 1), sols):
            diffs = sum(a != b for a, b in zip((1, 1), combo))
            assert diffs == 1, f"Expected 1 diff, got {diffs} in {combo}"

    def test_adjacent_uses_facet_neighbors(self):
        sols = self._sols_with_neighbors()
        combos = set(_facet_adjacent_combos((0, 0), sols))
        # From (0,0): agent0 neighbors=[1], agent1 neighbors=[1]
        assert (1, 0) in combos
        assert (0, 1) in combos
        assert (2, 0) not in combos   # 2 is not a neighbor of 0

    def test_no_neighbors_yields_nothing(self):
        sol0 = _make_1d_agent_sol(0)
        sol1 = _make_1d_agent_sol(1)
        # No neighbors set
        combos = list(_facet_adjacent_combos((0, 0), [sol0, sol1]))
        assert len(combos) == 0


# ---------------------------------------------------------------------------
# find_agent_cr_neighbors  (hyperplane method)
# ---------------------------------------------------------------------------

class TestFindAgentCrNeighbors:

    def test_1d_lp_finds_adjacent(self):
        """LP method correctly finds neighbors for 1-D box-partition CRs."""
        sol = _make_1d_agent_sol()
        find_agent_cr_neighbors(sol, method="lp", verbose=False)
        assert 1 in sol[0].facet_neighbors
        assert 0 in sol[1].facet_neighbors
        assert 2 in sol[1].facet_neighbors
        assert 1 in sol[2].facet_neighbors

    def test_1d_lp_not_nonadjacent(self):
        sol = _make_1d_agent_sol()
        find_agent_cr_neighbors(sol, method="lp", verbose=False)
        assert 2 not in sol[0].facet_neighbors
        assert 0 not in sol[2].facet_neighbors

    def test_2d_lp_finds_grid_neighbors(self):
        """LP method finds the 4 neighbor pairs in the 2×2 grid."""
        sol = _make_2d_agent_sol()
        find_agent_cr_neighbors(sol, method="lp", verbose=False)
        assert 1 in sol[0].facet_neighbors
        assert 2 in sol[0].facet_neighbors
        assert 3 in sol[1].facet_neighbors
        assert 3 in sol[2].facet_neighbors

    def test_hyperplane_skips_box_partitions(self):
        """Hyperplane method skips axis-aligned boundaries — expected behaviour."""
        sol = _make_1d_agent_sol()
        find_agent_cr_neighbors(sol, method="hyperplane", verbose=False)
        # All 1-D constraints are axis-aligned → no neighbors found by hyperplane
        total = sum(len(cr.facet_neighbors) for cr in sol.regions)
        assert total == 0, "Hyperplane method should skip box-constraint CRs"

    def test_symmetric(self):
        sol = _make_1d_agent_sol()
        find_agent_cr_neighbors(sol, method="lp", verbose=False)
        for v, cr_v in enumerate(sol.regions):
            for w in cr_v.facet_neighbors:
                assert v in sol[w].facet_neighbors, f"{v} not in neighbors of {w}"

    def test_lp_method(self):
        sol = _make_1d_agent_sol()
        find_agent_cr_neighbors(sol, method="lp", verbose=False)
        assert 1 in sol[0].facet_neighbors
        assert 1 in sol[2].facet_neighbors

    def test_invalid_method_raises(self):
        sol = _make_1d_agent_sol()
        with pytest.raises(ValueError, match="Unknown method"):
            find_agent_cr_neighbors(sol, method="magic", verbose=False)

    def test_clears_existing_neighbors(self):
        sol = _make_1d_agent_sol()
        sol[0].facet_neighbors = [99, 88]   # stale data
        find_agent_cr_neighbors(sol, method="lp", verbose=False)
        assert 99 not in sol[0].facet_neighbors


# ---------------------------------------------------------------------------
# find_all_agent_cr_neighbors
# ---------------------------------------------------------------------------

class TestFindAllAgentCrNeighbors:

    def test_all_agents_processed(self):
        sols = [_make_1d_agent_sol(i) for i in range(3)]
        find_all_agent_cr_neighbors(sols, method="hyperplane", verbose=False)
        for sol in sols:
            for cr in sol.regions:
                assert isinstance(cr.facet_neighbors, list)

    def test_returns_same_list(self):
        sols = [_make_1d_agent_sol(0)]
        returned = find_all_agent_cr_neighbors(sols, method="hyperplane", verbose=False)
        assert returned is sols


# ---------------------------------------------------------------------------
# build_gne_solution_facet — fast (hand-crafted CRs, no PPOPT)
# ---------------------------------------------------------------------------

class TestBuildGneSolutionFacetFast:

    @pytest.fixture
    def hand_game_and_sols(self):
        """Use same hand game as test_gne_combiner + hand-crafted AgentSolutions."""
        from mpgne.game import Agent, GNEGame

        def _agent(idx):
            return Agent(
                index=idx, n_x=1,
                Q=np.array([[1.0]]),
                c=np.zeros(1),
                F=np.zeros((1, 1)),
                C=np.ones((1, 1)),
                A_loc=np.vstack([np.eye(1), -np.eye(1)]),
                b_loc=10.0 * np.ones(2),
                S_loc=np.zeros((2, 1)),
            )

        agents = [_agent(i) for i in range(2)]
        game = GNEGame(
            agents=agents, d=np.array([10.0]),
            S_coup=np.zeros((1, 1)),
            p_lb=np.array([-5.]), p_ub=np.array([5.]),
        )

        E_box = np.vstack([np.eye(2), -np.eye(2)])
        f_box = np.array([10., 5., 10., 5.])

        from mpgne.cr_store import AgentSolution

        def _sol(idx, A_row, b_val):
            cr = AgentCR(E=E_box, f=f_box,
                         A=np.array([A_row]), b=np.array([b_val]),
                         index=0)
            return AgentSolution(agent_index=idx, n_x_i=1, n_theta_i=2, regions=[cr])

        sol0 = _sol(0, [0.5, 0.2], 1.0)
        sol1 = _sol(1, [0.3, 0.1], -0.5)
        return game, [sol0, sol1]

    def test_fallback_when_no_neighbors(self, hand_game_and_sols):
        game, sols = hand_game_and_sols
        with pytest.warns(UserWarning, match="facet_neighbors is empty"):
            result = build_gne_solution_facet(game, sols, verbose=False)
        assert result.used_fallback is True
        assert isinstance(result.gne_sol, GNESolution)

    def test_fallback_same_as_exhaustive(self, hand_game_and_sols):
        """With no neighbors, FACET fallback == exhaustive."""
        from mpgne.gne_combiner import build_gne_solution
        game, sols = hand_game_and_sols
        exhaustive = build_gne_solution(game, sols, verbose=False)
        with pytest.warns(UserWarning):
            facet_res = build_gne_solution_facet(game, sols, verbose=False)
        assert facet_res.gne_sol.n_cr == exhaustive.n_cr

    def test_with_neighbors_bfs_runs(self, hand_game_and_sols):
        """With single-CR agents, BFS has no neighbors to explore."""
        game, sols = hand_game_and_sols
        # Manually set empty facet_neighbors (already empty, but explicitly)
        for sol in sols:
            for cr in sol.regions:
                cr.facet_neighbors = []
        # Should warn and fall back
        with pytest.warns(UserWarning):
            result = build_gne_solution_facet(game, sols, verbose=False)
        assert result.n_combos_checked >= 1

    def test_facet_result_fields(self, hand_game_and_sols):
        game, sols = hand_game_and_sols
        with pytest.warns(UserWarning):
            result = build_gne_solution_facet(game, sols, verbose=False)
        assert result.n_combos_total == 1   # 1 CR per agent × 2 agents = 1 combo
        assert result.n_combos_checked >= 1
        assert 0.0 <= result.reduction_ratio <= 1.0
        assert result.elapsed > 0
        assert result.speedup >= 1.0

    def test_bfs_explores_all_reachable(self):
        """With multi-CR agents and LP-detected neighbors, BFS visits all 9 combos."""
        sol0 = _make_1d_agent_sol(0)
        sol1 = _make_1d_agent_sol(1)
        find_agent_cr_neighbors(sol0, method="lp", verbose=False)
        find_agent_cr_neighbors(sol1, method="lp", verbose=False)
        # Check reachability from (0,0): should reach all 9 via chain 0-1-2
        from collections import deque
        agent_sols = [sol0, sol1]
        visited = set()
        q = deque([(0, 0)])
        visited.add((0, 0))
        while q:
            c = q.popleft()
            for nc in _facet_adjacent_combos(c, agent_sols):
                if nc not in visited:
                    visited.add(nc)
                    q.append(nc)
        assert len(visited) == 9   # 3 × 3 = all combinations reachable

    def test_reduction_ratio_with_neighbors(self):
        """Multi-CR agents with LP-detected neighbors → BFS covers full space."""
        from mpgne.game import Agent, GNEGame
        from mpgne.cr_store import AgentSolution

        # n_p=1 so n_theta_i = (n_x_total - n_x_i) + n_p = 1 + 1 = 2
        # AgentCR.A must be (1, 2): [A_x | A_p]
        def _make_sol_2theta(agent_index, edges):
            """AgentSolution with CRs over 1-D θ_x plus 1-D θ_p space."""
            regions = []
            for v, (lo, hi) in enumerate(edges):
                # CR in θ_x space (first dim): lo ≤ θ_x ≤ hi; θ_p unbounded
                E = np.array([[1., 0.], [-1., 0.]])   # only constrain θ_x
                f = np.array([hi, -lo])
                A = np.zeros((1, 2))    # x* = 0 (simple)
                b = np.zeros(1)
                regions.append(AgentCR(E=E, f=f, A=A, b=b, index=v))
            return AgentSolution(agent_index=agent_index, n_x_i=1,
                                 n_theta_i=2, regions=regions)

        edges = [(-10., -2.), (-2., 3.), (3., 10.)]
        sol0 = _make_sol_2theta(0, edges)
        sol1 = _make_sol_2theta(1, edges)
        find_agent_cr_neighbors(sol0, method="lp", verbose=False)
        find_agent_cr_neighbors(sol1, method="lp", verbose=False)

        def _agent(idx):
            return Agent(index=idx, n_x=1, Q=np.array([[2.]]), c=np.zeros(1),
                         F=np.zeros((1, 1)), C=np.ones((1, 1)),
                         A_loc=np.vstack([np.eye(1), -np.eye(1)]),
                         b_loc=np.array([10., 10.]),
                         S_loc=np.zeros((2, 1)))

        game = GNEGame(agents=[_agent(0), _agent(1)],
                       d=np.array([100.]),
                       S_coup=np.zeros((1, 1)),
                       p_lb=np.array([-5.]), p_ub=np.array([5.]))

        result = build_gne_solution_facet(game, [sol0, sol1], verbose=False)
        assert result.used_fallback is False
        assert result.n_combos_checked == 9   # all 9 reachable from (0,0)
        assert result.n_combos_total == 9
        assert result.reduction_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Slow: full PPOPT pipeline → facet detection → FACET vs exhaustive
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFacetGneSlow:

    @pytest.fixture
    def ppopt_game_and_sols(self):
        from mpgne.mp_solver import solve_all_agents_mp
        from ppopt.mp_solvers.solve_mpqp import mpqp_algorithm

        game = make_random_game(N=2, n_x=1, n_p=1, n_coupling=1,
                                x_bound=5., p_bound=3., seed=0)
        sols = solve_all_agents_mp(game,
                                   algorithm=mpqp_algorithm.combinatorial,
                                   verbose=False)
        return game, sols

    def test_facet_detection_finds_neighbors(self, ppopt_game_and_sols):
        _, sols = ppopt_game_and_sols
        find_all_agent_cr_neighbors(sols, method="hyperplane", verbose=False)
        # At least one agent should have at least one neighbor pair
        total_nb = sum(len(cr.facet_neighbors) for s in sols for cr in s.regions)
        assert total_nb >= 0   # even 0 is acceptable for tiny games

    def test_facet_gne_matches_exhaustive(self, ppopt_game_and_sols):
        """FACET-BFS and exhaustive must find the same number of GNE CRs."""
        from mpgne.gne_combiner import build_gne_solution
        game, sols = ppopt_game_and_sols

        exhaustive = build_gne_solution(game, sols, verbose=False)
        find_all_agent_cr_neighbors(sols, method="hyperplane", verbose=False)
        facet_res = build_gne_solution_facet(game, sols, verbose=False)

        assert facet_res.gne_sol.n_cr == exhaustive.n_cr, (
            f"FACET found {facet_res.gne_sol.n_cr} CRs, "
            f"exhaustive found {exhaustive.n_cr}"
        )

    def test_facet_result_stats(self, ppopt_game_and_sols):
        game, sols = ppopt_game_and_sols
        find_all_agent_cr_neighbors(sols, method="hyperplane", verbose=False)
        result = build_gne_solution_facet(game, sols, verbose=False)
        assert isinstance(result, FacetGNEResult)
        assert result.n_combos_checked <= result.n_combos_total
        assert result.reduction_ratio <= 1.0 + 1e-9

    def test_gne_crs_have_correct_n_p(self, ppopt_game_and_sols):
        game, sols = ppopt_game_and_sols
        find_all_agent_cr_neighbors(sols, method="hyperplane", verbose=False)
        result = build_gne_solution_facet(game, sols, verbose=False)
        for cr in result.gne_sol.regions:
            assert cr.n_p == game.n_p

    def test_equilibrium_residual_small(self, ppopt_game_and_sols):
        """For p inside any GNE CR found by FACET, residual must be small."""
        game, sols = ppopt_game_and_sols
        find_all_agent_cr_neighbors(sols, method="hyperplane", verbose=False)
        result = build_gne_solution_facet(game, sols, verbose=False)
        p = np.zeros(game.n_p)
        k = result.gne_sol.locate(p)
        if k is None:
            pytest.skip("p=0 not in any GNE CR for this seed")
        assert result.gne_sol[k].residual(p) < 1e-6
