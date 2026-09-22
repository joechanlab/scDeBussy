"""Focused tests for the landmark_matchable propagation fix in compile_graph.

RU831's compression/gap pathology was traced (see
../joint_barycenter/ru831_diagnostic/RU831_DIAGNOSIS.md) to landmark_matchable=False
landmarks still contributing ordinary match edges via their raw landmark_expression.
These tests use small synthetic cost matrices (build_soft_graph is hardcoded to a
100x100 landmark grid, so the matrix size itself can't shrink) to verify the fix in
isolation, without needing the real HTAN dataset.
"""

import numpy as np
import pytest
from three_patient_joint import compile_graph, infer_graph

BETA_TAIL_PRICE = 0.517  # arbitrary fixed terminal price, matches ETA's rough magnitude


def ramp_cost():
    """A well-behaved synthetic cost matrix: cheapest along the diagonal, so an
    unconstrained fit would naturally want to match every landmark near-identically."""
    i = np.arange(100)[:, None]
    j = np.arange(100)[None, :]
    return ((i - j) / 20.0) ** 2


def compile_and_infer(cost, matchable=None):
    compiled, nodes = compile_graph(cost, BETA_TAIL_PRICE, matchable=matchable)
    return infer_graph(compiled, nodes, cost)


def test_fully_matchable_reproduces_default_behavior():
    """Passing matchable=None and an explicit all-True mask must be identical, and
    both must match the (pre-fix) unmasked compile_graph/infer_graph output."""
    cost = ramp_cost()
    compiled_none, nodes_none = compile_graph(cost, BETA_TAIL_PRICE)
    compiled_true, nodes_true = compile_graph(cost, BETA_TAIL_PRICE, matchable=np.ones(100, bool))
    assert nodes_none == nodes_true
    for a, b in zip(compiled_none, compiled_true, strict=True):
        np.testing.assert_array_equal(a, b)

    result_none = infer_graph(compiled_none, nodes_none, cost)
    result_true = infer_graph(compiled_true, nodes_true, cost)
    np.testing.assert_array_equal(result_none["matrix"], result_true["matrix"])
    np.testing.assert_array_equal(result_none["map_index"], result_true["map_index"])
    assert result_none["internal_patient_gaps"] == result_true["internal_patient_gaps"]


def test_unmatchable_landmarks_are_never_matched():
    """Landmarks flagged unmatchable must contribute zero posterior match mass and
    never appear as a matched (kind 0/1/2) MAP-path source, even though nothing
    else about the cost matrix changes."""
    cost = ramp_cost()
    matchable = np.ones(100, bool)
    matchable[40:50] = False

    result = compile_and_infer(cost, matchable=matchable)

    # No ordinary match edge sourced from an unmatchable landmark: map_index stays
    # unset (-1) for all of them, and their posterior row mass is exactly zero.
    assert np.all(result["map_index"][40:50] == -1)
    np.testing.assert_array_equal(result["matrix"][40:50, :].sum(axis=1), np.zeros(10))

    # The region must still be crossed -- via gap edges, not silently dropped.
    assert result["internal_patient_gaps"] > 0


def test_unsupported_region_handled_via_gap_not_by_relying_on_huge_cost():
    """The fix must work even when the unsupported landmarks look artificially
    *cheap* to match (simulating unreliable-but-not-obviously-bad imputed
    expression) -- proving avoidance comes from the matchable penalty, not from
    the underlying expression cost happening to be large. Contrast against the
    unfixed (matchable=None) run on the identical attractive-cost matrix, which
    must actually match through that region, to show the fix changes behavior in
    the intended direction rather than being a no-op."""
    cost = ramp_cost()
    cost[40:50, :] = 1e-6  # artificially attractive to every reference position
    matchable = np.ones(100, bool)
    matchable[40:50] = False

    before = compile_and_infer(cost, matchable=None)
    after = compile_and_infer(cost, matchable=matchable)

    # Without the fix, the artificially cheap region gets matched (not gapped).
    assert np.any(before["map_index"][40:50] >= 0)
    assert before["matrix"][40:50, :].sum() > 0.5

    # With the fix, the same artificially cheap region is fully avoided and
    # crossed via gap edges instead, despite the low raw expression cost.
    assert np.all(after["map_index"][40:50] == -1)
    np.testing.assert_array_equal(after["matrix"][40:50, :].sum(axis=1), np.zeros(10))
    assert after["internal_patient_gaps"] >= 10

    # Landmarks outside the unsupported block are essentially unaffected (a tiny
    # renormalization shift right at the boundary is expected and fine).
    np.testing.assert_allclose(
        after["matrix"][:35, :].sum(axis=1), before["matrix"][:35, :].sum(axis=1), atol=1e-3
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
