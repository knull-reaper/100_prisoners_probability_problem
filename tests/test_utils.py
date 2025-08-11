import numpy as np
import pytest
from prisoners.utils import max_cycle_length, wilson_ci

def test_max_cycle_length_identity():
    """Test with an identity permutation (n cycles of length 1)."""
    perm = np.arange(5)
    assert max_cycle_length(perm) == 1

def test_max_cycle_length_full_cycle():
    """Test with a single cycle of length 5."""
    perm = np.array([1, 2, 3, 4, 0])
    assert max_cycle_length(perm) == 5

def test_max_cycle_length_multiple_cycles():
    """Test with multiple cycles of different lengths."""
    # Two cycles: (0 1 2) and (3 4)
    perm = np.array([1, 2, 0, 4, 3])
    assert max_cycle_length(perm) == 3

def test_max_cycle_length_empty():
    """Test with an empty permutation."""
    perm = np.array([])
    assert max_cycle_length(perm) == 0

def test_wilson_ci_standard():
    """Test the Wilson CI with a standard example."""
    # Values calculated manually to verify the implementation
    successes, trials = 80, 100
    ci_low, ci_high = wilson_ci(successes, trials, confidence=0.95)
    # Using pytest.approx for float comparison
    assert ci_low == pytest.approx(0.711, abs=1e-3)
    assert ci_high == pytest.approx(0.867, abs=1e-3)

def test_wilson_ci_edge_cases():
    """Test Wilson CI with edge cases."""
    # 0 trials -> returns a non-informative (0, 1) interval
    assert wilson_ci(0, 0) == (0.0, 1.0)

    # 0 successes
    ci_low, ci_high = wilson_ci(0, 100, confidence=0.95)
    assert ci_low == pytest.approx(0.0)
    assert ci_high == pytest.approx(0.037, abs=1e-3)

    # all successes
    ci_low, ci_high = wilson_ci(100, 100, confidence=0.95)
    assert ci_low == pytest.approx(1.0 - 0.037, abs=1e-3)
    assert ci_high == pytest.approx(1.0)
