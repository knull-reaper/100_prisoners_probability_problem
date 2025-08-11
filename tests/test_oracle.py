import pytest
import math
from prisoners.oracle import exact_probability

def test_exact_probability_small_n():
    """Test the exact probability for small n against manually calculated values."""
    # For n=3, m=floor(3*0.5)=1. Only the identity perm is valid. P = 1/6.
    assert exact_probability(3, 0.5) == pytest.approx(1/6)

    # For n=4, m=floor(4*0.5)=2. Successful perms are identity and those with 2-cycles.
    # Number of successful permutations is 10. Total permutations is 24. P = 10/24.
    assert exact_probability(4, 0.5) == pytest.approx(10/24)

@pytest.mark.slow
def test_exact_probability_classic_problem():
    """Test the exact probability for n=100 against the known value."""
    # This test can be slow due to the recursion depth for n=100.
    # The exact probability is 1 - Sum_{k=51 to 100} 1/k.
    # The value is known to be approx 0.3118278.
    p_100 = exact_probability(100, 0.5)
    assert p_100 == pytest.approx(0.3118278, abs=1e-6)
    # Also check that it's reasonably close to the famous approximation.
    assert p_100 == pytest.approx(1 - math.log(2), abs=1e-2)

def test_exact_probability_edge_cases():
    """Test edge cases for the oracle function."""
    # alpha=1.0 -> m=n, all permutations are successful.
    assert exact_probability(10, 1.0) == 1.0

    # alpha near zero -> m=0 (for n > 0), no permutations can be successful.
    assert exact_probability(10, 0.05) == 0.0 # m = floor(0.5) = 0

    # n=1, any valid alpha should be 1.0
    assert exact_probability(1, 0.5) == 1.0
