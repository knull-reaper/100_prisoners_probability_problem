import pytest
from prisoners.simulate_cpu import simulate_cpu_numpy, simulate_cpu_numba
from prisoners.oracle import exact_probability

@pytest.mark.parametrize("n, trials, seed", [(10, 1000, 42), (20, 500, 1337)])
def test_cpu_implementations_are_deterministic(n, trials, seed):
    """
    Tests that for a given seed, numpy and numba implementations produce
    the exact same number of successes.
    """
    numpy_result = simulate_cpu_numpy(n=n, trials=trials, alpha=0.5, seed=seed)
    numba_result = simulate_cpu_numba(n=n, trials=trials, alpha=0.5, seed=seed)

    # The elapsed_ms will be different, so we don't compare the whole dict.
    # The most important thing is that they ran the same simulation.
    assert numpy_result["successes"] == numba_result["successes"]
    assert numpy_result["p_hat"] == pytest.approx(numba_result["p_hat"])


@pytest.mark.slow
def test_simulation_matches_oracle():
    """
    For a small n, run a simulation with many trials and check if the
    oracle's exact probability falls within the simulation's confidence interval.
    This is a good sanity check for the simulation logic.
    """
    n = 8
    # Use a high number of trials for statistical significance, but not so high
    # that the test takes forever.
    trials = 100000
    alpha = 0.5
    seed = 123

    # Get the exact probability from the oracle
    p_exact = exact_probability(n, alpha)

    # Run the simulation (numba is faster)
    sim_result = simulate_cpu_numba(n=n, trials=trials, alpha=alpha, seed=seed, ci=0.999)

    ci_low = sim_result["ci_low"]
    ci_high = sim_result["ci_high"]

    print(f"Sim p-hat: {sim_result['p_hat']:.6f}")
    print(f"Oracle p:    {p_exact:.6f}")
    print(f"99.9% CI:    ({ci_low:.6f}, {ci_high:.6f})")

    # Check that the exact probability is within the 99.9% confidence interval
    # of the simulation result.
    assert ci_low <= p_exact <= ci_high
