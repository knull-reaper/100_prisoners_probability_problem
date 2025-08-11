import time
import numpy as np
from .utils import max_cycle_length, wilson_ci

def simulate_cpu_numpy(n: int, trials: int, alpha: float, seed: int | None, ci: float = 0.95) -> dict:
    """
    Simulates the prisoner problem using a pure NumPy implementation.
    """
    impl = "numpy"
    start_time = time.perf_counter()

    successes = 0
    max_allowed_cycle = int(n * alpha)

    # For consistency with Numba, use the legacy RandomState generator
    rng = np.random.RandomState(seed)

    for _ in range(trials):
        # Generate a random permutation by shuffling an array, to match Numba's method
        perm = np.arange(n)
        rng.shuffle(perm)

        # Find the length of the longest cycle
        longest_cycle = max_cycle_length(perm)

        # Check for success
        if longest_cycle <= max_allowed_cycle:
            successes += 1

    p_hat = successes / trials if trials > 0 else 0
    ci_low, ci_high = wilson_ci(successes, trials, confidence=ci)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "n": n,
        "trials": trials,
        "alpha": alpha,
        "successes": successes,
        "p_hat": p_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "seed": seed,
        "impl": impl,
        "elapsed_ms": elapsed_ms,
    }


from numba import njit

@njit
def _run_numba_simulation(n: int, trials: int, max_allowed_cycle: int, seed: int) -> int:
    """
    Core simulation loop, JIT-compiled with Numba.
    """
    # Seed the legacy numpy random state from within the JIT-compiled function
    np.random.seed(seed)

    successes = 0
    for _ in range(trials):
        # Numba works with np.random.shuffle
        perm = np.arange(n, dtype=np.int64)
        np.random.shuffle(perm)

        visited = np.zeros(n, dtype=np.bool_)
        trial_is_successful = True
        for i in range(n):
            if not visited[i]:
                cycle_len = 0
                j = i
                # Follow the cycle
                while not visited[j]:
                    visited[j] = True
                    j = perm[j]
                    cycle_len += 1

                # Early abort for the trial
                if cycle_len > max_allowed_cycle:
                    trial_is_successful = False
                    break  # No need to check other cycles

        if trial_is_successful:
            successes += 1

    return successes

def simulate_cpu_numba(n: int, trials: int, alpha: float, seed: int | None, ci: float = 0.95) -> dict:
    """
    Simulates the prisoner problem using a Numba-jitted implementation.
    """
    impl = "numba"
    start_time = time.perf_counter()

    # If seed is None, create one. This makes runs with no seed non-deterministic,
    # but ensures that when a seed is passed, it's used deterministically.
    if seed is None:
        # Use a high-entropy source to seed the PRNG for this run
        seed = np.random.randint(0, 2**32 - 1)

    max_allowed_cycle = int(n * alpha)

    # The first run of a Numba function has a compilation overhead.
    # For fair benchmarking, one might run it once before timing.
    successes = _run_numba_simulation(n, trials, max_allowed_cycle, seed)

    p_hat = successes / trials if trials > 0 else 0
    ci_low, ci_high = wilson_ci(successes, trials, confidence=ci)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "n": n,
        "trials": trials,
        "alpha": alpha,
        "successes": successes,
        "p_hat": p_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "seed": seed,
        "impl": impl,
        "elapsed_ms": elapsed_ms,
    }
