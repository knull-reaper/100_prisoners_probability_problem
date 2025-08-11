import numpy as np
from scipy.stats import norm

def seed_rng(seed: int | None):
    """Seeds the random number generators for reproducibility."""
    np.random.seed(seed)

def max_cycle_length(perm: np.ndarray) -> int:
    """
    Calculates the length of the longest cycle in a permutation.

    The permutation is represented as a numpy array where perm[i] = j
    means there is a directed edge from i to j.
    """
    n = len(perm)
    visited = np.zeros(n, dtype=bool)
    max_len = 0
    for i in range(n):
        if not visited[i]:
            # Follow the cycle and count its length
            cycle_len = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                cycle_len += 1
            if cycle_len > max_len:
                max_len = cycle_len
    return max_len

def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Calculates the Wilson score interval for a binomial proportion.
    """
    if trials == 0:
        return (0.0, 1.0)

    # Handle cases where successes are 0 or trials, to avoid domain errors with sqrt
    if successes == 0:
        p_hat = 0
    elif successes == trials:
        p_hat = 1
    else:
        p_hat = successes / trials

    z = norm.ppf(1 - (1 - confidence) / 2)

    n = trials
    denominator = 1 + z**2 / n
    center = p_hat + z**2 / (2 * n)
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))

    ci_low = (center - margin) / denominator
    ci_high = (center + margin) / denominator

    # Clamp values to [0, 1] range
    return (max(0.0, ci_low), min(1.0, ci_high))

def is_gpu_available():
    """
    Safely checks if CuPy is installed and a GPU is available.
    Catches both ImportError and runtime errors from CuPy.
    """
    try:
        import cupy
        return cupy.is_available()
    except (ImportError, RuntimeError):
        return False

def get_longest_cycle(perm: np.ndarray) -> list[int]:
    """
    Finds the nodes belonging to the longest cycle in a permutation.
    """
    n = len(perm)
    if n == 0:
        return []

    visited = np.zeros(n, dtype=bool)
    longest_cycle_nodes = []

    for i in range(n):
        if not visited[i]:
            cycle_nodes = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle_nodes.append(j)
                j = perm[j]

            if len(cycle_nodes) > len(longest_cycle_nodes):
                longest_cycle_nodes = cycle_nodes

    return longest_cycle_nodes
