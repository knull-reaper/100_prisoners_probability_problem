import math
from functools import lru_cache

@lru_cache(maxsize=None)
def _count_successful_perms(n: int, m: int) -> int:
    """
    Recursively counts the number of permutations of n elements where all
    cycle lengths are less than or equal to m.

    This is based on the recurrence relation for permutations with restricted
    cycle lengths.
    f(n, m) = sum_{k=1 to min(n, m)} [ C(n-1, k-1) * (k-1)! * f(n-k, m) ]
    """
    if n == 0:
        return 1

    count = 0
    # Consider the cycle containing the element 'n'. Let its length be k.
    # We must have 1 <= k <= m.
    # We choose k-1 other elements from the n-1 available elements.
    # The number of ways to do this is C(n-1, k-1).
    # These k elements can form (k-1)! distinct cycles.
    # The remaining n-k elements must form a valid permutation among themselves.
    for k in range(1, min(n, m) + 1):
        # Using math.comb is better to avoid large intermediate numbers from factorials
        # that could lead to overflow.
        # C(n-1, k-1) * (k-1)!
        ways_to_form_cycle = math.comb(n - 1, k - 1) * math.factorial(k - 1)

        # Recursively find the number of valid permutations for the rest
        count += ways_to_form_cycle * _count_successful_perms(n - k, m)

    return count

def exact_probability(n: int, alpha: float) -> float:
    """
    Calculates the exact probability of success for the cycle strategy
    for a given n and alpha.

    The probability is the number of permutations where the longest cycle
    is less than or equal to floor(n * alpha), divided by n!.
    """
    if n <= 1:
        return 1.0

    m = math.floor(n * alpha)

    # If the max allowed cycle length is less than 1, no permutation is possible.
    # (This case is theoretical as n >= 2 and alpha > 0 usually)
    if m < 1:
        return 0.0

    # If the max allowed cycle length is n or more, all permutations are successful.
    if m >= n:
        return 1.0

    num_successful = _count_successful_perms(n, m)
    total_perms = math.factorial(n)

    if total_perms == 0: # Should not happen for n > 1
        return 0.0

    return num_successful / total_perms
