import time
import cupy as cp

# It's good practice to import from the package's own modules
from prisoners.utils import wilson_ci

# Using a static array for `visited` in the kernel is simple, but has a size limit.
MAX_N_CUDA = 1024

# The CUDA kernel to find max cycle length for a batch of permutations.
# We define it as a template and then replace the placeholder for MAX_N_CUDA
# to avoid issues with f-string formatting of C-style braces.
cuda_kernel_template = """
extern "C" __global__
void find_successes_kernel(
    const int* all_perms,
    int* successes_out,
    int n,
    int trials_in_batch,
    int max_allowed_cycle)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= trials_in_batch) {
        return;
    }

    // Pointer to the start of the permutation for this thread
    const int* perm = all_perms + tid * n;

    // Use a local array for the visited flags.
    bool visited[__MAX_N__];
    for (int i = 0; i < n; ++i) {
        visited[i] = false;
    }

    bool trial_is_successful = true;
    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            int cycle_len = 0;
            int j = i;
            while (!visited[j]) {
                visited[j] = true;
                j = perm[j];
                cycle_len++;
            }
            if (cycle_len > max_allowed_cycle) {
                trial_is_successful = false;
                break; // Early abort for this trial
            }
        }
    }

    if (trial_is_successful) {
        successes_out[tid] = 1;
    } else {
        successes_out[tid] = 0;
    }
}
"""

# Substitute the placeholder with the actual value
cuda_kernel_code = cuda_kernel_template.replace('__MAX_N__', str(MAX_N_CUDA))

# Compile the CUDA kernel
find_successes_kernel = cp.RawKernel(cuda_kernel_code, 'find_successes_kernel')

def simulate_cuda(n: int, trials: int, alpha: float, seed: int | None, ci: float = 0.95, batch_size: int = 128 * 1024) -> dict:
    """
    Simulates the prisoner problem on the GPU using CuPy and a custom CUDA kernel.
    """
    impl = "cuda"
    start_time = time.perf_counter()

    if n > MAX_N_CUDA:
        raise ValueError(f"n must be less than or equal to {MAX_N_CUDA} for this CUDA implementation.")

    if seed is None:
        seed = cp.random.randint(0, 2**32 - 1)

    cp.random.seed(seed)

    max_allowed_cycle = int(n * alpha)
    total_successes = 0
    processed_trials = 0

    while processed_trials < trials:
        current_batch_size = min(batch_size, trials - processed_trials)

        # Generate permutations for the batch on the GPU using the argsort trick
        random_matrix = cp.random.rand(current_batch_size, n, dtype=cp.float32)
        perms_gpu = cp.argsort(random_matrix, axis=1).astype(cp.int32)

        # Allocate output array on the GPU for the results of this batch
        successes_gpu = cp.zeros(current_batch_size, dtype=cp.int32)

        # Launch the CUDA kernel
        block_size = 256
        grid_size = (current_batch_size + block_size - 1) // block_size

        find_successes_kernel(
            (grid_size,),
            (block_size,),
            (perms_gpu, successes_gpu, n, current_batch_size, max_allowed_cycle)
        )

        # Sum successes for the batch and add to total
        total_successes += int(cp.sum(successes_gpu).get())
        processed_trials += current_batch_size

    p_hat = total_successes / trials if trials > 0 else 0
    ci_low, ci_high = wilson_ci(total_successes, trials, confidence=ci)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "n": n,
        "trials": trials,
        "alpha": alpha,
        "successes": total_successes,
        "p_hat": p_hat,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "seed": seed,
        "impl": impl,
        "elapsed_ms": elapsed_ms,
    }
