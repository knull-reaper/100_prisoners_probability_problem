import numpy as np
from numba import cuda, int32, boolean, NumbaPerformanceWarning
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numba.cuda.cudadrv.driver import CudaAPIError
import time, sys, gc, warnings, shutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from simple_term_menu import TerminalMenu
from tqdm import tqdm

LOCAL_MODE_LIMIT = 16 * 1024 // 4  # max N that fits in CUDA local memory
THREADS_PER_BLOCK = 256            # kernel launch size
RNG_STATE_BYTES = 16               # xoroshiro128p state footprint
BATCH_MEM_SAFETY_FACTOR = 0.55    # use 70% of free GPU memory for batching
BAR_FMT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
    "[elapsed: {elapsed}, eta: {remaining}, speed: {rate_fmt}]"
)

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

def center_line(txt: str, width: int) -> str:
    pad = max((width - len(txt)) // 2, 0)
    return " " * pad + txt

def mem_per_trial_bytes(n: int, mode: str) -> int:
    """Return per‑trial GPU bytes needed for *n* prisoners and kernel *mode*."""
    base = 1 + RNG_STATE_BYTES  # result bool + RNG state
    if mode == "Global":       # global kernel needs extra arrays
        base += (n * 4) + (n * 1)  # perm int32 + visited bool
    return base

def humanise(bytes_: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if bytes_ < 1024:
            return f"{bytes_:.2f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.2f} PB"


def create_prisoners_kernel_local(N):
    @cuda.jit
    def kernel(results, rng_states):
        idx = cuda.grid(1)
        if idx >= results.size:
            return
        perm = cuda.local.array(N, int32)
        visited = cuda.local.array(N, boolean)
        # initialise permutation 1…N
        for i in range(N):
            perm[i] = i + 1
        # Fisher–Yates shuffle
        for i in range(N - 1, 0, -1):
            j = int(xoroshiro128p_uniform_float32(rng_states, idx) * (i + 1))
            perm[i], perm[j] = perm[j], perm[i]
        # cycle‑length test
        for i in range(N):
            visited[i] = False
        half = N // 2
        for start in range(N):
            if not visited[start]:
                length, j = 0, start
                while not visited[j]:
                    visited[j] = True
                    j = perm[j] - 1
                    length += 1
                    if length > half:
                        results[idx] = False
                        return
        results[idx] = True
    return kernel

@cuda.jit
def prisoners_kernel_global(results, rng_states, perms, visited_global, N):
    idx = cuda.grid(1)
    if idx >= results.size:
        return
    perm = perms[idx]
    visited = visited_global[idx]
    for i in range(N):
        perm[i] = i + 1
    for i in range(N - 1, 0, -1):
        j = int(xoroshiro128p_uniform_float32(rng_states, idx) * (i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    for i in range(N):
        visited[i] = False
    half = N // 2
    for start in range(N):
        if not visited[start]:
            length, j = 0, start
            while not visited[j]:
                visited[j] = True
                j = perm[j] - 1
                length += 1
                if length > half:
                    results[idx] = False
                    return
    results[idx] = True


def _show_stats(rate: float, elapsed: float, console: Console):
    tbl = Table(show_header=True, header_style="bold cyan")
    tbl.add_column("Metric", justify="center")
    tbl.add_column("Value", justify="center")
    tbl.add_row("Success Rate", f"[bold]{rate:.4%}[/]")
    tbl.add_row("Elapsed", f"{elapsed:.3f} s")
    console.print(Align.center(tbl))


def _run_local(N, TRIALS, batch_size):
    successes, processed = 0, 0
    kernel = create_prisoners_kernel_local(N)
    with tqdm(total=TRIALS, unit="trial", colour="cyan", desc="Batch mode" if batch_size else "Running", bar_format=BAR_FMT, dynamic_ncols=True) as bar:
        while processed < TRIALS:
            chunk = batch_size or (TRIALS - processed)
            chunk = min(chunk, TRIALS - processed)
            res_d = cuda.device_array(chunk, dtype=np.bool_)
            rng = create_xoroshiro128p_states(chunk, seed=int(time.time()) + processed)
            blocks = (chunk + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            kernel[blocks, THREADS_PER_BLOCK](res_d, rng)
            cuda.synchronize()
            successes += res_d.copy_to_host().sum()
            processed += chunk
            bar.update(chunk)
    return successes / TRIALS


def _run_global(N, TRIALS, batch_size):
    successes, processed = 0, 0
    with tqdm(total=TRIALS, unit="trial", colour="cyan", desc="Batch mode" if batch_size else "Running", bar_format=BAR_FMT, dynamic_ncols=True) as bar:
        while processed < TRIALS:
            chunk = batch_size or (TRIALS - processed)
            chunk = min(chunk, TRIALS - processed)
            res_d = cuda.device_array(chunk, dtype=np.bool_)
            rng = create_xoroshiro128p_states(chunk, seed=int(time.time()) + processed)
            perms_d = cuda.device_array((chunk, N), dtype=np.int32)
            visited_d = cuda.device_array((chunk, N), dtype=np.bool_)
            blocks = (chunk + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            prisoners_kernel_global[blocks, THREADS_PER_BLOCK](res_d, rng, perms_d, visited_d, N)
            cuda.synchronize()
            successes += res_d.copy_to_host().sum()
            processed += chunk
            bar.update(chunk)
    return successes / TRIALS


def main():
    console = Console()
    console.clear()
    console.print(Align.center(Panel("[bold magenta]100 Prisoners Problem[/bold magenta]", border_style="green")))

    # Detect GPU ----------------------------------------------------------------
    try:
        device = cuda.get_current_device()
        free_mem, total_mem = cuda.current_context().get_memory_info()
    except CudaAPIError:
        console.print(Align.center("[bold red]CUDA error:[/] unable to initialise GPU."))
        sys.exit(1)
    console.print(Align.center(f"Found GPU: [bold green]{device.name.decode()}[/] ({total_mem/1024**2:.1f} MB total)"))

    term_width = shutil.get_terminal_size((80, 24)).columns

    prisoner_base = [100, 1_000, 10_000, 100_000, 1_000_000]
    prisoner_opts = [center_line(f"{n:,} [{'Local' if n<=LOCAL_MODE_LIMIT else 'Global'}] ~{humanise(mem_per_trial_bytes(n, 'Local' if n<=LOCAL_MODE_LIMIT else 'Global'))}/trial", term_width) for n in prisoner_base]
    p_idx = TerminalMenu(prisoner_opts, title="Select # of Prisoners", menu_cursor_style=("fg_cyan","bold"), menu_highlight_style=("bg_cyan","fg_black"), clear_screen=True).show()
    if p_idx is None:
        sys.exit(0)
    N = prisoner_base[p_idx]
    mode = "Local" if N <= LOCAL_MODE_LIMIT else "Global"

    trials_base = [100_000, 1_000_000, 10_000_000]
    per_trial = mem_per_trial_bytes(N, mode)
    safe_mem = free_mem * BATCH_MEM_SAFETY_FACTOR
    trial_opts = []
    for t in trials_base:
        tag = "[yellow](batched)" if per_trial * t > safe_mem else ""
        trial_opts.append(center_line(f"{t:,} (~{humanise(per_trial*t)}) {tag}", term_width))
    t_idx = TerminalMenu(trial_opts, title="Select # of Trials", menu_cursor_style=("fg_cyan","bold"), menu_highlight_style=("bg_cyan","fg_black"), clear_screen=False).show()
    if t_idx is None:
        sys.exit(0)
    TRIALS = trials_base[t_idx]

    # decide batching
    batch_mode = per_trial * TRIALS > safe_mem
    batch_size = int(safe_mem // per_trial) if batch_mode else 0
    if batch_mode:
        console.print(Align.center("[yellow]Switching to batch mode to fit GPU memory.[/]"))

    console.print(Align.center(f"Running {TRIALS:,} trials with {N:,} prisoners ({mode} kernel)…"))

    # run simulation
    start = time.time()
    if mode == "Local":
        success_rate = _run_local(N, TRIALS, batch_size)
    else:
        success_rate = _run_global(N, TRIALS, batch_size)
    elapsed = time.time() - start

    _show_stats(success_rate, elapsed, console)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        gc.collect()
        cuda.close()
