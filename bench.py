import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

from prisoners.simulate_cpu import simulate_cpu_numpy, simulate_cpu_numba

def run_benchmarks():
    """
    Runs a suite of benchmarks across different parameters and simulation
    implementations.
    """
    # Define the parameter space for the benchmark
    # Using smaller trial counts for larger n to keep benchmark time reasonable.
    param_space = [
        {'n': 10, 'trials': 1_000_000, 'impls': ['numpy', 'numba']},
        {'n': 50, 'trials': 500_000, 'impls': ['numpy', 'numba']},
        {'n': 100, 'trials': 100_000, 'impls': ['numpy', 'numba']},
        {'n': 200, 'trials': 50_000, 'impls': ['numpy', 'numba']},
        {'n': 400, 'trials': 10_000, 'impls': ['numpy', 'numba']},
    ]

    results = []

    # Use tqdm for a progress bar
    for params in tqdm(param_space, desc="Running benchmarks"):
        n = params['n']
        trials = params['trials']
        for impl in params['impls']:
            print(f"\nRunning: n={n}, trials={trials}, impl={impl}")

            # Select the function to run
            if impl == 'numpy':
                sim_func = simulate_cpu_numpy
            elif impl == 'numba':
                sim_func = simulate_cpu_numba
            else:
                continue

            # Run simulation and store result
            # We use a fixed seed for consistency, though it doesn't affect performance
            result = sim_func(n=n, trials=trials, alpha=0.5, seed=42)
            results.append(result)

    # Convert results to a pandas DataFrame
    df = pd.DataFrame(results)

    # Save results to CSV
    output_csv = "benchmark_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nBenchmark results saved to {output_csv}")

    return df

def plot_results(df: pd.DataFrame):
    """
    Generates and saves a plot from the benchmark results DataFrame.
    """
    if df.empty:
        print("DataFrame is empty, skipping plot generation.")
        return

    # Set plot style
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 8))

    # Create a bar plot
    ax = sns.barplot(
        data=df,
        x='n',
        y='elapsed_ms',
        hue='impl',
        palette='viridis'
    )

    ax.set_title('Simulation Performance: Numba vs NumPy', fontsize=16)
    ax.set_xlabel('Number of Prisoners (n)', fontsize=12)
    ax.set_ylabel('Elapsed Time (ms, log scale)', fontsize=12)
    ax.set_yscale('log') # Use a log scale for better visibility of large differences

    plt.legend(title='Implementation')
    plt.tight_layout()

    # Save the plot
    output_png = "benchmark_performance.png"
    plt.savefig(output_png)
    print(f"Benchmark plot saved to {output_png}")


if __name__ == "__main__":
    print("Starting benchmark suite...")
    start_time = time.time()

    results_df = run_benchmarks()
    plot_results(results_df)

    end_time = time.time()
    print(f"Benchmark suite finished in {end_time - start_time:.2f} seconds.")
