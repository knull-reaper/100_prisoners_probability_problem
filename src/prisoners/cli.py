import typer
from enum import Enum
import json
from typing_extensions import Annotated
from pathlib import Path
from .simulate_cpu import simulate_cpu_numpy, simulate_cpu_numba
from .simulate_cuda import simulate_cuda

class Device(str, Enum):
    cpu = "cpu"

class Implementation(str, Enum):
    numpy = "numpy"
    numba = "numba"
    cuda = "cuda"

class Strategy(str, Enum):
    cycle = "cycle"
    random = "random"

app = typer.Typer()

@app.command()
def simulate(
    n: Annotated[int, typer.Option("--n", help="Number of prisoners and boxes.")] = 100,
    trials: Annotated[int, typer.Option("--trials", help="Number of simulations to run.")] = 1_000_000,
    device: Annotated[Device, typer.Option("--device", help="Device to run the simulation on.")] = Device.cpu,
    impl: Annotated[Implementation, typer.Option("--impl", help="Implementation to use.")] = Implementation.numba,
    alpha: Annotated[float, typer.Option("--alpha", help="Fraction of boxes to open.")] = 0.5,
    strategy: Annotated[Strategy, typer.Option("--strategy", help="Strategy to use.")] = Strategy.cycle,
    seed: Annotated[int, typer.Option("--seed", help="Random seed.")] = None,
    out: Annotated[Path, typer.Option("--out", help="Output file for results (JSON).", dir_okay=False, writable=True)] = None,
    ci: Annotated[float, typer.Option("--ci", help="Confidence interval level.")] = 0.95,
):
    """
    Run the 100 prisoners simulation.
    """
    typer.echo(f"Simulating with n={n}, trials={trials}, alpha={alpha}, strategy='{strategy.value}'")

    # Actual simulation logic
    if impl == Implementation.numpy:
        results = simulate_cpu_numpy(n=n, trials=trials, alpha=alpha, seed=seed, ci=ci)
    elif impl == Implementation.numba:
        results = simulate_cpu_numba(n=n, trials=trials, alpha=alpha, seed=seed, ci=ci)
    elif impl == Implementation.cuda:
        # A simple check to provide a better error message if cupy is not installed
        # The web dashboard will have a more robust check.
        try:
            from prisoners.simulate_cuda import simulate_cuda
            results = simulate_cuda(n=n, trials=trials, alpha=alpha, seed=seed, ci=ci)
        except ImportError:
            typer.echo("Could not import cupy. Is it installed correctly?", err=True)
            raise typer.Exit(1)
    else:
        # This case should not be reached due to the Enum
        typer.echo(f"Error: Unknown implementation '{impl.value}'", err=True)
        raise typer.Exit(1)

    typer.echo("Simulation complete.")

    if out:
        typer.echo(f"Saving results to {out}")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
    else:
        # Use rich print for better formatting if no output file is specified
        from rich import print
        print(results)


def main():
    app()

if __name__ == "__main__":
    main()
