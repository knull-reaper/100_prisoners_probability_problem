import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

def _get_cycles(perm: np.ndarray) -> list[list[int]]:
    """Helper function to find all cycles in a permutation."""
    n = len(perm)
    visited = np.zeros(n, dtype=bool)
    cycles = []
    for i in range(n):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = perm[j]
            cycles.append(cycle)
    return cycles

def draw_permutation_graph(perm: np.ndarray, save_to: str | None = "permutation_graph.png"):
    """
    Draws the cycle graph of a permutation, highlights the longest cycle.

    Args:
        perm: The permutation as a numpy array.
        save_to: If not None, saves the plot to this filename.

    Returns:
        matplotlib.figure.Figure: The figure object for the plot.
    """
    n = len(perm)
    if n == 0:
        print("Cannot draw graph for empty permutation.")
        return None

    G = nx.DiGraph()
    for i in range(n):
        G.add_edge(i, perm[i])

    cycles = _get_cycles(perm)
    if len(cycles) == 0:
        longest_cycle = []
    else:
        longest_cycle = max(cycles, key=len)

    # Use a layout that shows cycles well
    pos = nx.circular_layout(G)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw all nodes and edges
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', node_size=700)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='->', arrowsize=20, edge_color='gray')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_color='black')

    # Highlight the longest cycle
    if longest_cycle:
        longest_cycle_edges = [(u, perm[u]) for u in longest_cycle]
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=longest_cycle, node_color='salmon', node_size=700)
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=longest_cycle_edges, edge_color='red', width=2.5, arrowsize=25)

    ax.set_title(f"Permutation Graph (n={n}), Longest Cycle Length: {len(longest_cycle)}", fontsize=16)
    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=150)
        print(f"Permutation graph saved to {save_to}")

    plt.close(fig) # Close the figure to free memory
    return fig

def plot_longest_cycle_histogram(cycle_lengths: list[int], save_to: str | None = "cycle_histogram.png"):
    """
    Creates and saves a histogram of longest cycle lengths from a simulation.
    """
    if len(cycle_lengths) == 0:
        print("Cannot plot histogram for empty list of cycle lengths.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.histplot(cycle_lengths, ax=ax, kde=True, bins=max(cycle_lengths))

    ax.set_title('Distribution of Longest Cycle Lengths', fontsize=16)
    ax.set_xlabel('Length of Longest Cycle', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)

    plt.tight_layout()

    if save_to:
        plt.savefig(save_to, dpi=150)
        print(f"Cycle length histogram saved to {save_to}")

    plt.close(fig)
    return fig


if __name__ == '__main__':
    # --- Test draw_permutation_graph ---
    print("Testing draw_permutation_graph...")
    # A permutation with a few cycles for n=15
    test_perm = np.array([1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 9, 12, 11, 14, 13])
    # Cycles are: (0 1 2), (3 4 5), (6 7 8), (9 10), (11 12), (13 14)
    # Longest cycles are of length 3.
    draw_permutation_graph(test_perm, save_to="test_permutation_graph.png")

    # --- Test plot_longest_cycle_histogram ---
    print("\nTesting plot_longest_cycle_histogram...")
    # Generate some sample data for the histogram
    # In a real scenario, this would come from many simulation trials
    rng = np.random.default_rng(42)
    sample_cycle_lengths = rng.integers(1, 50, size=10000)
    plot_longest_cycle_histogram(sample_cycle_lengths, save_to="test_cycle_histogram.png")
