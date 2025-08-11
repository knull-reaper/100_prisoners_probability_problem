import streamlit as st
import numpy as np
import json
from io import BytesIO

# It's good practice to import functions from the installed package
# This requires the package to be installed (e.g., `pip install -e .`)
from prisoners.simulate_cpu import simulate_cpu_numba, simulate_cpu_numpy
from prisoners.simulate_cuda import simulate_cuda
from prisoners.utils import is_gpu_available
from prisoners.viz import draw_permutation_graph

# --- App Configuration ---
st.set_page_config(
    page_title="100 Prisoners Simulator",
    page_icon="🎲",
    layout="wide"
)

# --- Helper Functions ---
def get_session_state():
    """Initializes session state variables if they don't exist."""
    if 'sim_results' not in st.session_state:
        st.session_state.sim_results = None
    if 'permutation' not in st.session_state:
        st.session_state.permutation = None

# --- Sidebar (Inputs) ---
with st.sidebar:
    st.title(" prisoners.ai")
    st.header("Simulation Settings")

    # Check for GPU availability
    GPU_AVAILABLE = is_gpu_available()

    impl_options = ["numba", "numpy"]
    if GPU_AVAILABLE:
        impl_options.append("cuda")

    impl = st.selectbox(
        "Implementation",
        impl_options,
        help="Numba is fastest on CPU. CUDA uses the GPU."
    )

    if not GPU_AVAILABLE and impl == "cuda":
        st.warning("GPU not available. Please select a CPU implementation.")
    elif not GPU_AVAILABLE:
        st.info("GPU not available. Running in CPU-only mode.")

    n = st.slider("Number of Prisoners (n)", min_value=2, max_value=200, value=100, step=2)
    alpha = st.slider("Fraction of Boxes to Open (alpha)", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    trials = st.select_slider(
        "Number of Trials",
        options=[100, 1000, 10_000, 100_000, 1_000_000],
        value=100_000
    )
    seed = st.number_input("Random Seed", value=42, min_value=0)

    run_button = st.button("Run Simulation", type="primary")

# --- Main Page (Outputs) ---
st.title("100 Prisoners Problem Simulator")
st.write(
    "This app simulates the classic 100 prisoners problem. "
    "Set the parameters in the sidebar and click 'Run Simulation' to see the results."
)

get_session_state()

if run_button:
    with st.spinner("Running simulation... this may take a moment."):
        if impl == 'cuda':
            sim_func = simulate_cuda
        elif impl == 'numpy':
            sim_func = simulate_cpu_numpy
        else: # default to numba
            sim_func = simulate_cpu_numba

        st.session_state.sim_results = sim_func(n=n, trials=trials, alpha=alpha, seed=seed)
        # Generate a single permutation to visualize
        rng = np.random.default_rng(seed)
        st.session_state.permutation = rng.permutation(n)

if st.session_state.sim_results:
    results = st.session_state.sim_results

    st.header("Simulation Results")

    # Display key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Success Probability (p̂)", f"{results['p_hat']:.4f}")
    col2.metric("Successes", f"{results['successes']:,}")
    col3.metric("Trials", f"{results['trials']:,}")

    st.write(f"**95% Confidence Interval:** `({results['ci_low']:.4f}, {results['ci_high']:.4f})`")
    st.write(f"**Elapsed Time:** `{results['elapsed_ms']:.2f} ms`")

    # --- Visualizations and Downloads ---
    st.header("Visualizations & Data")

    # Permutation Graph
    with st.expander("Show Permutation Cycle Graph", expanded=True):
        fig = draw_permutation_graph(st.session_state.permutation)
        st.pyplot(fig)

        # Download button for the image
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format="png", dpi=300)
        st.download_button(
            label="Download Graph (PNG)",
            data=img_buffer,
            file_name=f"permutation_n{n}_seed{seed}.png",
            mime="image/png"
        )

    # Raw Results
    with st.expander("Show Raw JSON Results"):
        st.json(results)

        # Download button for the JSON data
        st.download_button(
            label="Download Results (JSON)",
            data=json.dumps(results, indent=2),
            file_name=f"simulation_results_n{n}_t{trials}.json",
            mime="application/json"
        )
else:
    st.info("Click 'Run Simulation' in the sidebar to begin.")
