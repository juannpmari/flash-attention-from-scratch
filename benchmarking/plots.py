import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union

def plot_latencies(
    triton_latency: List[Union[int, float]], 
    naive_latency: List[Union[int, float]], 
    seq_len_labels: List[str], # Renamed for clarity: these are labels
    output_filename: str = "scaling_line_comparison.pdf",
    legend1: str = "Triton Flash (L1)",
    legend2: str = "Naive Baseline (L2)",
    y_label: str = "Latency (ms)",
    x_label: str = "Sequence Length (N)",
    is_log_scale: bool = False # Set default to True as per scaling discussion
):
    """
    Creates a professional line plot comparing two latency metrics across sequence lengths 
    to visualize scaling rates.
    """
    
    # 1. Input Validation
    if not (len(triton_latency) == len(naive_latency) == len(seq_len_labels)):
        raise ValueError("All three lists must have the same length.")

    # Apply professional style context
    plt.style.use('seaborn-v0_8-paper') 
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.figsize'] = (7, 4.5) 

    N = len(seq_len_labels)
    # Use simple numerical indices for the X-axis plot position
    x_positions = np.arange(N)
    
    # --- Plotting ---
    fig, ax = plt.subplots()

    # Define professional colors and markers/linestyles
    COLOR_TRITON = '#0072B2' # Blue
    COLOR_NAIVE = '#D55E00'  # Orange-Red

    # Plot line for the first metric (solid line, circles)
    ax.plot(x_positions, triton_latency, 
            label=legend1, color=COLOR_TRITON, 
            linestyle='-', marker='o', markersize=5, linewidth=2, zorder=3)

    # Plot line for the second metric (dashed line, squares)
    ax.plot(x_positions, naive_latency, 
            label=legend2, color=COLOR_NAIVE, 
            linestyle='--', marker='s', markersize=5, linewidth=2, zorder=3)

    # --- Formatting ---
    
    ax.set_xlabel(x_label) 
    ax.set_ylabel(y_label)
    
    # Set X-axis ticks to the numerical positions, but use the sequence length strings as labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(seq_len_labels)
    
    # Professional Style: Add light grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0) 

    # Log Scale Application
    if is_log_scale:
        ax.set_yscale('log')
        # We need to manually convert X-axis labels to be interpreted as numeric if we want x-log scaling
        # For simplicity and clean labels, we keep X as categorical labels but often X is also log-scaled 
        # (See Example 2 for numerical X-axis plot)

    # Legend and Layout
    ax.legend(frameon=False, loc='upper left') 
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_filename, bbox_inches='tight')
    plt.close(fig) 
    
    print(f"Professional line plot saved as {output_filename}")

def plot_loglog_scaling(
    latency_l1: List[Union[int, float]], 
    latency_l2: List[Union[int, float]], 
    seq_lengths: List[int],
    output_filename: str = "latency_loglog_scaling.png",
    title: str = "Scaling Rate Comparison",
    legend1: str = "Metric 1 (Fast)",
    legend2: str = "Metric 2 (Naive)",
    y_label: str = "Latency (ms)",
    x_label: str = "Sequence Length (N)"
):
    """
    Creates a log-log plot to compare the scaling rate of two metrics vs. sequence length.

    Args:
        latency_l1 (list): Latencies for the optimized method (L1).
        latency_l2 (list): Latencies for the naive method (L2).
        seq_lengths (list): List of sequence lengths (X-axis data).
        output_filename (str): Name of the file to save the plot.
        ... (optional seq_len_list)
    """
    
    if not (len(latency_l1) == len(latency_l2) == len(seq_lengths)):
        raise ValueError("All three input lists must have the same length.")

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Plotting the Data ---
    
    # Plot L1 (Optimized/Triton)
    ax.plot(seq_lengths, latency_l1, marker='o', linestyle='-', label=legend1, color='#1f77b4')

    # Plot L2 (Naive/Baseline)
    ax.plot(seq_lengths, latency_l2, marker='s', linestyle='--', label=legend2, color='#ff7f0e')

    # --- Applying Logarithmic Scales (The Key Step) ---
    ax.set_xscale('log', base=2) # Base 2 often cleans up sequence length plots
    ax.set_yscale('log')         # Logarithmic scale for latency

    # --- Formatting ---
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Custom X-tick seq_len_list for clarity on log scale
    ax.set_xticks(seq_lengths)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5) # Show log grid lines
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Log-log scaling plot saved as {output_filename}")