import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union


def plot_loglog_scaling(
    latency_l1: List[Union[int, float]],
    latency_l2: List[Union[int, float]],
    seq_lengths: List[int],
    output_filename: str = "latency_loglog_scaling.png",
    title: str = "Forward pass latency",
    legend1: str = "Flash Attention (Triton)",
    legend2: str = "Pytorch (Naive)",
    y_label: str = "Latency (ms)",
    x_label: str = "Sequence Length (N)",
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
    ax.plot(
        seq_lengths,
        latency_l1,
        marker="o",
        linestyle="-",
        label=legend1,
        color="#1f77b4",
    )

    # Plot L2 (Naive/Baseline)
    ax.plot(
        seq_lengths,
        latency_l2,
        marker="s",
        linestyle="--",
        label=legend2,
        color="#ff7f0e",
    )

    # --- Applying Logarithmic Scales (The Key Step) ---
    ax.set_xscale("log", base=2)  # Base 2 often cleans up sequence length plots
    ax.set_yscale("log")  # Logarithmic scale for latency

    # --- Formatting ---
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Custom X-tick seq_len_list for clarity on log scale
    ax.set_xticks(seq_lengths)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)  # Show log grid lines

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Log-log scaling plot saved as {output_filename}")
