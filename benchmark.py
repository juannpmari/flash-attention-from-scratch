import triton
import torch
from benchmarking.naive_attention import naive_attention, naive_attention_backward
from benchmarking.plots import plot_loglog_scaling
from flash_attn_triton import FlashAttnTriton


def _create_powers_of_2_list(A: float, B: float) -> list:
    """Creates a list [A, A*2, A*4, ...] up to B."""

    result = []
    current_num = A

    while current_num <= B:
        result.append(current_num)
        current_num *= 2
    return result


def benchmark_latency_triton_vs_naive(
    dtype_list, embed_dim_list, seq_len_list, eval="forward"
):
    batch_size = 1
    for dtype in dtype_list:
        for embed_dim in embed_dim_list:
            triton_latency = []
            naive_latency = []
            for seq_len in seq_len_list:
                Q = torch.randn(
                    batch_size,
                    seq_len,
                    embed_dim,
                    dtype=dtype,
                    device="cuda",
                    requires_grad=True,
                )
                K = torch.randn(
                    batch_size,
                    seq_len,
                    embed_dim,
                    dtype=dtype,
                    device="cuda",
                    requires_grad=True,
                )
                V = torch.randn(
                    batch_size,
                    seq_len,
                    embed_dim,
                    dtype=dtype,
                    device="cuda",
                    requires_grad=True,
                )

                fraction = 1.0  # Be careful with 1.0; 0.95 is safer to leave room for context overhead
                torch.cuda.set_per_process_memory_fraction(fraction, 0)

                if eval == "forward":
                    fn = lambda: FlashAttnTriton.apply(Q, K, V, True)
                    triton_latency.append(
                        triton.testing.do_bench(
                            fn,
                            warmup=25,
                            rep=100,
                            grad_to_none=None,
                            quantiles=None,
                            return_mode="median",
                        )
                    )

                    fn = lambda: naive_attention(Q, K, V)
                    naive_latency.append(
                        triton.testing.do_bench(
                            fn,
                            warmup=25,
                            rep=100,
                            grad_to_none=None,
                            quantiles=None,
                            return_mode="median",
                        )
                    )
                elif eval == "both":  # Both forward and backward

                    def fn():
                        O, L = FlashAttnTriton.apply(Q, K, V, True)  # forward
                        dO = torch.randn_like(O)
                        dL = torch.randn_like(L)
                        torch.autograd.backward(
                            [O, L], [dO, dL], retain_graph=True
                        )  # backward

                    triton_latency.append(
                        triton.testing.do_bench(
                            fn,
                            warmup=25,
                            rep=100,
                            grad_to_none=None,
                            quantiles=None,
                            return_mode="median",
                        )
                    )

                    def fn():
                        O, P = naive_attention(Q, K, V)  # forward
                        dO = torch.randn_like(O)
                        naive_attention_backward(Q, K, V, P, dO, True)  # backward

                    naive_latency.append(
                        triton.testing.do_bench(
                            fn,
                            warmup=25,
                            rep=100,
                            grad_to_none=None,
                            quantiles=None,
                            return_mode="median",
                        )
                    )

            plot_loglog_scaling(
                triton_latency,
                naive_latency,
                seq_len_list,
                output_filename=f"latency_loglog_scaling_embed{embed_dim}_dtype{str(dtype).split('.')[-1]}.png",
                title="Latency - forward and backward pass",
            )


def benchmark_memory_footprint_triton_vs_naive(
    dtype_list, embed_dim_list, seq_len_list
):
    batch_size = 1
    for dtype in dtype_list:
        for embed_dim in embed_dim_list:
            triton_memory = []
            naive_memory = []
            for seq_len in seq_len_list:
                Q = torch.randn(
                    batch_size, seq_len, embed_dim, dtype=dtype, device="cuda"
                )
                K = torch.randn(
                    batch_size, seq_len, embed_dim, dtype=dtype, device="cuda"
                )
                V = torch.randn(
                    batch_size, seq_len, embed_dim, dtype=dtype, device="cuda"
                )

                fraction = 1.0  # Be careful with 1.0; 0.95 is safer to leave room for context overhead
                torch.cuda.set_per_process_memory_fraction(fraction, 0)
                # --- Triton Memory Benchmark ---
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                try:             
                    FlashAttnTriton.apply(Q, K, V, True)
                    torch.cuda.synchronize()
                    triton_peak_mb = torch.cuda.max_memory_reserved() / (1024**2)
                    triton_memory.append(triton_peak_mb)
                except torch.cuda.OutOfMemoryError:
                    print(f"Triton OOM for seq_len={seq_len}, embed_dim={embed_dim}, dtype={dtype}")
                    # triton_peak_mb = 0

                # --- Naive Memory Benchmark ---
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                try:
                    naive_attention(Q, K, V, True)
                    torch.cuda.synchronize()
                    # naive_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
                    naive_peak_mb = torch.cuda.max_memory_reserved() / (1024**2)
                    naive_memory.append(naive_peak_mb)
                except torch.cuda.OutOfMemoryError:
                    print(f"Naive OOM for seq_len={seq_len}, embed_dim={embed_dim}, dtype={dtype}")
                    # naive_peak_mb = 0
            plot_loglog_scaling(
                triton_memory,
                naive_memory,
                seq_len_list,
                output_filename=f"memory_loglog_scaling_embed{embed_dim}_dtype{str(dtype).split('.')[-1]}.png",
                y_label="Memory Footprint (MB)",
                title="Peak Memory Footprint (fp32) - forward pass",
            )


if __name__ == "__main__":
    batch_size = 1
    seq_len_list = _create_powers_of_2_list(512, 65536 / 2)  # Up to 32k
    embed_dim_list = [256]  # _create_powers_of_2_list(16,128)
    dtype_list = [torch.float32]

    benchmark_latency_triton_vs_naive(
        dtype_list, embed_dim_list, seq_len_list, eval="both"
    )
    benchmark_memory_footprint_triton_vs_naive(dtype_list, embed_dim_list, seq_len_list)
