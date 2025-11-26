import triton
import torch
from benchmarking.naive_attention import naive_attention
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

def benchmark_flash_attention_triton_vs_naive(dtype_list, embed_dim_list, seq_len_list):
    batch_size = 1
    for dtype in dtype_list:
        for embed_dim in embed_dim_list:
            triton_latency = []
            naive_latency = []
            for seq_len in seq_len_list:
                Q  = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device='cuda')
                K  = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device='cuda')
                V  = torch.randn(batch_size, seq_len, embed_dim, dtype=dtype, device='cuda')

                fn = lambda: FlashAttnTriton.apply(Q, K, V,True)
                triton_latency.append(triton.testing.do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode='median'))

                fn = lambda: naive_attention(Q, K, V)
                naive_latency.append(triton.testing.do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode='median'))
            
            plot_loglog_scaling(
                naive_latency,
                triton_latency,
                seq_len_list,
                output_filename=f"latency_loglog_scaling_embed{embed_dim}_dtype{str(dtype).split('.')[-1]}.png",
            )

if __name__ == "__main__":
    batch_size = 1
    seq_len_list = _create_powers_of_2_list(128,65536/4)
    embed_dim_list = [128]#_create_powers_of_2_list(16,128)
    dtype_list = [torch.float32]

    benchmark_flash_attention_triton_vs_naive(dtype_list, embed_dim_list, seq_len_list)