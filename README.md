# Introduction
This is a custom implementation of the Flash Attention 2 paper[] using Triton[].

Attention is a memory-bound operation. When computed naively, it requires O(N2) data transfer from HBM to SRAM and back to HBM.

Flash Attention is a highly optimized attention implementation that leverages GPU memory hierarchy and 3 well stablished GPU programming techniques:
- kernel fusion
- tiling
- (dont remember)
This is done by moving tensors ...

This effectively reduces data transfer from O(N2) to O()

# Implementation details
This repo presents a Flash Attention implementation using Triton, a cuda compiler released by OpenAI in 2023

# Benchmarking

Experiments where run on a GeForce RTX 4050 (6Gb), CUDA 13.0

### Forward pass

embedding_dimension = 128
dtype = torch.float32
tile_size = 16
The following plot shows latency vs sequence length:
[Insert plot here resources/latency_loglog_scaling_embed128_dtypefloat32.png]

The following plot shows memory footprint vs sequence length:
[Insert plot here resources/memory_loglog_scaling_embed128_dtypefloat32.png]

Key insigths:
- As can be seen, the shape of the plot is very similar to that of fig 3 from the original paper.
- Flash-attention is several times faster than the naive implementation


## Backward pass latency

## End to end latency