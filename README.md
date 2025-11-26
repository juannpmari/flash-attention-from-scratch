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

## Forward pass latency

embedding_dimension = 128
dtype = torch.float32

### Latency vs sequence length
Insert plot here

Key insigths:
- Flash-attention is several times faster than the naive implementation



## Backward pass latency

## End to end latency