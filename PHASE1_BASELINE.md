# Phase 1: Baseline Performance Analysis

**Date**: December 8, 2025  
**Objective**: Establish accurate baseline for hybrid Metal optimization

## MLX Baseline Results (Fresh Run)

Ran fresh MLX benchmarks using `benches/mlx_baseline.py`:

**MLX Version**: 0.30.0  
**Device**: Device(gpu, 0)

### LoRA Forward Pass

| Operation | MLX (µs) |
|-----------|----------|
| small_512x512_r8 | 5.79 |
| medium_1024x1024_r8 | 5.24 |
| large_2048x2048_r8 | 11.86 |

### Layer Operations

| Operation | MLX (µs) |
|-----------|----------|
| softmax_stable | 5.04 |
| layer_norm | 2.41 |
| rms_norm | 4.96 |

### LoRA Rank Scaling (1024x1024)

| Rank | MLX (µs) |
|------|----------|
| 4 | 5.50 |
| 8 | 8.35 |
| 16 | 5.25 |
| 32 | 5.52 |
| 64 | 5.30 |

## metal-candle Current Performance

From BENCHMARKS.md (Metal GPU - Apple Silicon):

### LoRA Forward Pass

| Operation | metal-candle (µs) |
|-----------|-------------------|
| 512x512, r=8 | 37.0 |
| 1024x1024, r=8 | 54.8 |
| 2048x2048, r=8 | 98.4 |

### Layer Operations

| Operation | metal-candle (µs) |
|-----------|-------------------|
| Softmax (1024) | 41.5 |
| Layer Norm (1024) | 45.8 |
| RMS Norm (1024) | 25.0 |

### LoRA Rank Scaling (1024x1024, Metal GPU)

| Rank | metal-candle (µs) |
|------|-------------------|
| 4 | 52.2 |
| 8 | 52.5 |
| 16 | 54.1 |
| 32 | 54.1 |
| 64 | 71.4 |

## Performance Gap Analysis

### LoRA Operations

| Operation | MLX (µs) | metal-candle (µs) | Gap (x slower) |
|-----------|----------|-------------------|----------------|
| Small (512x512, r=8) | 5.79 | 37.0 | **6.4x** |
| Medium (1024x1024, r=8) | 5.24 | 54.8 | **10.5x** |
| Large (2048x2048, r=8) | 11.86 | 98.4 | **8.3x** |

**Average LoRA Gap**: **8.4x slower**

### Layer Operations

| Operation | MLX (µs) | metal-candle (µs) | Gap (x slower) |
|-----------|----------|-------------------|----------------|
| Softmax | 5.04 | 41.5 | **8.2x** |
| Layer Norm | 2.41 | 45.8 | **19.0x** |
| RMS Norm | 4.96 | 25.0 | **5.0x** |

**Average Layer Ops Gap**: **10.7x slower**

## Key Findings

### Primary Bottlenecks

1. **LoRA Forward Pass**: 6-10x slower than MLX
   - Multiple kernel launches (matmul A, matmul B, scaling, addition)
   - Each operation triggers GPU overhead
   - No fusion of operations

2. **Layer Norm**: 19x slower than MLX
   - Most critical bottleneck
   - Multiple reduction operations not fused

3. **Softmax**: 8.2x slower than MLX
   - Max reduction + exp + sum reduction + division
   - Each step is separate kernel launch

4. **RMS Norm**: 5x slower (least bad)
   - Simpler than Layer Norm
   - Still room for improvement

### Performance Target

**Goal**: Achieve 95-110% of MLX performance

**Required Improvement**:
- LoRA operations: Need **8-10x speedup** to match MLX
- Layer operations: Need **5-19x speedup** to match MLX

### Strategy

Based on profiling data and benchmarks:

1. **Highest Impact**: Fused LoRA kernel
   - Currently: 2+ kernel launches
   - Target: 1 kernel launch
   - Expected gain: **5-8x speedup**

2. **Second Priority**: Fused softmax kernel
   - Currently: 4+ operations
   - Target: 1 kernel with shared memory
   - Expected gain: **6-8x speedup**

3. **Third Priority**: Fused RMS Norm
   - Currently: 3+ operations  
   - Target: 1 kernel with threadgroup reduction
   - Expected gain: **4-5x speedup**

4. **Fourth Priority**: Fused Layer Norm
   - Currently: 5+ operations
   - Target: 1 kernel with shared memory
   - Expected gain: **10-15x speedup**

## Next Steps (Phase 2)

1. Set up Metal shader infrastructure (metal-rs)
2. Create Metal kernel compiler and integration layer
3. Implement fused LoRA kernel (biggest impact)
4. Benchmark and validate correctness
5. Implement layer operation kernels
6. Achieve 95-110% of MLX performance

## Success Criteria

- LoRA forward: 5-6 µs (currently 37-98 µs)
- Softmax: 4-5 µs (currently 41.5 µs)
- Layer Norm: 2-3 µs (currently 45.8 µs)
- RMS Norm: 4-5 µs (currently 25.0 µs)

---

**Status**: Baseline established ✅  
**Next Phase**: Metal shader infrastructure

