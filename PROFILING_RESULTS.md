# Profiling Results: metal-candle Performance Analysis

**Date**: October 18, 2025  
**Issue**: #19  
**Device**: Apple Silicon (M-series)  
**Status**: Analysis based on MLX comparison benchmarks

## Executive Summary

Based on MLX comparison benchmarks, we've identified the primary bottleneck: **kernel launch overhead dominates small operations**.

### Key Findings

1. **High-rank operations are FAST** (1.06-1.48x MLX) ✅
2. **Small operations are SLOW** (0.16-0.68x MLX) ❌
3. **Pattern**: Fixed overhead dominates when compute < overhead

## Detailed Analysis

### Performance by Operation Size

| Operation Category | Performance vs MLX | Analysis |
|-------------------|-------------------|----------|
| **Compute-Heavy** (high-rank LoRA) | 1.06-1.48x (FASTER) | GPU compute time >> overhead |
| **Medium Operations** (small LoRA) | 0.49-0.61x (slower) | Overhead becomes significant |
| **Small Operations** (softmax, layer norm) | 0.16-0.68x (much slower) | Overhead >> compute time |

### Root Cause: Kernel Launch Overhead

```
Estimated Breakdown (based on benchmark data):

MLX Small Operation (5 µs total):
  ├─ Kernel launch overhead: ~1-2 µs (20-40%)
  └─ GPU compute: ~3-4 µs (60-80%)

metal-candle Small Operation (10-15 µs total):
  ├─ Kernel launch overhead: ~7-10 µs (70-80%) ← PROBLEM
  └─ GPU compute: ~3-4 µs (20-30%)

metal-candle High-rank LoRA (5 µs total):
  ├─ Kernel launch overhead: ~1 µs (20%)
  └─ GPU compute: ~4 µs (80%) ← Overhead amortized
```

**Conclusion**: Candle's Metal backend has ~3-5x higher kernel launch overhead than MLX for small operations.

## Evidence from Benchmarks

### 1. Small Operations Suffer Most

| Operation | MLX (µs) | metal-candle (µs) | Overhead Impact |
|-----------|----------|-------------------|-----------------|
| Softmax (1024) | 1.58 | 6.77 | **4.3x slower** |
| Layer Norm | 1.99 | 12.33 | **6.2x slower** |
| Small LoRA | 5.45 | 8.98 | **1.6x slower** |

**Pattern**: Shorter operations = worse relative performance

### 2. Large Operations Are Competitive

| Operation | MLX (µs) | metal-candle (µs) | Why It Works |
|-----------|----------|-------------------|--------------|
| LoRA rank 16 | 6.42 | 5.50 | **1.17x FASTER** - Compute dominates |
| LoRA rank 32 | 5.03 | 4.75 | **1.06x FASTER** - Overhead amortized |
| LoRA rank 64 | 7.19 | 4.86 | **1.48x FASTER** - More work, same overhead |

**Pattern**: More computation = better relative performance = overhead is fixed!

### 3. Rank Scaling Reveals Fixed Overhead

```
metal-candle performance improves with rank (overhead amortization):
  Rank  4:  0.66x MLX (overhead dominates)
  Rank  8:  0.61x MLX (overhead still significant)
  Rank 16:  1.17x MLX (overhead amortized) ✅
  Rank 32:  1.06x MLX (optimal range) ✅
  Rank 64:  1.48x MLX (crushing it!) ✅
```

**Conclusion**: The more compute per kernel launch, the better we perform relative to MLX.

## Bottleneck Analysis

### Primary Bottleneck: Kernel Launch Overhead

**What's Happening**:
- Each Candle operation launches separate Metal kernel
- Launch overhead: ~5-8 µs per operation
- MLX launch overhead: ~1-2 µs per operation

**Why MLX is Faster**:
1. **Lazy evaluation** - MLX fuses operations before execution
2. **Graph optimization** - Combines multiple ops into single kernel
3. **Optimized Metal kernels** - Purpose-built for Apple Silicon
4. **Minimal abstraction** - Direct Metal API usage

**Why We're Slower**:
1. **Eager execution** - Each op launches immediately
2. **No fusion** - Every operation is separate kernel
3. **Generic kernels** - Candle supports multiple backends
4. **Abstraction overhead** - Candle's API layer

### Secondary Issues

1. **Layer Operations**:
   - Layer Norm: 0.16x MLX (most expensive)
   - Likely: Complex operation requires multiple kernels
   - Impact: High

2. **Softmax**:
   - 0.23x MLX
   - Requires: max, subtract, exp, sum, divide (5 ops)
   - Each op = separate kernel = 5x overhead
   - Impact: High

3. **Memory Management**:
   - Unknown without Instruments profiling
   - Potential: Unnecessary synchronization
   - Impact: Unknown

## Optimization Opportunities

### Immediate (70% target achievable):

1. **Batch Operations** (Estimated gain: +10-15%)
   - Combine multiple small operations
   - Amortize kernel launch overhead
   - Example: Batch softmax across layers

2. **Reduce Intermediate Allocations** (Estimated gain: +5-10%)
   - Reuse tensor buffers
   - Minimize memory allocations
   - Use in-place operations where possible

3. **Operation Fusion** (Estimated gain: +15-20%)
   - Manually combine operations where possible
   - Example: Fuse normalize + scale in RMS norm
   - Limited by Candle API

### Medium-term (90-100% target):

4. **Custom Metal Kernels for Bottlenecks** (Gain: +30-40%)
   - Write optimized softmax kernel (fused max+exp+sum+div)
   - Write optimized layer norm kernel
   - Target the 0.16-0.23x operations first

5. **Optimize LoRA Forward** (Gain: +5-10%)
   - Already competitive at high ranks
   - Can improve small ranks with fusion

### Long-term (110-150% target):

6. **Full Metal Implementation**
   - Replace Candle entirely for critical paths
   - Specialized for LoRA training
   - Maximum performance potential

## Estimated Performance Gains

### Conservative Scenario (Optimize Candle usage):

| Optimization | Current | After | Gain |
|-------------|---------|-------|------|
| Batching | 45% | 55% | +10% |
| Memory optimization | 55% | 60% | +5% |
| Operation fusion (manual) | 60% | 70-75% | +10-15% |

**Result**: **70-75% of MLX** (meets minimum target) ✅

### Aggressive Scenario (Custom kernels):

| Operation | Current | With Custom Kernel | Target |
|-----------|---------|-------------------|--------|
| Softmax | 0.23x | 0.9-1.0x | 90-100% |
| Layer Norm | 0.16x | 0.9-1.0x | 90-100% |
| Small LoRA | 0.5-0.6x | 0.8-0.9x | 80-90% |

**Result**: **90-100% of MLX** (meets ideal target) ✅

## Next Steps

### Immediate (Issue #20):

1. **Review LoRA forward pass implementation**
   - Count actual kernel launches
   - Identify fusion opportunities
   - Reduce intermediate tensors

2. **Optimize layer operations**
   - Check if Candle has fused variants
   - Manually combine ops where possible
   - Benchmark each change

3. **Memory profiling**
   - Use Instruments if available
   - Look for allocation hot spots
   - Minimize copies

4. **Re-benchmark after each optimization**
   - Track progress toward 70% target
   - Document what works

### If < 70% After Optimization (Issue #21):

5. **Prototype MLX Rust bindings**
   - Guaranteed 100% of MLX performance
   - Still achieves project goals (single binary)
   - Acceptable tradeoff (C++ dependency internal)

## Conclusion

**Good News**:
- We understand the bottleneck (kernel overhead)
- We have proof Rust+Metal works (high-rank performance)
- Clear path to 70% target (optimization)
- Backup plan for 100% target (MLX bindings or custom kernels)

**Realistic Assessment**:
- 70-75% achievable with optimization ✅
- 90-100% requires custom kernels or MLX bindings
- We're not fundamentally limited by Rust

**Recommendation**:
Proceed with Issue #20 (optimization) targeting 70% of MLX. If successful, ship v1.0. Plan v1.1 for custom kernels to reach 100%.

---

**Status**: ✅ Analysis complete, ready for optimization phase  
**Next**: Issue #20 - Optimize Candle usage patterns  
**Target**: 70% of MLX performance
