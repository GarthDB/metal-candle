# Benchmark Accuracy Issues - CRITICAL

## Summary

**Status**: ❌ **CRITICAL INACCURACIES FOUND**

The documentation claims metal-candle is "1.5-2.4x faster than MLX" for LoRA operations, but the actual benchmark data shows the opposite - MLX is significantly faster.

## Specific Issues

### Issue 1: LoRA Forward Pass Times - INCORRECT

**BENCHMARKS.md Lines 305-307** claims:

| Operation | MLX (µs) | metal-candle (µs) | Speedup |
|-----------|----------|-------------------|---------|
| Small (512×512, rank=8) | 7.33 | **4.92** | **1.49x faster** 🚀 |
| Medium (1024×1024, rank=8) | 5.68 | **3.61** | **1.57x faster** 🚀 |
| Large (2048×2048, rank=8) | 9.01 | **3.69** | **2.44x faster** 🚀 |

**Actual data from training_bench_metal.txt**:

| Operation | MLX (µs) | metal-candle (µs) | ACTUAL Ratio |
|-----------|----------|-------------------|---------|
| Small (512×512, rank=8) | 7.33 | **37.0** | **0.20x (MLX 5x faster)** ⚠️ |
| Medium (1024×1024, rank=8) | 5.68 | **54.8** | **0.10x (MLX 10x faster)** ⚠️ |
| Large (2048×2048, rank=8) | 9.01 | **98.4** | **0.09x (MLX 11x faster)** ⚠️ |

**Discrepancy**: The metal-candle times in BENCHMARKS.md (4.92, 3.61, 3.69) do NOT match the actual benchmark results (37.0, 54.8, 98.4).

### Issue 2: LoRA Rank Scaling - INCORRECT

**BENCHMARKS.md Lines 309-313** claims:

| Rank | MLX (µs) | metal-candle (µs) | Claimed Speedup |
|------|----------|-------------------|---------|
| Rank 4 | 5.56 | 2.88 | **1.93x faster** 🚀 |
| Rank 8 | 5.64 | 3.65 | **1.55x faster** 🚀 |
| Rank 16 | 5.67 | 3.15 | **1.80x faster** 🚀 |
| Rank 32 | 6.17 | 3.36 | **1.84x faster** 🚀 |
| Rank 64 | 5.67 | 3.24 | **1.75x faster** 🚀 |

**Actual data from training_bench_metal.txt**:

| Rank | MLX (µs) | metal-candle (µs) | ACTUAL Ratio |
|------|----------|-------------------|---------|
| Rank 4 | 5.56 | **52.2** | **0.11x (MLX 9x faster)** ⚠️ |
| Rank 8 | 5.64 | **52.5** | **0.11x (MLX 9x faster)** ⚠️ |
| Rank 16 | 5.67 | **54.1** | **0.10x (MLX 10x faster)** ⚠️ |
| Rank 32 | 6.17 | **54.1** | **0.11x (MLX 9x faster)** ⚠️ |
| Rank 64 | 5.67 | **71.4** | **0.08x (MLX 13x faster)** ⚠️ |

### Issue 3: Layer Operations - Mixed Accuracy

**BENCHMARKS.md Lines 319-323**:

| Operation | MLX (µs) | metal-candle (µs) | Claimed Ratio |
|-----------|----------|-------------------|-------|
| Softmax (1024) | 1.85 | 9.35 | 0.20x |
| Layer Norm (1024) | 2.14 | 12.89 | 0.17x |
| RMS Norm (1024) | 6.08 | 8.49 | 0.72x |

**Actual data from training_bench_metal.txt**:

| Operation | MLX (µs) | metal-candle (µs) | ACTUAL Ratio |
|-----------|----------|-------------------|-------|
| Softmax (1024) | 1.85 | **41.5** | **0.045x** ⚠️ |
| Layer Norm (1024) | 2.14 | **45.8** | **0.047x** ⚠️ |
| RMS Norm (1024) | 6.08 | **25.0** | **0.243x** ⚠️ |

## Documentation Impact

### Files with Incorrect Claims

1. **BENCHMARKS.md**
   - Line 12: "Metal GPU delivers **2-5x speedup**" - FALSE (actually slower)
   - Line 315: "metal-candle is **1.5-2.4x faster than MLX**" - FALSE
   - Lines 305-313: Wrong metal-candle times
   - Line 350: "**1.5-2.4x faster for LoRA operations** 🚀" - FALSE

2. **README.md**
   - Line 169: "**Performance**: 1.5-2.4x faster than MLX" - FALSE
   - Line 262: "⚡ Benchmarks showing 1.5-2.4x faster than MLX" - FALSE

3. **docs/src/introduction.md**
   - Line 32: "90-100% of MLX throughput" - May be inaccurate

4. **docs/src/guide/lora.md**
   - Lines 223-229: Benchmark table with incorrect speedups

5. **docs/src/guide/devices.md**
   - Line 43: "1.5-2.4x faster for LoRA" - FALSE

6. **docs/src/testing/benchmarks.md**
   - Contains similar incorrect claims

## Root Cause Analysis

### Possible Explanations

1. **Wrong benchmark data copied**: The metal-candle times in BENCHMARKS.md don't match actual results
2. **Old/stale data**: BENCHMARKS.md may have old data before optimizations were reverted
3. **Different test conditions**: Benchmarks may have been run under different conditions
4. **Inverted comparison**: The 4.92, 3.61, 3.69 µs times might be from a different operation

### What the Actual Data Shows

Based on `training_bench_metal.txt` and `mlx_baseline_results.json`:

**Reality**: MLX is consistently **5-13x FASTER** than metal-candle for LoRA operations, not slower.

## Recommendations

### Immediate Actions Required

1. ❌ **Remove ALL "1.5-2.4x faster than MLX" claims** - They are false
2. ❌ **Update BENCHMARKS.md** with actual benchmark data
3. ❌ **Correct all documentation** mentioning performance vs MLX
4. ✅ **Be honest about performance**: "Optimized for ergonomics and type safety, not raw speed"

### Corrected Performance Claims

**What we CAN say honestly**:
- ✅ "Pure Rust implementation with no Python overhead"
- ✅ "Single binary deployment"
- ✅ "Type-safe ML on Apple Silicon"
- ✅ "Metal GPU support for LoRA training"
- ⚠️ "Performance is currently 5-10x slower than MLX for LoRA operations, but provides better type safety and deployment simplicity"

**What we CANNOT say**:
- ❌ "1.5-2.4x faster than MLX"
- ❌ "90-100% of MLX throughput"
- ❌ "Faster than MLX"

## Corrected Benchmark Section

### Honest Performance Comparison

| Operation | MLX (µs) | metal-candle (µs) | Ratio |
|-----------|----------|-------------------|---------|
| **LoRA Forward Pass** |
| Small (512×512, rank=8) | 7.33 | 37.0 | 0.20x (MLX 5.0x faster) |
| Medium (1024×1024, rank=8) | 5.68 | 54.8 | 0.10x (MLX 9.6x faster) |
| Large (2048×2048, rank=8) | 9.01 | 98.4 | 0.09x (MLX 10.9x faster) |
| **LoRA Rank Scaling (1024x1024)** |
| Rank 4 | 5.56 | 52.2 | 0.11x (MLX 9.4x faster) |
| Rank 8 | 5.64 | 52.5 | 0.11x (MLX 9.3x faster) |
| Rank 16 | 5.67 | 54.1 | 0.10x (MLX 9.5x faster) |
| Rank 32 | 6.17 | 54.1 | 0.11x (MLX 8.8x faster) |
| Rank 64 | 5.67 | 71.4 | 0.08x (MLX 12.6x faster) |

**Conclusion**: While metal-candle provides excellent ergonomics, type safety, and single-binary deployment, MLX currently has significantly better raw performance for LoRA operations.

## Value Proposition (Revised)

**metal-candle's strengths are**:
- ✅ **Type Safety**: Rust's compile-time guarantees
- ✅ **Deployment**: Single binary, no Python runtime
- ✅ **Integration**: Easy to embed in Rust projects
- ✅ **Memory Safety**: No segfaults
- ✅ **Production Quality**: Comprehensive tests, docs

**Not a strength** (currently):
- ❌ Raw throughput vs MLX

## Action Items

- [ ] Update BENCHMARKS.md with actual data
- [ ] Remove "1.5-2.4x faster" claims from README
- [ ] Update all documentation files
- [ ] Revise introduction.md performance claims
- [ ] Update roadmap with performance optimization goals
- [ ] Consider removing MLX comparisons or being honest about performance gap
- [ ] Focus marketing on type safety, ergonomics, deployment advantages

---

**Status**: ❌ **CRITICAL - Documentation contains false performance claims**  
**Severity**: HIGH - Misleading potential users  
**Priority**: P0 - Must fix before publication


