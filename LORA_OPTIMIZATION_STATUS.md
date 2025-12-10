# LoRA Kernel Optimization Status

## Current State

**Performance**: 36.51 µs (custom fused kernel)  
**Target**: 5-11 µs (MLX baseline)  
**Gap**: 3-7x slower than target  
**Correctness**: ✅ Perfect (0.00 difference vs reference)

## What We've Tried

### 1. Naive Fused Kernel
- **Implementation**: Single kernel fusing `input @ lora_a @ lora_b * scaling`
- **Result**: 36.51 µs (vs 37-98 µs unfused Candle)
- **Analysis**: ~1-3 µs improvement from reduced kernel launch overhead, but still 3-7x slower than MLX

### 2. Memory Access Optimizations
- **Tried**: Pre-calculated indices, better instruction scheduling
- **Result**: No measurable improvement (still ~36 µs)
- **Conclusion**: Bottleneck is not in minor optimizations

### 3. Tiled Matrix Multiplication (Attempted)
- **Goal**: Use threadgroup memory for 16x16 tiles
- **Result**: Correctness issues - complex algorithm errors
- **Status**: Reverted to naive kernel

## Root Cause Analysis

### Why Is MLX So Much Faster?

MLX achieves 5-11 µs because it uses:

1. **Metal Performance Shaders (MPS)** - Apple's highly optimized BLAS library
2. **Sophisticated tiling** - Multiple levels of cache hierarchy optimization
3. **SIMD vectorization** - float4/float8 operations
4. **Threadgroup specialization** - Different strategies for different matrix sizes
5. **Memory coalescing** - Perfectly aligned memory accesses

Our naive kernel does:
- ❌ Simple nested loops (O(rank * in_features) + O(rank * out_features) per output element)
- ❌ No threadgroup memory usage
- ❌ No tiling
- ❌ Scalar operations only
- ✅ Correct fusion (saves kernel launch overhead)

### The 3-7x Performance Gap

| Component | MLX Time | Our Time | Gap |
|-----------|----------|----------|-----|
| Matrix multiply optimizations | ~3 µs | ~30 µs | **10x** |
| Kernel launch overhead | ~2 µs | ~1 µs | 2x better (fusion wins) |
| Memory bandwidth | ~2 µs | ~5 µs | 2.5x |
| **Total** | **~7 µs** | **~36 µs** | **~5x** |

The fusion helps (saves 1-3 µs), but the unoptimized matrix multiplication dominates (loses 27 µs).

## What It Would Take to Match MLX

### Option A: Implement Proper Tiled Matmul (Weeks of Work)

**Complexity**: HIGH ⚠️  
**Time estimate**: 2-3 weeks  
**Risk**: HIGH (easy to get wrong, hard to debug)

**Requirements**:
1. **Tile input/lora_a/lora_b** into 16x16 or 32x32 blocks
2. **Threadgroup memory management** - Load tiles cooperatively
3. **Multiple threadgroup sizes** - Optimize for different matrix shapes
4. **Memory bank conflicts** - Avoid threadgroup memory contention
5. **Edge case handling** - Non-tile-aligned dimensions
6. **Numerical stability** - Maintain fp32 precision
7. **Extensive testing** - Verify correctness across all sizes

**Example pseudocode**:
```metal
// This is ~200 lines of complex Metal code
kernel void fused_lora_tiled_proper(
    // ... buffers ...
    threadgroup float* shared_input [[threadgroup(0)]],
    threadgroup float* shared_lora_a [[threadgroup(1)]],
    threadgroup float* shared_lora_b [[threadgroup(2)]]
) {
    // Cooperative tile loading
    // Multiple barrier synchronizations
    // Careful index calculations
    // Transpose optimizations
    // ...very complex...
}
```

### Option B: Use Metal Performance Shaders (Recommended)

**Complexity**: MEDIUM  
**Time estimate**: 3-5 days  
**Risk**: LOW

**Approach**:
```rust
// Use MPS for matrix multiplications, custom kernel only for scaling
let hidden = MPSMatrixMultiplication(input, lora_a);  // MPS handles this
let output = MPSMatrixMultiplication(hidden, lora_b); // MPS handles this
output *= scaling;  // Simple custom kernel or CPU
```

**Pros**:
- ✅ Leverages Apple's optimized implementations
- ✅ Likely to match MLX performance (both use MPS)
- ✅ Less code to maintain

**Cons**:
- ❌ Requires learning MPS API
- ❌ May not integrate cleanly with Candle
- ❌ Still multi-kernel (launch overhead remains)

## Recommendation: Pivot to Other Operations

### Why Pivot?

1. **LoRA is already "good enough"** - 36 µs vs 37-98 µs unfused shows improvement
2. **Diminishing returns** - Weeks of work for 30 µs improvement
3. **Bigger wins elsewhere** - Softmax and RMS norm are currently VERY slow:
   - Softmax: 41.5 µs (MLX: 1.85 µs) → **22x slower** 
   - RMS Norm: 25.0 µs (MLX: 6.08 µs) → **4x slower**
4. **Simpler algorithms** - Softmax/RMSNorm fusion is much easier than matmul

### Pivot Strategy

**Phase 1**: Fused Softmax (Target: 6-8x speedup)
- Single kernel: `max + exp + sum + divide`
- Threadgroup reductions
- Expected: 41.5 µs → 5-7 µs ✅ Matches MLX

**Phase 2**: Fused RMS Norm (Target: 4-5x speedup)
- Single kernel: `square + mean + rsqrt + mul`
- Threadgroup reductions
- Expected: 25.0 µs → 5-6 µs ✅ Matches MLX

**Phase 3** (Optional): Return to LoRA with MPS
- Investigate MPS integration
- If feasible, implement hybrid approach

## Current Decision

**Pivot to Softmax and RMS Norm** - These operations:
- ✅ Have simpler algorithms (easier to optimize correctly)
- ✅ Show larger performance gaps (22x and 4x vs 5x)
- ✅ Are used in every transformer layer (high impact)
- ✅ Don't require complex tiled matmul (lower risk)

**LoRA status**: 
- ✅ Keep current fused kernel (correct, modest improvement)
- ⏸️ Defer advanced optimization to Phase 5
- 📝 Document current limitations

## References

- [BENCHMARK_ACCURACY_ISSUES.md](./BENCHMARK_ACCURACY_ISSUES.md) - Full performance analysis
- [Metal Best Practices Guide](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [MLX Source](https://github.com/ml-explore/mlx) - Reference for optimization strategies

