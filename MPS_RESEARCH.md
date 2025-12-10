# Metal Performance Shaders (MPS) Research

## Overview

**Goal**: Investigate using Apple's Metal Performance Shaders to achieve MLX-level performance

**Current State**: Our custom kernels achieve 1-2x speedups
**Target**: MPS could provide 5-20x speedups (matching MLX)

## What is MPS?

**Metal Performance Shaders** is Apple's collection of highly optimized GPU kernels for:
- Matrix operations (multiply, add, etc.)
- Neural network operations (convolution, pooling, etc.)
- Image processing
- Linear algebra

**Key Advantage**: Hand-optimized by Apple, assembly-level tuning, direct tensor core access

## MPS Classes Relevant to ML

### Matrix Operations

**MPSMatrixMultiplication**:
- Highly optimized matrix multiplication
- `C = αAB + βC`
- Supports various data types (float16, float32, int8)
- **Expected performance**: 5-20x faster than naive custom kernels

**MPSMatrixSoftMax**:
- Optimized softmax operation
- **Expected performance**: 10-20x faster than custom kernels

**MPSMatrixNeuron**:
- Various activation functions
- Includes ReLU, Sigmoid, TanH, etc.

### Neural Network Operations

**MPSNNGraph**:
- Graph-based neural network execution
- Automatic optimization and fusion
- Might be too high-level for our needs

## MPS vs Custom Kernels vs MLX

| Operation | Custom Kernel | MPS (Expected) | MLX (Actual) |
|-----------|---------------|----------------|--------------|
| **LoRA Matmul** | 36.51 µs | 3-8 µs | 5-11 µs |
| **Softmax** | 39.45 µs | 2-4 µs | 1.85 µs |
| **RMS Norm** | 46.92 µs | 5-10 µs | 6.08 µs |

**Hypothesis**: MPS would match or exceed MLX performance

## Integration Approaches

### Approach A: Direct MPS Calls from CustomOp

**Architecture**:
```rust
// In custom_ops.rs
impl CustomOp1 for MPSMatrixMulOp {
    fn metal_fwd(&self, storage: &MetalStorage, ...) -> Result<...> {
        // Get Metal device
        let device = storage.device();
        
        // Create MPS command buffer
        let command_buffer = MPSCommandBuffer::from_metal(device);
        
        // Create MPSMatrixMultiplication
        let matmul = MPSMatrixMultiplication::new(
            device,
            transpose_a, transpose_b,
            m, n, k,
            alpha, beta
        );
        
        // Create MPSMatrix wrappers around Metal buffers
        let matrix_a = MPSMatrix::from_buffer(input_buffer, ...);
        let matrix_b = MPSMatrix::from_buffer(weight_buffer, ...);
        let matrix_c = MPSMatrix::from_buffer(output_buffer, ...);
        
        // Encode operation
        matmul.encode(command_buffer, matrix_a, matrix_b, matrix_c);
        
        // Commit (Candle handles sync)
        command_buffer.commit();
        
        Ok(output_storage)
    }
}
```

**Pros**:
- Clean integration with CustomOp framework
- Reuses our existing architecture
- Easy to test and benchmark

**Cons**:
- Need metal-rs bindings for MPS (may not exist)
- Command buffer coordination with Candle

### Approach B: Hybrid - MPS for Standard, Custom for Unique

**Strategy**:
```rust
// Use MPS where available
tensor.matmul() -> MPSMatrixMultiplication (via custom op)
tensor.softmax() -> MPSMatrixSoftMax (via custom op)

// Custom kernels for unique ops
tensor.lora_forward_fused() -> Custom kernel (LoRA-specific)
tensor.flash_attention() -> Custom kernel (not in MPS)
```

**Pros**:
- Best of both worlds
- Leverage Apple's optimization
- Still have flexibility for custom ops

**Cons**:
- More complexity
- Need to maintain both paths

### Approach C: MPS at Candle Level

**Strategy**: Contribute MPS backend to Candle upstream

**Pros**:
- Benefits entire Candle ecosystem
- Proper integration
- Long-term sustainable

**Cons**:
- Much larger effort (weeks)
- Upstream coordination needed
- Not immediate solution

## Technical Challenges

### Challenge 1: metal-rs MPS Bindings

**Question**: Does `metal-rs` crate provide MPS bindings?

**Research Needed**:
```bash
# Check metal-rs source
grep -r "MPSMatrix" ~/.cargo/registry/src/*/metal-*

# Check documentation
cargo doc --open --package metal
```

**If NO bindings exist**: Need to create them via `objc` crate
**If bindings exist**: Can use directly

### Challenge 2: Metal Buffer Compatibility

**Question**: Can we create MPSMatrix from Candle's Metal buffers?

**Key APIs**:
```objc
// Objective-C/Swift
let matrix = MPSMatrix(buffer: metalBuffer, 
                       descriptor: desc)
```

**Need to verify**: Buffer layout compatibility

### Challenge 3: Command Buffer Coordination

**Issue**: Candle manages its own command buffers

**Options**:
1. Use Candle's command buffer for MPS ops
2. Create separate MPS command buffer, sync with barriers
3. Let MPS create its own, rely on Metal's automatic sync

### Challenge 4: Data Layout

**MPS Expectations**:
- Row-major or column-major
- Specific stride requirements
- Data type alignment

**Candle's Layout**:
- May not match MPS expectations
- Might need transpose/reshape

## Implementation Plan

### Phase 1: Feasibility Check (2-4 hours)

**Goal**: Determine if MPS is accessible from Rust

**Tasks**:
1. ✅ Check if metal-rs has MPS bindings
2. ✅ If not, explore objc-based bindings
3. ✅ Create minimal MPS matmul example
4. ✅ Benchmark vs custom kernel
5. ✅ Assess complexity

**Deliverable**: Go/No-Go decision document

### Phase 2: Prototype Integration (4-8 hours)

**If Phase 1 is GO**:

**Tasks**:
1. Create `MPSMatrixMulOp` CustomOp
2. Wire up to `CustomMetalOps` trait
3. Add correctness tests
4. Benchmark vs current + MLX
5. Document findings

**Deliverable**: Working prototype with benchmarks

### Phase 3: Production Implementation (1-2 weeks)

**If Phase 2 shows promise**:

**Tasks**:
1. Implement all relevant MPS operations
2. Handle edge cases
3. Comprehensive testing
4. Performance optimization
5. Documentation

**Deliverable**: Production-ready MPS integration

## Expected Outcomes

### Best Case

**MPS matmul**: 3-8 µs (vs 36 µs custom) → **4-12x speedup**
**MPS softmax**: 2-4 µs (vs 39 µs custom) → **10-20x speedup**

**Result**: Match or exceed MLX performance ✅

### Realistic Case

**Some overhead** from Rust/Metal boundary
**MPS matmul**: 5-12 µs → **3-7x speedup**
**MPS softmax**: 3-6 µs → **6-13x speedup**

**Result**: Close to MLX performance (within 2x) ✅

### Worst Case

**MPS bindings don't exist** or **integration too complex**

**Fallback**: 
- Document MPS as future work
- Ship current custom kernel implementation
- Focus on other value props

## Decision Criteria

### Go Ahead with MPS if:
- ✅ metal-rs has MPS bindings OR objc bindings are straightforward
- ✅ Simple example works in < 4 hours
- ✅ Benchmark shows > 3x improvement
- ✅ Integration complexity is manageable

### Don't Pursue MPS if:
- ❌ No easy way to access MPS from Rust
- ❌ Complex C++ wrapper needed
- ❌ Minimal performance gain
- ❌ Would take weeks to implement properly

## Next Steps

1. **Check metal-rs for MPS bindings** ✅ (next)
2. **Try minimal MPS example** (if bindings exist)
3. **Benchmark** (if example works)
4. **Decide** (based on results)

---

**Status**: Research phase
**Time investment so far**: 1 hour
**Decision point**: After feasibility check

