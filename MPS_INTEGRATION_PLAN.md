# Metal Performance Shaders (MPS) Integration Plan

**Goal**: Achieve MLX-level performance (5-20x speedup) using Apple's optimized Metal Performance Shaders  
**Timeline**: 2-3 weeks  
**Target**: metal-candle v2.0  
**Status**: 📋 Planning Phase

---

## Executive Summary

### Why MPS?

Metal Performance Shaders are Apple's hand-optimized, assembly-level GPU kernels that provide:
- **5-20x faster** than our current custom kernels
- **Direct tensor core access** on Apple Silicon
- **Battle-tested** by Apple's own frameworks (Core ML, MLX likely uses them)
- **Zero maintenance** - optimized by Apple for each chip generation

### Current Performance Gap

| Operation | Our Custom | MLX (likely MPS) | Gap |
|-----------|------------|------------------|-----|
| LoRA | 37-98 µs | 5-11 µs | **6-10x** |
| Softmax | 39 µs | 5 µs | **7.8x** |
| RMS Norm | 47 µs | 5 µs | **9.4x** |

### Expected Outcome

With MPS integration:
- LoRA: **37µs → 5-10µs** (4-7x faster)
- Softmax: **39µs → 3-5µs** (8-13x faster)
- RMS Norm: **47µs → 4-6µs** (8-12x faster)
- **Total**: Match or exceed MLX performance

---

## Phase 1: Research & Prototyping (Week 1)

### Day 1-2: MPS API Research

**Objective**: Understand MPS capabilities and identify target operations

**Tasks**:
1. ✅ Study `MPSMatrixMultiplication` documentation
2. ✅ Study `MPSMatrixSoftMax` documentation  
3. ✅ Research `MPSNNReduceFeatureChannelsMean` for RMS Norm
4. ✅ Identify MPS ops that map to our operations
5. ✅ Document API surface and Rust FFI requirements

**MPS Operations to Use**:

| Our Operation | MPS Equivalent | Expected Speedup |
|---------------|----------------|------------------|
| LoRA (A @ B) | `MPSMatrixMultiplication` | 5-10x |
| Softmax | `MPSMatrixSoftMax` | 10-15x |
| RMS Norm | `MPSNNReduceFeatureChannelsMean` + custom | 8-12x |
| LayerNorm | `MPSCNNInstanceNormalization` | 10-15x |
| Attention | `MPSMatrixMultiplication` chain | 5-8x |

**Deliverables**:
- `MPS_API_RESEARCH.md` - Comprehensive MPS API documentation
- List of operations to migrate
- FFI binding strategy

### Day 3-4: Rust FFI Bindings

**Objective**: Create safe Rust wrappers for MPS operations

**Approach**:
1. Use `objc` crate for Objective-C interop
2. Create type-safe wrappers around MPS classes
3. Handle memory management correctly (retain/release)
4. Map Metal buffers to MPS descriptors

**New Module**: `src/backend/mps.rs`

**Key Types to Wrap**:
```rust
// src/backend/mps.rs

/// Wrapper for MPSMatrixDescriptor
pub struct MPSMatrixDescriptor {
    inner: *mut Object,
}

/// Wrapper for MPSMatrix
pub struct MPSMatrix {
    inner: *mut Object,
}

/// Wrapper for MPSMatrixMultiplication
pub struct MPSMatrixMultiplication {
    inner: *mut Object,
}

/// High-level MPS operation trait
pub trait MPSOperation {
    fn encode_to_command_buffer(&self, buffer: &metal::CommandBufferRef);
}
```

**Safety Invariants**:
- Always retain MPS objects on creation
- Release on drop
- Never expose raw pointers publicly
- Validate tensor shapes before MPS dispatch

**Deliverables**:
- `src/backend/mps.rs` (200-300 lines)
- Type-safe FFI wrappers
- Memory management tests

### Day 5: Prototype Integration

**Objective**: Get one MPS operation working end-to-end

**Target**: Matrix Multiplication (most critical for LoRA)

**Steps**:
1. Create MPS descriptor from Candle tensor
2. Wrap Metal buffer in MPSMatrix
3. Dispatch MPSMatrixMultiplication
4. Extract result back to Candle tensor
5. Verify correctness vs Candle's matmul

**Test Case**:
```rust
#[test]
fn test_mps_matmul_correctness() {
    let a = Tensor::randn(0.0f32, 1.0f32, (512, 256), &device)?;
    let b = Tensor::randn(0.0f32, 1.0f32, (256, 512), &device)?;
    
    // Candle reference
    let expected = a.matmul(&b)?;
    
    // MPS implementation
    let actual = mps_matmul(&a, &b)?;
    
    // Verify correctness
    let diff = (expected - actual)?.abs()?.max_all()?;
    assert!(diff.to_scalar::<f32>()? < 1e-5);
}
```

**Success Criteria**:
- ✅ Correctness: max error < 1e-5
- ✅ Performance: 2-5x faster than custom kernel
- ✅ No memory leaks (verified with Instruments)

**Deliverables**:
- Working MPS matmul prototype
- Correctness test passing
- Initial performance measurement

---

## Phase 2: Core Operations Migration (Week 2)

### Day 6-7: Matrix Multiplication

**Objective**: Production-quality MPS matmul for LoRA

**Implementation**: `src/backend/mps/matmul.rs`

**Features**:
- Handle batched matmul (3D tensors)
- Support transpose flags
- Optimal descriptor configuration
- Error handling for invalid shapes

**Integration**:
```rust
// src/backend/metal_ops.rs
impl CustomMetalOps for Tensor {
    fn matmul_mps(&self, rhs: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "mps")]
        {
            mps::matmul(self, rhs)
        }
        #[cfg(not(feature = "mps"))]
        {
            self.matmul(rhs)
        }
    }
}
```

**Testing**:
- 5 shape combinations
- Batched operations
- Transpose modes
- Performance benchmarks

**Deliverables**:
- `src/backend/mps/matmul.rs` (150-200 lines)
- 5+ correctness tests
- Performance benchmark showing 5-10x speedup

### Day 8: Softmax

**Objective**: MPS-accelerated softmax

**Implementation**: `src/backend/mps/softmax.rs`

**MPS API**: `MPSMatrixSoftMax`

**Challenges**:
- Dimension handling (MPS may only support last dim)
- Numerical stability (max subtraction)
- Shape preservation

**Fallback Strategy**:
```rust
pub fn mps_softmax(tensor: &Tensor, dim: i64) -> Result<Tensor> {
    if dim == -1 || dim == tensor.dims().len() as i64 - 1 {
        // Use MPS for last dimension
        mps_softmax_native(tensor)
    } else {
        // Fall back to custom kernel for other dims
        custom_softmax(tensor, dim)
    }
}
```

**Testing**:
- 2D and 3D tensors
- Different dimensions
- Numerical stability with large values
- Performance benchmark

**Deliverables**:
- `src/backend/mps/softmax.rs` (100-150 lines)
- 6+ correctness tests
- Performance benchmark showing 10-15x speedup

### Day 9: RMS Norm

**Objective**: MPS-accelerated RMS normalization

**Implementation**: `src/backend/mps/rms_norm.rs`

**Approach**:
1. Use `MPSNNReduceFeatureChannelsMean` for mean(x²)
2. Custom Metal kernel for `x / sqrt(mean + eps)`
3. Hybrid MPS + custom approach

**Alternative**:
- Full custom if MPS doesn't map well
- But optimize memory access patterns

**Testing**:
- 2D and 3D tensors
- Numerical properties (RMS ≈ 1.0)
- Edge cases (very small/large values)
- Performance benchmark

**Deliverables**:
- `src/backend/mps/rms_norm.rs` (100-150 lines)
- 5+ correctness tests
- Performance benchmark showing 8-12x speedup

### Day 10: LoRA Integration

**Objective**: End-to-end MPS LoRA forward pass

**Implementation**: Update `src/training/lora.rs`

**Approach**:
```rust
impl LoRALayer {
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "mps")]
        {
            // hidden = input @ lora_a
            let hidden = mps::matmul(input, &self.lora_a)?;
            
            // output = hidden @ lora_b
            let output = mps::matmul(&hidden, &self.lora_b)?;
            
            // Apply scaling
            output * self.config.scaling()
        }
        #[cfg(not(feature = "mps"))]
        {
            // Existing implementation
        }
    }
}
```

**Testing**:
- Compare MPS vs custom LoRA output
- Verify gradient computation still works
- End-to-end training test
- Performance benchmark

**Success Criteria**:
- ✅ Correctness: max error < 1e-5 vs custom
- ✅ Performance: 5-10x faster than custom
- ✅ Training still converges

**Deliverables**:
- Updated LoRALayer with MPS support
- 3+ integration tests
- Full training workflow test
- Benchmark showing **LoRA: 37µs → 5-10µs**

---

## Phase 3: Validation & Benchmarking (Week 3, Days 11-13)

### Day 11: Comprehensive Benchmarking

**Objective**: Measure actual speedups vs custom kernels and MLX

**Benchmark Suite**: `benches/mps_vs_custom.rs`

**Comparisons**:
1. MPS vs Custom (our baseline)
2. MPS vs MLX (competitive benchmark)
3. MPS vs Candle unfused (overall improvement)

**Operations to Benchmark**:
- Matrix multiplication (all LoRA sizes)
- Softmax (various dimensions)
- RMS Norm (various shapes)
- Full LoRA forward pass
- End-to-end training step

**Expected Results**:
```
LoRA Forward (512×512, r=8):
  Custom:  37.0 µs
  MPS:      5-7 µs  (5-7x faster) ✅
  MLX:      5.8 µs  (competitive) ✅

Softmax (1024):
  Custom:  39.4 µs
  MPS:      3-5 µs  (8-13x faster) ✅
  MLX:      5.0 µs  (competitive) ✅

RMS Norm (1024):
  Custom:  46.9 µs
  MPS:      4-6 µs  (8-12x faster) ✅
  MLX:      5.0 µs  (competitive) ✅
```

**Deliverables**:
- `benches/mps_vs_custom.rs` (300-400 lines)
- Comprehensive benchmark results
- `MPS_BENCHMARKS.md` with analysis
- Performance regression tests

### Day 12: Correctness Validation

**Objective**: Ensure MPS maintains perfect accuracy

**Test Strategy**:
1. Property-based testing (proptest)
2. Fuzzing with random inputs
3. Extreme value testing
4. Numerical stability verification

**Property Tests**:
```rust
proptest! {
    #[test]
    fn mps_matmul_associative(
        shape_a in (1usize..100, 1usize..100),
        shape_b in (1usize..100, 1usize..100),
    ) {
        // Generate random tensors
        let a = random_tensor(shape_a);
        let b = random_tensor(shape_b);
        
        // Compare MPS vs Candle
        let mps_result = mps_matmul(&a, &b)?;
        let candle_result = a.matmul(&b)?;
        
        // Verify within tolerance
        assert_tensors_close(&mps_result, &candle_result, 1e-5);
    }
}
```

**Edge Cases**:
- Zero matrices
- Identity matrices
- Very small values (1e-10)
- Very large values (1e10)
- NaN/Inf handling

**Deliverables**:
- Property-based test suite
- Edge case test coverage
- Correctness verification report

### Day 13: Memory & Stability Testing

**Objective**: Ensure production quality

**Memory Testing**:
```bash
cargo instruments -t Allocations --release --features mps --bench mps_vs_custom
```

**Checks**:
- No memory leaks (all MPS objects released)
- Reasonable peak memory (≤ custom kernels)
- No retain cycles
- Proper cleanup on error

**Stability Testing**:
- Run benchmarks 1000x (no crashes)
- Stress test with large tensors
- Concurrent execution (if applicable)
- Error recovery (invalid inputs)

**Deliverables**:
- Memory profiling report
- Stability test results
- Any fixes for leaks/crashes

---

## Phase 4: Integration & Documentation (Week 3, Days 14-15)

### Day 14: Feature Flag & Build System

**Objective**: Clean feature-gated integration

**Cargo.toml**:
```toml
[features]
default = ["custom-metal", "graph"]
mps = ["dep:objc", "custom-metal"]  # MPS requires Metal framework
custom-metal = ["dep:metal", "dep:objc"]
graph = ["dep:dashmap"]
async-exec = ["dep:tokio", "dep:async-trait"]

# Mutually exclusive optimization levels
fast = ["mps"]  # Maximum performance (MPS)
balanced = ["custom-metal"]  # Good performance (custom kernels)
portable = []  # CPU fallback
```

**Conditional Compilation**:
```rust
// src/backend/mod.rs
#[cfg(feature = "mps")]
pub mod mps;

#[cfg(feature = "mps")]
pub use mps::{matmul as fast_matmul, softmax as fast_softmax};

#[cfg(not(feature = "mps"))]
pub use custom_ops::{matmul as fast_matmul, softmax as fast_softmax};
```

**Build Testing**:
- ✅ `cargo build --features mps`
- ✅ `cargo build --no-default-features`
- ✅ `cargo test --all-features`
- ✅ `cargo bench --features mps`

**Deliverables**:
- Clean feature flag architecture
- Build works with all feature combinations
- CI updated for MPS feature

### Day 15: Documentation & README Update

**Objective**: Document MPS integration and performance

**Update**: `README.md`
```markdown
## Performance

metal-candle v2.0 with Metal Performance Shaders achieves **MLX-competitive performance**:

| Operation | metal-candle (MPS) | MLX | Status |
|-----------|-------------------|-----|--------|
| LoRA Forward | 5-7 µs | 5.8 µs | ✅ Competitive |
| Softmax | 3-5 µs | 5.0 µs | ✅ Faster |
| RMS Norm | 4-6 µs | 5.0 µs | ✅ Competitive |

**Plus** all the benefits of Rust:
- 🦀 Type safety and memory safety
- 📦 Single binary deployment (no Python)
- ✅ Production quality (139 tests, full docs)
```

**Update**: `BENCHMARKS.md`
- Add MPS section
- Document MPS vs Custom vs MLX
- Update performance targets (achieved!)

**Create**: `MPS_INTEGRATION.md`
- Technical details of MPS integration
- API usage guide
- Troubleshooting
- Future optimizations

**Update**: `ARCHITECTURE.md`
- Document MPS layer
- Explain fallback strategy
- Performance characteristics

**Deliverables**:
- README updated with MPS performance
- BENCHMARKS.md updated
- MPS_INTEGRATION.md created
- ARCHITECTURE.md updated

---

## Technical Challenges & Solutions

### Challenge 1: Metal Buffer Sharing

**Problem**: MPS needs Metal buffers, Candle uses MetalStorage

**Solution**:
```rust
fn tensor_to_mps_matrix(tensor: &Tensor) -> Result<MPSMatrix> {
    let storage = tensor.storage_and_layout();
    let metal_storage = match &*storage.0 {
        Storage::Metal(s) => s,
        _ => bail!("Tensor must be on Metal device"),
    };
    
    let buffer = metal_storage.buffer();
    let descriptor = MPSMatrixDescriptor::new(
        tensor.dims()[0],
        tensor.dims()[1],
        buffer.length(),
        DType::F32,
    );
    
    MPSMatrix::new(buffer, &descriptor)
}
```

### Challenge 2: Tensor Layout

**Problem**: MPS expects specific layouts (row-major, contiguous)

**Solution**:
- Check `tensor.is_contiguous()`
- If not, call `tensor.contiguous()?` first
- Document layout requirements

### Challenge 3: Error Handling

**Problem**: MPS errors are Objective-C exceptions

**Solution**:
```rust
fn safe_mps_call<F, T>(f: F) -> Result<T>
where
    F: FnOnce() -> *mut Object,
{
    let result = f();
    if result.is_null() {
        bail!("MPS operation failed");
    }
    Ok(unsafe { T::from_ptr(result) })
}
```

### Challenge 4: Dimension Mismatches

**Problem**: MPS has specific dimension requirements

**Solution**:
- Validate shapes before MPS dispatch
- Clear error messages
- Fallback to custom kernels for unsupported shapes

---

## Risk Mitigation

### Risk 1: MPS Performance Underwhelming

**Likelihood**: Low (Apple's own kernels are highly optimized)

**Mitigation**:
- Early prototyping (Week 1)
- Measure actual gains before full migration
- Keep custom kernels as fallback

**Contingency**: If MPS is <3x faster, stick with custom kernels

### Risk 2: API Instability

**Likelihood**: Low (MPS is mature API)

**Mitigation**:
- Use stable MPS APIs only
- Test on multiple macOS versions
- Document minimum macOS version

### Risk 3: Correctness Issues

**Likelihood**: Medium (FFI always risky)

**Mitigation**:
- Extensive testing (property-based + edge cases)
- Cross-validate against Candle
- Fuzzing with random inputs

### Risk 4: Memory Leaks

**Likelihood**: Medium (manual retain/release in FFI)

**Mitigation**:
- Use RAII wrappers (drop = release)
- Instruments profiling
- Automated leak detection in CI

---

## Success Criteria

### Must Have ✅

1. **Performance**: 5-10x faster than custom kernels
2. **Correctness**: Max error < 1e-5 vs Candle
3. **Stability**: No crashes in 1000-iteration stress test
4. **Memory**: No leaks detected by Instruments
5. **Tests**: All existing tests pass + new MPS tests

### Should Have 🎯

1. **MLX Parity**: Within 20% of MLX performance
2. **Documentation**: Comprehensive MPS guide
3. **Examples**: Updated to use MPS
4. **Benchmarks**: MPS vs Custom vs MLX report

### Nice to Have 🌟

1. **Auto-selection**: Choose MPS vs custom based on shape
2. **Profiling**: Built-in performance monitoring
3. **Optimization**: Kernel fusion opportunities identified

---

## Timeline Summary

**Week 1: Research & Prototyping**
- Days 1-2: MPS API research
- Days 3-4: Rust FFI bindings
- Day 5: Prototype matmul

**Week 2: Core Operations**
- Days 6-7: Matrix multiplication
- Day 8: Softmax
- Day 9: RMS Norm
- Day 10: LoRA integration

**Week 3: Validation & Release**
- Day 11: Comprehensive benchmarking
- Day 12: Correctness validation
- Day 13: Memory & stability
- Days 14-15: Documentation & release

**Total**: 15 working days (~3 weeks)

---

## Deliverables Checklist

### Code
- [ ] `src/backend/mps.rs` - Core MPS module
- [ ] `src/backend/mps/matmul.rs` - MPS matrix multiplication
- [ ] `src/backend/mps/softmax.rs` - MPS softmax
- [ ] `src/backend/mps/rms_norm.rs` - MPS RMS norm
- [ ] Updated `src/training/lora.rs` - MPS LoRA integration
- [ ] `benches/mps_vs_custom.rs` - Performance benchmarks

### Tests
- [ ] 20+ MPS correctness tests
- [ ] Property-based tests
- [ ] Memory leak tests
- [ ] Integration tests

### Documentation
- [ ] `MPS_API_RESEARCH.md`
- [ ] `MPS_INTEGRATION.md`
- [ ] `MPS_BENCHMARKS.md`
- [ ] Updated `README.md`
- [ ] Updated `BENCHMARKS.md`
- [ ] Updated `ARCHITECTURE.md`

---

## Expected Outcome

### Performance (Projected)

| Operation | Before (Custom) | After (MPS) | Speedup | vs MLX |
|-----------|----------------|-------------|---------|--------|
| LoRA (512×512, r=8) | 37.0 µs | 5-7 µs | **5-7x** | **Competitive** |
| LoRA (1024×1024, r=8) | 54.8 µs | 5-8 µs | **7-10x** | **Competitive** |
| Softmax (1024) | 39.4 µs | 3-5 µs | **8-13x** | **Faster** |
| RMS Norm (1024) | 46.9 µs | 4-6 µs | **8-12x** | **Competitive** |

### Value Proposition (Updated for v2.0)

**metal-candle v2.0**: Type-safe, MLX-competitive ML for Apple Silicon

**Strengths**:
1. 🦀 **Type Safety**: Rust compile-time guarantees
2. 📦 **Single Binary**: No Python runtime required
3. ⚡ **Performance**: MLX-competitive with MPS (5-10x faster than v1.0)
4. ✅ **Quality**: Production-ready with comprehensive tests
5. 🔧 **Ergonomics**: Clean APIs, excellent error messages

**Use When**:
- ✅ Rust integration needed
- ✅ Type safety important
- ✅ Single binary deployment desired
- ✅ Performance comparable to MLX acceptable

**Don't Use When**:
- ❌ Python ecosystem strongly preferred
- ❌ Need bleeding-edge model support (MLX has more)

---

## Post-Integration Opportunities

### v2.1: Additional MPS Operations
- `MPSMatrixFindTopK` for top-k sampling
- `MPSNNSoftMax` for attention
- `MPSCNNConvolution` if needed

### v2.2: Optimization
- Automatic MPS vs custom selection based on profiling
- Kernel fusion opportunities
- Metal command buffer batching

### v2.3: Advanced Features
- Multi-GPU support via MPS
- Quantization support
- Flash Attention integration

---

## Getting Started (Week 1, Day 1)

### Immediate Next Steps

1. **Create MPS research document**:
   ```bash
   touch MPS_API_RESEARCH.md
   ```

2. **Study MPS documentation**:
   - `MPSMatrixMultiplication`
   - `MPSMatrixSoftMax`
   - `MPSNNReduceFeatureChannelsMean`

3. **Set up development environment**:
   ```bash
   # Ensure objc crate available
   cargo add objc --optional
   
   # Add MPS feature flag
   # Edit Cargo.toml
   ```

4. **Create initial module structure**:
   ```bash
   mkdir -p src/backend/mps
   touch src/backend/mps/mod.rs
   touch src/backend/mps/ffi.rs
   ```

---

**Status**: 📋 Ready to begin  
**Start Date**: December 10, 2024  
**Target Completion**: December 31, 2024  
**Expected Outcome**: MLX-competitive performance with Rust safety

Let's build the fastest type-safe ML framework for Apple Silicon! 🚀

