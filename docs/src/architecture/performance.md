# Performance Architecture

How metal-candle achieves 1.5-2.4x faster LoRA than MLX.

## Key Optimizations

### 1. Pre-Transposed Matrices

**Problem**: Matrix transpose requires kernel launch

**Solution**: Store matrices pre-transposed

```rust
// During initialization
self.lora_a = lora_a.transpose(0, 1)?.contiguous()?;

// During forward pass (no transpose needed!)
let result = input.matmul(&self.lora_a)?;  // Fast!
```

**Impact**: 5 kernels → 2 kernels (60% reduction)

### 2. Zero-Cost Abstractions

Rust compile-time optimization:
- No runtime type checking
- Inlined function calls
- Eliminated allocations

### 3. Specialized for LoRA

Not general-purpose:
- Optimized matrix layouts for LoRA pattern
- Pre-computed scaling factors
- Minimal abstraction overhead

### 4. Direct Metal Integration

Via Candle:
- Minimal layers between code and GPU
- Efficient memory transfers
- Unified memory advantage

## Benchmarks

See [Benchmarks](../testing/benchmarks.md) for:
- MLX comparison (1.5-2.4x faster)
- Metal vs CPU (2-5x faster)
- KV-cache performance

## Future Optimizations

v1.1+:
- [ ] Custom kernels for layer ops
- [ ] Flash Attention integration
- [ ] Operation fusion

v2.0:
- [ ] Full custom Metal implementation
- [ ] Multi-GPU support

## See Also

- [Benchmarks](../testing/benchmarks.md)
- [Philosophy](./philosophy.md)
