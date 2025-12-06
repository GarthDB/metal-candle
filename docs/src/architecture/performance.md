# Performance Architecture

How metal-candle leverages Metal GPU acceleration for LoRA training on Apple Silicon.

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

**Impact**: Reduces kernel launch overhead

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
- Metal GPU vs CPU comparison (1.76-3.14x faster)
- Layer operation performance
- KV-cache efficiency
- Complete performance metrics

## Optimization Roadmap

v1.1+:
- [ ] Custom Metal kernels for layer operations (softmax, layer norm)
- [ ] Flash Attention integration for transformer efficiency
- [ ] Operation fusion to reduce kernel launches
- [ ] Quantization support (4-bit, 8-bit)

v2.0:
- [ ] Enhanced Metal kernel optimizations
- [ ] Multi-GPU support
- [ ] Streaming generation improvements

## See Also

- [Benchmarks](../testing/benchmarks.md)
- [Philosophy](./philosophy.md)
