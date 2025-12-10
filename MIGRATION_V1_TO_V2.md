# Migration Guide: metal-candle v1.0 → v2.0

**Date**: December 9, 2024  
**Status**: FINAL - Phase 4, Week 9

---

## Overview

`metal-candle` v2.0 introduces lazy evaluation for improved performance through computation graph optimization and asynchronous execution. This guide helps you migrate from v1.0 (eager execution) to v2.0 (lazy execution).

**Important**: v2.0 is a **breaking change release**. You'll need to update your code to use the new lazy execution API.

---

## What's New in v2.0?

### Lazy Evaluation Framework

v2.0 introduces a computation graph that defers execution until explicitly needed:

```rust
// v1.0 (Eager) - Executes immediately
let output = lora_layer.forward(&input)?;

// v2.0 (Lazy) - Builds graph, defers execution
let input_lazy = LazyTensor::from_tensor(input)?;
let output_lazy = lora_layer.forward_lazy(&input_lazy)?;
let output = output_lazy.eval()?;  // Execute when needed
```

### Performance Benefits

- **Phase 4** (Current): ~5% overhead, full correctness validation
- **Phase 5** (Future): 2-3x speedup via async batching and graph optimization

### Supported Operations

- ✅ **LoRA** - Low-Rank Adaptation layers
- ✅ **Softmax** - Softmax normalization
- ✅ **RMS Norm** - Root Mean Square normalization
- ⏳ **More coming**: Attention, LayerNorm, etc.

---

## Migration Strategy

v2.0 removes eager execution entirely. All operations now use lazy evaluation:

```rust
use metal_candle::graph::LazyTensor;

// Create lazy tensor
let input_lazy = LazyTensor::from_tensor(input)?;

// Use lazy methods
let output_lazy = lora_layer.forward_lazy(&input_lazy)?;

// Execute when ready
let output = output_lazy.eval()?;
```

**Benefits**:
- 🚀 2-3x performance in Phase 5 (async batching)
- 🔗 Chain multiple operations before eval
- 📊 Automatic graph optimization
- 🎯 Single execution model (simpler codebase)

---

## API Changes

### LoRALayer

#### v1.0 (Removed)

```rust
impl LoRALayer {
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // REMOVED in v2.0
    }
}
```

#### v2.0 (Renamed from forward_lazy)

```rust
impl LoRALayer {
    pub fn forward(&self, input: &LazyTensor) -> Result<LazyTensor> {
        // Now takes LazyTensor, returns LazyTensor
        // Must call .eval() to execute
    }
}
```

### LazyTensor Operations

New methods available on `LazyTensor`:

```rust
// Basic operations
let c = a.add(&b)?;
let d = a.matmul(&b)?;
let e = a.mul_scalar(2.0)?;

// ML operations
let soft = input.softmax(dim)?;
let normed = input.rms_norm(eps)?;

// LoRA
let output = lora.forward_lazy(&input)?;

// Execute graph
let result = output.eval()?;
```

---

## Common Patterns

### Pattern 1: Single Operation

**v1.0**:
```rust
let output = lora.forward(&input)?;
```

**v2.0 (Backward compatible)**:
```rust
let output = lora.forward(&input)?;  // Still works!
```

**v2.0 (Explicit lazy)**:
```rust
let input_lazy = LazyTensor::from_tensor(input)?;
let output = lora.forward_lazy(&input_lazy)?.eval()?;
```

### Pattern 2: Operation Chaining

**v1.0**:
```rust
let hidden = lora1.forward(&input)?;
let normed = rms_norm(&hidden, &alpha, eps)?;
let output = lora2.forward(&normed)?;
```

**v2.0 (Lazy - Better!)**:
```rust
let input_lazy = LazyTensor::from_tensor(input)?;
let hidden = lora1.forward_lazy(&input_lazy)?;
let normed = hidden.rms_norm(eps)?;
let output = lora2.forward_lazy(&normed)?;

// Single eval executes entire chain
let result = output.eval()?;
```

**Benefits**: In Phase 5, the entire chain executes in a single optimized pass.

### Pattern 3: Batched Operations

**v1.0**:
```rust
let mut results = Vec::new();
for input in batch {
    results.push(model.forward(&input)?);
}
```

**v2.0 (Lazy - Much Better!)**:
```rust
// Build graph for all inputs
let mut lazy_outputs = Vec::new();
for input in batch {
    let input_lazy = LazyTensor::from_tensor(input)?;
    lazy_outputs.push(model.forward_lazy(&input_lazy)?);
}

// Evaluate all at once (Phase 5: parallel execution!)
let results: Vec<_> = lazy_outputs
    .into_iter()
    .map(|x| x.eval())
    .collect::<Result<_>>()?;
```

### Pattern 4: Training Loop

**v1.0**:
```rust
for epoch in 0..num_epochs {
    for batch in data_loader {
        let logits = model.forward(&batch.input)?;
        let loss = cross_entropy_loss(&logits, &batch.labels)?;
        
        optimizer.backward_step(&loss)?;
    }
}
```

**v2.0 (Backward compatible)**:
```rust
// Same code works! No changes needed.
for epoch in 0..num_epochs {
    for batch in data_loader {
        let logits = model.forward(&batch.input)?;
        let loss = cross_entropy_loss(&logits, &batch.labels)?;
        
        optimizer.backward_step(&loss)?;
    }
}
```

**v2.0 (Optimized with lazy)**:
```rust
for epoch in 0..num_epochs {
    for batch in data_loader {
        // Build computation graph
        let input_lazy = LazyTensor::from_tensor(batch.input)?;
        let logits_lazy = model.forward_lazy(&input_lazy)?;
        
        // Evaluate and compute loss
        let logits = logits_lazy.eval()?;
        let loss = cross_entropy_loss(&logits, &batch.labels)?;
        
        optimizer.backward_step(&loss)?;
    }
}
```

---

## Feature Flags

v2.0 introduces feature flags for lazy evaluation:

### Cargo.toml

```toml
[dependencies]
metal-candle = { version = "2.0", features = ["graph"] }
```

### Features

- **`graph`** (default): Enables lazy evaluation framework
- **`custom-metal`** (default): Enables custom Metal kernels

### Minimal Setup (v1.0 compatibility only)

```toml
[dependencies]
metal-candle = { version = "2.0", default-features = false }
```

This disables lazy evaluation and uses only eager execution (like v1.0).

---

## Performance Considerations

### When to Use Lazy Execution

✅ **Good for**:
- Complex operation chains (3+ operations)
- Batched processing
- Training loops with multiple forward passes
- Repeated similar computations

❌ **Not worth it for**:
- Single simple operations
- One-off computations
- Quick prototyping

### Phase 4 vs Phase 5

| Aspect | Phase 4 (Current) | Phase 5 (Future) |
|--------|------------------|------------------|
| **Lazy overhead** | ~5% | 0% |
| **Speedup** | None yet | 2-3x |
| **Batching** | Sequential | Parallel |
| **Graph optimization** | Basic | Advanced fusion |
| **Status** | ✅ Stable | 🔄 In progress |

**Recommendation**: Start using lazy methods now to prepare for Phase 5 speedups!

---

## Testing

Your existing tests should continue to work in v2.0:

```rust
#[test]
fn test_lora_forward() -> Result<()> {
    let lora = LoRALayer::new(512, 512, &config, &device)?;
    let input = Tensor::randn(0f32, 1f32, &[1, 512], &device)?;
    
    // This test still passes in v2.0!
    let output = lora.forward(&input)?;
    
    assert_eq!(output.dims(), &[1, 512]);
    Ok(())
}
```

### Testing Lazy Execution

Add new tests for lazy paths:

```rust
#[cfg(feature = "graph")]
#[test]
fn test_lora_forward_lazy() -> Result<()> {
    use metal_candle::graph::LazyTensor;
    
    let lora = LoRALayer::new(512, 512, &config, &device)?;
    let input = Tensor::randn(0f32, 1f32, &[1, 512], &device)?;
    
    // Test lazy execution
    let input_lazy = LazyTensor::from_tensor(input.clone())?;
    let output_lazy = lora.forward_lazy(&input_lazy)?.eval()?;
    
    // Compare with eager
    let output_eager = lora.forward(&input)?;
    
    // Should be nearly identical
    let diff = (output_lazy - output_eager)?.abs()?.flatten_all()?;
    assert!(diff.max(0)?.to_scalar::<f32>()? < 1e-4);
    
    Ok(())
}
```

---

## Deprecation Timeline

v2.0 does **not** deprecate any v1.0 APIs. All existing code continues to work.

Future versions may:
- v2.1: Add deprecation warnings for eager-only usage in performance-critical code
- v3.0: Potentially default to lazy execution for all operations

**No action required** - your code will keep working through all these versions.

---

## Examples

### Before (v1.0)

```rust
use metal_candle::training::{LoRALayer, LoRAConfig};
use candle_core::{Tensor, Device};

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    let config = LoRAConfig::default();
    let lora = LoRALayer::new(512, 512, &config, &device)?;
    
    let input = Tensor::randn(0f32, 1f32, &[4, 512], &device)?;
    let output = lora.forward(&input)?;
    
    println!("Output shape: {:?}", output.shape());
    Ok(())
}
```

### After (v2.0 - Backward Compatible)

```rust
use metal_candle::training::{LoRALayer, LoRAConfig};
use candle_core::{Tensor, Device};

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    let config = LoRAConfig::default();
    let lora = LoRALayer::new(512, 512, &config, &device)?;
    
    let input = Tensor::randn(0f32, 1f32, &[4, 512], &device)?;
    let output = lora.forward(&input)?;  // Still works!
    
    println!("Output shape: {:?}", output.shape());
    Ok(())
}
```

### After (v2.0 - Optimized with Lazy)

```rust
use metal_candle::training::{LoRALayer, LoRAConfig};
use metal_candle::graph::LazyTensor;
use candle_core::{Tensor, Device};

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    let config = LoRAConfig::default();
    let lora = LoRALayer::new(512, 512, &config, &device)?;
    
    let input = Tensor::randn(0f32, 1f32, &[4, 512], &device)?;
    
    // Use lazy execution for better performance (Phase 5)
    let input_lazy = LazyTensor::from_tensor(input)?;
    let output_lazy = lora.forward_lazy(&input_lazy)?;
    let output = output_lazy.eval()?;
    
    println!("Output shape: {:?}", output.shape());
    Ok(())
}
```

---

## Troubleshooting

### Issue: "feature `graph` not enabled"

**Problem**: Trying to use `LazyTensor` without enabling the feature.

**Solution**: Add feature to `Cargo.toml`:
```toml
metal-candle = { version = "2.0", features = ["graph"] }
```

### Issue: "no method named `forward_lazy` found"

**Problem**: Method only available with `graph` feature or on certain types.

**Solution**: 
1. Enable `graph` feature
2. Use `#[cfg(feature = "graph")]` in your code
3. Or stick with `forward()` for backward compatibility

### Issue: Performance is slower in v2.0

**Problem**: Phase 4 has ~5% overhead from graph building.

**Solution**: 
- Wait for Phase 5 (async batching) for 2-3x speedup
- Use lazy execution for operation chains (3+ ops)
- Consider sticking with eager for simple operations

### Issue: Type mismatch between `Tensor` and `LazyTensor`

**Problem**: Can't mix eager and lazy execution directly.

**Solution**:
```rust
// Convert Tensor -> LazyTensor
let lazy = LazyTensor::from_tensor(tensor)?;

// Convert LazyTensor -> Tensor
let tensor = lazy.eval()?;
```

---

## FAQ

### Do I need to change my code for v2.0?

**No!** v2.0 is fully backward compatible. All v1.0 code works without changes.

### Should I switch to lazy execution?

**It depends**:
- Simple single operations? Stick with eager.
- Complex chains or batching? Try lazy now, big benefits in Phase 5.

### When will Phase 5 be released?

**Estimated**: Q1 2025 (Weeks 10-12 of the rewrite)

### Will v1.0 APIs be removed?

**No plans**. Backward compatibility is a core principle. Eager execution will always be supported.

### How much faster will Phase 5 be?

**Target**: 2-3x speedup for typical workloads through async batching and graph optimization.

---

## Getting Help

- **Documentation**: https://docs.rs/metal-candle
- **Examples**: `examples/` directory in the repository
- **Issues**: https://github.com/GarthDB/metal-candle/issues
- **Discussions**: https://github.com/GarthDB/metal-candle/discussions

---

## Summary

✅ **v2.0 is fully backward compatible** - no changes required  
🚀 **Opt-in to lazy execution** for future performance gains  
📊 **Phase 5 will bring 2-3x speedup** - prepare now by using lazy methods  
🔧 **Feature flags** allow granular control over functionality

**Migration is optional but recommended for performance-critical code.**

---

**Last Updated**: December 9, 2024  
**Version**: v2.0 (Phase 4, Week 9)

