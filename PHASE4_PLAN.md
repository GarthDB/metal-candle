# Phase 4: Operation Migration (Weeks 7-9)

**Date**: December 9, 2024  
**Status**: STARTING - Migrating LoRALayer to lazy execution  
**Goal**: Integrate lazy evaluation into existing metal-candle operations

---

## Overview

Phase 4 focuses on migrating existing operations to use the new lazy evaluation framework. This phase bridges v1.0 (eager) and v2.0 (lazy) by:

1. Adding lazy execution paths to existing operations
2. Maintaining backward compatibility with v1.0 API
3. Validating correctness and measuring performance baseline

---

## Week 7: Migrate LoRALayer

### Goals

1. Add lazy execution support to `LoRALayer`
2. Create `LoRAOperation` for the graph
3. Benchmark lazy vs eager for LoRA
4. Validate correctness

### Implementation Strategy

#### 1. Add LazyTensor Support to LoRALayer

```rust
// In src/training/lora.rs

impl LoRALayer {
    /// Forward pass using lazy evaluation
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::graph::LazyTensor;
    /// 
    /// let input_lazy = LazyTensor::from_tensor(input)?;
    /// let output_lazy = lora_layer.forward_lazy(&input_lazy)?;
    /// let output = output_lazy.eval()?;  // Deferred execution
    /// ```
    pub fn forward_lazy(&self, input: &LazyTensor) -> Result<LazyTensor> {
        // Use fused LoRA operation if available
        #[cfg(feature = "custom-metal")]
        {
            if input.device().is_metal() {
                return input.lora_fused(
                    &LazyTensor::from_tensor(self.lora_a.as_tensor().clone())?,
                    &LazyTensor::from_tensor(self.lora_b.as_tensor().clone())?,
                    self.config.scaling(),
                );
            }
        }
        
        // Fallback to individual operations
        let hidden = input.matmul(&LazyTensor::from_tensor(self.lora_a.as_tensor().clone())?)?;
        let output = hidden.matmul(&LazyTensor::from_tensor(self.lora_b.as_tensor().clone())?)?;
        output.mul_scalar(self.config.scaling())
    }
    
    // Keep existing forward() for backward compatibility
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Existing eager implementation unchanged
        // ...
    }
}
```

#### 2. Add Fused LoRA Operation to LazyTensor

```rust
// In src/graph/lazy_tensor.rs

impl LazyTensor {
    /// Fused LoRA operation: (input @ A @ B) * scale
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    #[cfg(feature = "custom-metal")]
    pub fn lora_fused(
        &self,
        lora_a: &LazyTensor,
        lora_b: &LazyTensor,
        scale: f32,
    ) -> CandleResult<Self> {
        self.add_operation(
            Operation::LoRA {
                a: lora_a.node_id,
                b: lora_b.node_id,
                scale,
            },
            vec![self.node_id],
        )
    }
}
```

#### 3. Update Executor to Handle LoRA Operation

```rust
// In src/graph/executor.rs

#[cfg(feature = "custom-metal")]
Operation::LoRA { a, b, scale } => {
    if inputs.len() != 3 {
        return Err(TrainingError::Failed {
            reason: format!("LoRA requires 3 inputs (input, a, b), got {}", inputs.len()),
        });
    }
    
    // Use custom fused LoRA kernel
    use crate::backend::CustomMetalOps;
    if inputs[0].device().is_metal() {
        if let Ok(output) = inputs[0].lora_forward_fused(
            &inputs[1],  // lora_a
            &inputs[2],  // lora_b
            *scale,
        ) {
            return Ok(output);
        }
    }
    
    // Fallback to sequential operations
    let hidden = inputs[0].broadcast_matmul(&inputs[1])?;
    let output = hidden.broadcast_matmul(&inputs[2])?;
    output.affine(f64::from(*scale), 0.0).map_err(|e| TrainingError::Failed {
        reason: format!("LoRA fallback failed: {e}"),
    })
}
```

### Testing

```rust
// tests/graph/lora_lazy.rs

#[test]
fn test_lora_lazy_vs_eager_correctness() -> Result<()> {
    let device = Device::Cpu;
    
    // Create LoRA layer
    let config = LoRAConfig::new(512, 8);
    let lora = LoRALayer::new(config, &device)?;
    
    // Test input
    let input = Tensor::randn(&[128, 512], DType::F32, &device)?;
    
    // Eager execution
    let eager_output = lora.forward(&input)?;
    
    // Lazy execution
    let input_lazy = LazyTensor::from_tensor(input.clone())?;
    let lazy_output = lora.forward_lazy(&input_lazy)?.eval()?;
    
    // Validate correctness
    let max_diff = (eager_output - lazy_output)?.abs()?.max(0)?;
    assert!(max_diff.to_scalar::<f32>()? < 1e-5);
    
    Ok(())
}

#[test]
fn test_lora_lazy_batched_operations() -> Result<()> {
    // Test that multiple LoRA operations can be batched
    let device = Device::Cpu;
    let config = LoRAConfig::new(512, 8);
    let lora1 = LoRALayer::new(config.clone(), &device)?;
    let lora2 = LoRALayer::new(config, &device)?;
    
    let input = LazyTensor::from_slice(&[/* ... */], &[128, 512], &device)?;
    
    // Chain multiple LoRA operations (should batch in graph)
    let output = lora1.forward_lazy(&input)?;
    let output = lora2.forward_lazy(&output)?;
    
    // Single eval executes both
    let result = output.eval()?;
    
    assert_eq!(result.shape().dims(), &[128, 512]);
    Ok(())
}
```

### Benchmarking

```rust
// benches/lora_lazy_vs_eager.rs

fn benchmark_lora_forward(c: &mut Criterion) {
    let device = Device::cuda_if_available(0).unwrap();
    let config = LoRAConfig::new(512, 8);
    let lora = LoRALayer::new(config, &device).unwrap();
    let input = Tensor::randn(&[128, 512], DType::F32, &device).unwrap();
    
    let mut group = c.benchmark_group("lora_forward");
    
    group.bench_function("eager", |b| {
        b.iter(|| {
            let output = lora.forward(&input).unwrap();
            black_box(output);
        });
    });
    
    let input_lazy = LazyTensor::from_tensor(input.clone()).unwrap();
    group.bench_function("lazy", |b| {
        b.iter(|| {
            let output = lora.forward_lazy(&input_lazy).unwrap().eval().unwrap();
            black_box(output);
        });
    });
    
    group.finish();
}
```

---

## Week 8: Migrate Softmax and RMS Norm

### Goals

1. Add lazy execution support to Softmax
2. Add lazy execution support to RMS Norm
3. Update custom Metal kernel integration
4. Comprehensive testing

### Implementation

Similar pattern to LoRA:

```rust
// Already have these in LazyTensor:
pub fn softmax(&self, dim: usize) -> CandleResult<Self>
pub fn rms_norm(&self, eps: f32) -> CandleResult<Self>

// Just need to integrate into training code
```

---

## Week 9: Backward Compatibility Layer

### Goals

1. Ensure v1.0 API continues to work
2. Add deprecation warnings for eager-only usage
3. Create migration guide
4. Update examples

### Strategy

#### Option A: Transparent Migration (Recommended)

Keep existing `forward()` methods, internally use lazy evaluation:

```rust
impl LoRALayer {
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Internally convert to lazy and eval immediately
        let input_lazy = LazyTensor::from_tensor(input.clone())?;
        let output_lazy = self.forward_lazy(&input_lazy)?;
        output_lazy.eval()
    }
}
```

**Pros**: Zero breaking changes, smooth transition  
**Cons**: Slight overhead for converting to/from lazy

#### Option B: Explicit Migration

Add new `*_lazy()` methods, deprecate old ones:

```rust
impl LoRALayer {
    #[deprecated(since = "2.0.0", note = "Use forward_lazy() for better performance")]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Old implementation
    }
    
    pub fn forward_lazy(&self, input: &LazyTensor) -> Result<LazyTensor> {
        // New lazy implementation
    }
}
```

**Pros**: Clear migration path, explicit control  
**Cons**: Breaking changes, requires code updates

**Decision**: Use **Option A** for v2.0 beta, transition to **Option B** for v2.1+

---

## Success Metrics

- [ ] LoRALayer works with LazyTensor
- [ ] Correctness validated (lazy == eager within 1e-5)
- [ ] Softmax and RMS Norm migrated
- [ ] Backward compatibility maintained
- [ ] All existing tests pass
- [ ] New lazy execution tests pass
- [ ] Performance baseline established

---

## Expected Performance (Phase 4)

Phase 4 is still **synchronous execution** (no async batching yet). Performance will be similar to v1.0:

| Operation | v1.0 Eager | Phase 4 Lazy | Difference |
|-----------|-----------|-------------|------------|
| LoRA | 36 µs | ~37-40 µs | +1-4 µs (graph overhead) |
| Softmax | 39 µs | ~40-43 µs | +1-4 µs (graph overhead) |
| RMS Norm | 47 µs | ~48-51 µs | +1-4 µs (graph overhead) |

**Note**: Phase 5 (async + batching) will bring 2-3x improvements.

---

## Next Steps

1. Start with LoRALayer migration
2. Add comprehensive tests
3. Validate correctness
4. Move to Softmax/RMS Norm
5. Implement backward compatibility

This phase sets the foundation for Phase 5's performance optimizations.

