# Next Steps - Quick Reference Guide

## Current Status
✅ `FusedLoRAOp` fully implemented and compiling  
⏭️ Ready for integration and testing

## Immediate Next Actions (2-3 hours)

### 1. Update `src/training/lora.rs` (15 min)

Find the `forward_metal_fused` method and replace with:

```rust
#[cfg(feature = "custom-metal")]
fn forward_metal_fused(&self, input: &Tensor) -> Result<Tensor> {
    use crate::backend::custom_ops::FusedLoRAOp;
    
    let op = FusedLoRAOp::new(
        self.lora_a.as_tensor().clone(),
        self.lora_b.as_tensor().clone(),
        self.config.scaling(),
    )?;
    
    input.apply_op1(op)
}
```

Also update the `forward()` method to call this:

```rust
pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "custom-metal")]
    {
        // Try fused kernel if on Metal
        if input.device().is_metal() {
            if let Ok(output) = self.forward_metal_fused(input) {
                return Ok(output);
            }
            // Fall through to Candle implementation on error
        }
    }
    
    // Existing Candle implementation
    let hidden = input.broadcast_matmul(self.lora_a.as_tensor())?;
    let output = hidden.broadcast_matmul(self.lora_b.as_tensor())?;
    Ok(output.affine(f64::from(self.config.scaling()), 0.0)?)
}
```

### 2. Complete Metal Shader (30 min)

Edit `src/backend/kernels.metal` - replace placeholder with actual implementation:

```metal
kernel void fused_lora_forward(
    const device float* input [[buffer(0)]],
    const device float* lora_a [[buffer(1)]],
    const device float* lora_b [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant FusedLoraParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch = gid.x;
    uint seq = gid.y;
    uint out_feature = gid.z;
    
    if (batch >= params.batch_size || seq >= params.seq_len || out_feature >= params.out_features) {
        return;
    }
    
    // Compute: output[batch, seq, out_feature] = sum_k sum_r (input[batch, seq, in_feature] * lora_a[in_feature, r] * lora_b[r, out_feature]) * scaling
    
    float sum = 0.0;
    
    // For each rank dimension
    for (uint r = 0; r < params.rank; r++) {
        float hidden_sum = 0.0;
        
        // For each input feature
        for (uint in_feature = 0; in_feature < params.in_features; in_feature++) {
            uint input_idx = batch * params.seq_len * params.in_features + seq * params.in_features + in_feature;
            uint lora_a_idx = in_feature * params.rank + r;
            
            hidden_sum += input[input_idx] * lora_a[lora_a_idx];
        }
        
        // Multiply by lora_b
        uint lora_b_idx = r * params.out_features + out_feature;
        sum += hidden_sum * lora_b[lora_b_idx];
    }
    
    // Apply scaling and store result
    uint output_idx = batch * params.seq_len * params.out_features + seq * params.out_features + out_feature;
    output[output_idx] = sum * params.scaling;
}
```

### 3. Write Correctness Test (30 min)

Create `tests/custom_ops_correctness.rs`:

```rust
#[cfg(all(test, feature = "custom-metal"))]
mod tests {
    use candle_core::{Device, DType, Tensor};
    use metal_candle::backend::custom_ops::FusedLoRAOp;
    use metal_candle::training::LoRAConfig;

    #[test]
    fn test_fused_lora_correctness() {
        let device = Device::new_metal(0).expect("Metal device required");
        
        // Create test tensors
        let input = Tensor::randn(0.0f32, 1.0f32, (2, 64, 512), &device).unwrap();
        let lora_a = Tensor::randn(0.0f32, 0.01f32, (512, 8), &device).unwrap();
        let lora_b = Tensor::randn(0.0f32, 0.01f32, (8, 512), &device).unwrap();
        let scaling = 2.0f32;
        
        // Compute with fused kernel
        let op = FusedLoRAOp::new(lora_a.clone(), lora_b.clone(), scaling).unwrap();
        let fused_output = input.apply_op1(op).unwrap();
        
        // Compute with standard Candle (reference)
        let hidden = input.matmul(&lora_a).unwrap();
        let candle_output = hidden.matmul(&lora_b).unwrap();
        let candle_output = candle_output.affine(scaling as f64, 0.0).unwrap();
        
        // Compare
        let diff = (&fused_output - &candle_output).unwrap();
        let max_diff = diff.abs().unwrap()
            .flatten_all().unwrap()
            .max(0).unwrap()
            .to_scalar::<f32>().unwrap();
        
        println!("Max difference: {:.2e}", max_diff);
        assert!(max_diff < 1e-4, "Max difference too large: {:.2e}", max_diff);
    }
}
```

### 4. Run Tests (15 min)

```bash
cargo test --features custom-metal test_fused_lora_correctness -- --nocapture
```

### 5. Add Performance Benchmark (30 min)

Update `benches/mlx_comparison.rs` to include fused kernel:

```rust
fn bench_lora_fused(c: &mut Criterion) {
    let device = Device::new_metal(0).unwrap();
    let input = Tensor::randn(0.0f32, 1.0f32, (1, 128, 512), &device).unwrap();
    let lora_a = Tensor::randn(0.0f32, 0.01f32, (512, 8), &device).unwrap();
    let lora_b = Tensor::randn(0.0f32, 0.01f32, (8, 512), &device).unwrap();
    let op = FusedLoRAOp::new(lora_a, lora_b, 2.0).unwrap();
    
    c.bench_function("lora_fused_forward", |b| {
        b.iter(|| input.apply_op1(&op).unwrap())
    });
}

criterion_group!(benches, bench_lora_fused);
criterion_main!(benches);
```

### 6. Run Benchmark (30 min)

```bash
cargo bench --bench mlx_comparison -- lora_fused_forward
```

Compare against baseline:
- Current unfused: 37-98 µs
- MLX baseline: 5-11 µs
- **Target fused: 6-15 µs (6-10x speedup)**

## Success Criteria

✅ Test passes with max_diff < 1e-4  
✅ Benchmark shows 6-10x speedup  
✅ Performance ≥95% of MLX (5-11 µs)

## If Something Goes Wrong

### Compilation Error in Metal Shader
- Check parameter struct matches Rust `LoRAParams`
- Verify all buffer indices are correct
- Use `#include <metal_stdlib>` at top

### Numerical Accuracy Issues
- Check matrix indexing in shader
- Verify transpose/layout assumptions
- Add debug prints to compare intermediate values

### Performance Not Meeting Target
- Profile with `cargo instruments -t Time`
- Check grid/threadgroup dimensions
- Verify kernel is actually being called (not falling back)

## Files to Edit

1. `src/training/lora.rs` - Integration
2. `src/backend/kernels.metal` - Shader implementation
3. `tests/custom_ops_correctness.rs` - NEW file
4. `benches/mlx_comparison.rs` - Benchmark

## Quick Test Commands

```bash
# Compile check
cargo build --release --features custom-metal

# Run test
cargo test --features custom-metal test_fused_lora_correctness -- --nocapture

# Run benchmark
cargo bench --bench mlx_comparison

# Profile
cargo instruments -t Time --release --example train_lora
```

## Expected Results

After completing these steps:
- ✅ Correctness test passes
- ✅ 6-10x speedup measured
- ✅ LoRA forward: 37-98 µs → 6-15 µs
- ✅ Comparable to MLX: 5-11 µs

**Total time**: 2-3 hours to working, tested prototype

---

**Current files ready**: `src/backend/custom_ops.rs` (DONE ✅)  
**Next focus**: Integration → Testing → Benchmarking

