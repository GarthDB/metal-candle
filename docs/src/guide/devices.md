# Device Management

metal-candle supports both CPU and Metal GPU devices on Apple Silicon.

## Device Types

### Metal GPU (Recommended)

Use Apple's Metal GPU for maximum performance:

```rust
use metal_candle::backend::Device;

// Create Metal device (GPU index 0)
let device = Device::new_metal(0)?;

// With automatic fallback to CPU
let device = Device::new_with_fallback(0);
```

### CPU

For testing or when GPU isn't needed:

```rust
let device = Device::new_cpu();
```

## Device Information

```rust
let device = Device::new_with_fallback(0);
let info = device.info();

println!("Device type: {:?}", info.device_type);
println!("Metal available: {}", info.metal_available);
println!("Device index: {}", info.index);
```

## Choosing the Right Device

### Use Metal GPU when:
- ✅ Training models (1.5-2.4x faster for LoRA)
- ✅ Large tensor operations
- ✅ Batch processing
- ✅ Model inference

### Use CPU when:
- ✅ Small operations (<1000 elements)
- ✅ Sampling/token selection
- ✅ Testing and debugging
- ✅ Metal unavailable

## Performance Tips

### 1. Keep Tensors on Same Device

```rust
// Good: All operations on same device
let a = Tensor::zeros((1024, 1024), DType::F32, &device)?;
let b = Tensor::ones((1024, 1024), DType::F32, &device)?;
let c = a.add(&b)?;

// Bad: Mixing devices causes slow transfers
let a = Tensor::zeros((1024, 1024), DType::F32, &gpu_device)?;
let b = Tensor::ones((1024, 1024), DType::F32, &cpu_device)?;
let c = a.add(&b)?; // Slow! Transfers b to GPU first
```

### 2. Use F16 for Metal

```rust
// Optimal for Metal GPU
let tensor = Tensor::zeros((1024, 1024), DType::F16, &metal_device)?;

// F64 not supported on Metal
// let tensor = Tensor::zeros((1024, 1024), DType::F64, &metal_device)?; // Error!
```

### 3. Batch Operations

```rust
// Better: Batch operations together
let results = model.encode(&vec![text1, text2, text3])?;

// Worse: Process individually
let r1 = model.encode(&vec![text1])?;
let r2 = model.encode(&vec![text2])?;
let r3 = model.encode(&vec![text3])?;
```

## Device-Specific Behavior

### Metal GPU
- **Fastest for**: Matrix operations, convolutions, large tensors
- **Precision**: F16, F32 (F64 not supported)
- **Memory**: Shared with system (unified memory architecture)
- **Synchronization**: Automatic, but can add overhead for tiny operations

### CPU
- **Fastest for**: Small operations, sampling, simple logic
- **Precision**: All types (F16, F32, F64, etc.)
- **Memory**: Standard system RAM
- **Parallel**: Uses Rayon for multi-threading

## Example: Mixed CPU/GPU Workflow

```rust
use metal_candle::{ModelLoader, inference::sample_token};
use candle_core::{DType, Device};

// Use Metal GPU for model forward pass
let gpu = Device::new_metal(0)?;
let model = ModelLoader::new()
    .with_device(gpu.clone())
    .with_dtype(DType::F16)
    .load("model.safetensors")?;

// Forward pass on GPU (fast!)
let logits = model.forward(&input_ids)?;

// Use CPU for sampling (more efficient for small ops)
let cpu = Device::Cpu;
let logits_cpu = logits.to_device(&cpu)?;
let token = sample_token(&logits_cpu, &strategy)?;
```

## Troubleshooting

### "Metal device not available"

Check Metal is supported:
```rust
use metal_candle::backend::Device;

if Device::is_metal_available() {
    println!("✅ Metal GPU available");
} else {
    println!("❌ Metal GPU not available - using CPU");
}
```

**Causes:**
- Not on Apple Silicon Mac
- macOS < 12.0
- Metal framework not installed

### "Unexpected striding" errors

Ensure tensors are contiguous after transpose:
```rust
// After transpose, make contiguous
let transposed = tensor.transpose(0, 1)?.contiguous()?;
let result = transposed.matmul(&other)?;
```

## See Also

- [Tensor Operations](./tensors.md)
- [Performance Guide](../architecture/performance.md)
- [Benchmarks](../testing/benchmarks.md)
