# Troubleshooting

Solutions to common problems and errors.

## Installation Issues

### Rust Version Too Old

**Error**: "requires rustc 1.75 or higher"

**Solution**:
```bash
rustup update stable
rustc --version  # Verify 1.75+
```

### Xcode Command Line Tools Missing

**Error**: Linker errors during build

**Solution**:
```bash
xcode-select --install
```

## Device Issues

### Metal Device Not Available

**Symptom**: `Device::new_metal(0)` returns error or falls back to CPU

**Causes & Solutions**:
1. **Not on Apple Silicon**: Metal only works on M1/M2/M3/M4
   ```bash
   sysctl -n macintosh.cpu.brand_string  # Check chip
   ```

2. **macOS too old**: Need macOS 12.0+
   ```bash
   sw_vers  # Check version
   ```

3. **Metal framework issue**: Restart your Mac

### CPU Fallback Unexpected

If using `Device::new_with_fallback(0)` and getting CPU:
- This is normal fallback behavior
- Check Metal availability: `Device::is_metal_available()`

## Tensor Errors

### Shape Mismatch

**Error**: "shapes are not compatible"

**Matrix multiplication example**:
```rust
// Error: (2, 3) × (4, 5) - inner dimensions don't match!
let c = a.matmul(&b)?;  // ❌

// Fix: Ensure compatible shapes
let a = Tensor::randn(0.0f32, 1.0f32, (2, 3), &device)?;  // (2, 3)
let b = Tensor::randn(0.0f32, 1.0f32, (3, 4), &device)?;  // (3, 4)
let c = a.matmul(&b)?;  // ✅ Result: (2, 4)
```

### Unexpected Striding

**Error**: "unexpected striding" on Metal

**Cause**: Tensor not contiguous after transpose/reshape

**Solution**:
```rust
// Make contiguous
let t = tensor.transpose(0, 1)?.contiguous()?;
let result = t.matmul(&other)?;  // ✅
```

### F64 on Metal

**Error**: "F64 not supported on Metal"

**Solution**: Use F32 or F16
```rust
// Error
let t = Tensor::zeros((10, 10), DType::F64, &metal_device)?;  // ❌

// Fix
let t = Tensor::zeros((10, 10), DType::F16, &metal_device)?;  // ✅
```

## Model Loading

### File Not Found

**Error**: "model file not found"

**Solutions**:
1. Check path is correct
2. Use absolute path or path relative to current directory
3. Verify file exists: `ls -lh model.safetensors`

### Invalid Format

**Error**: "invalid model format"

**Causes**:
1. File is corrupted - redownload
2. Wrong format - metal-candle v1.0 only supports safetensors
3. Incomplete download - check file size

### Incompatible Version

**Error**: "incompatible model version"

Model was created with different architecture. Ensure:
- Model is Qwen2.5-Coder architecture
- Config matches model weights

## Training Issues

### NaN Loss

**Symptom**: Loss becomes NaN during training

**Solutions**:
1. **Lower learning rate**:
   ```rust
   lr_scheduler: LRScheduler::constant(1e-5),  // Try lower LR
   ```

2. **Enable gradient clipping**:
   ```rust
   max_grad_norm: Some(1.0),
   ```

3. **Check for inf/nan in data**:
   ```rust
   assert!(!tensor.isnan().any());
   ```

### Out of Memory

**Symptom**: Kernel panic or process killed

**Solutions**:
1. **Use F16** instead of F32
2. **Reduce batch size**
3. **Use smaller model**
4. **Close other applications**
5. **Check Activity Monitor** for memory pressure

### Slow Training

**Causes & Solutions**:
1. **Using CPU**: Ensure Metal GPU is used
   ```rust
   let device = Device::new_metal(0)?;  // Not CPU!
   ```

2. **High rank**: Lower LoRA rank
   ```rust
   rank: 8,  // Instead of 32 or 64
   ```

3. **No KV-cache**: Use cache for generation
   ```rust
   model.forward_with_cache(&tokens, pos, &mut cache)?;
   ```

## Generation Issues

### Repetitive Output

**Symptom**: Model generates same text repeatedly

**Solutions**:
1. **Increase temperature**:
   ```rust
   SamplingStrategy::Temperature { temperature: 0.8 }
   ```

2. **Use top-p sampling**:
   ```rust
   SamplingStrategy::TopP { p: 0.9 }
   ```

3. **Avoid greedy**: Don't use `Greedy` for creative generation

### Nonsensical Output

**Symptom**: Generated text doesn't make sense

**Solutions**:
1. **Lower temperature**:
   ```rust
   SamplingStrategy::Temperature { temperature: 0.6 }
   ```

2. **Use top-k**:
   ```rust
   SamplingStrategy::TopK { k: 50 }
   ```

3. **Check model is loaded correctly**

### Cache Position Out of Bounds

**Error**: "cache position exceeds max_seq_len"

**Solution**: Increase cache size
```rust
let config = KVCacheConfig {
    max_seq_len: 4096,  // Increased from 2048
    ..config
};
```

## Performance Issues

### Slower Than Expected

**Check**:
1. Using Metal GPU? `device.info().device_type == DeviceType::Metal`
2. Using F16? More efficient than F32 on Metal
3. Tensors contiguous? Call `.contiguous()` after reshape/transpose
4. KV-cache enabled? For generation

### Metal GPU Not Being Used

**Diagnose**:
```rust
println!("Device: {:?}", tensor.device());
```

If showing CPU:
1. Ensure device passed to tensor creation
2. Check for device transfers in code
3. Verify Metal available: `Device::is_metal_available()`

## Build Errors

### Dependency Conflicts

**Solution**:
```bash
cargo clean
cargo update
cargo build
```

### Candle Build Fails

**Solution**:
```bash
# Clear cache
cargo clean

# Update Rust
rustup update stable

# Reinstall Command Line Tools
xcode-select --install

# Retry
cargo build
```

## Runtime Errors

### Thread Panics

**Check**:
1. `.unwrap()` calls - replace with proper error handling
2. Array indexing - use `.get()` instead
3. Check error messages in backtrace

### Segmentation Fault

**Rare** in safe Rust. Likely causes:
1. Bug in Candle/Metal backend - report issue
2. Memory corruption in unsafe code
3. System issue - restart Mac

**Debug**:
```bash
RUST_BACKTRACE=full cargo run
```

## Getting Help

### Still Stuck?

1. **Search issues**: [GitHub Issues](https://github.com/GarthDB/metal-candle/issues)
2. **Create detailed issue**:
   - Rust version (`rustc --version`)
   - macOS version (`sw_vers`)
   - Minimal code to reproduce
   - Full error message
   - Backtrace if crash

3. **Check documentation**:
   - [FAQ](./faq.md)
   - [User Guide](../guide/devices.md)
   - [API Docs](https://docs.rs/metal-candle)

### Useful Debug Commands

```bash
# System info
sw_vers
sysctl -n macintosh.cpu.brand_string

# Rust info
rustc --version
cargo --version

# Build with backtrace
RUST_BACKTRACE=1 cargo run

# Full backtrace
RUST_BACKTRACE=full cargo run

# Clean build
cargo clean && cargo build
```
