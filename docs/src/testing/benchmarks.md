# Benchmarks

metal-candle delivers exceptional performance on Apple Silicon, achieving **1.5-2.4x faster** LoRA operations than MLX.

## Performance Highlights 🚀

### LoRA Operations vs MLX

| Operation | metal-candle | MLX | Speedup |
|-----------|-------------|-----|---------|
| Small (512×512, r=8) | 4.92 µs | 7.33 µs | **1.49x faster** |
| Medium (1024×1024, r=8) | 3.61 µs | 5.68 µs | **1.57x faster** |
| Large (2048×2048, r=8) | 3.69 µs | 9.01 µs | **2.44x faster** |

**Overall**: 110-244% of MLX performance for LoRA training! ✅

### Metal GPU Acceleration

| Operation | Metal GPU | CPU | Speedup |
|-----------|-----------|-----|---------|
| LoRA Forward | 37-98 µs | 65-262 µs | **1.76-3.14x** |
| Softmax | 41.5 µs | 216 µs | **5.21x** |
| Layer Norm | 45.8 µs | 116 µs | **2.53x** |
| RMS Norm | 25.0 µs | 60.4 µs | **2.42x** |

## Why metal-candle is Faster

1. **Optimized Matrix Layout**: Pre-transposed matrices eliminate kernel launch overhead
2. **Zero-Cost Abstractions**: Rust's compile-time optimizations  
3. **Specialized for LoRA**: Not general-purpose, but best-in-class for our use case
4. **Direct Metal Integration**: Minimal abstraction overhead

## Use Case Recommendations

- ✅ **Best for**: LoRA training and fine-tuning (1.5-2.4x faster than MLX)
- ✅ **Good for**: Inference with LoRA adapters
- ⚠️ **Consider MLX for**: Full transformer inference without LoRA (layer ops slower)

## Running Benchmarks

```bash
# Training benchmarks
cargo bench --bench training

# Inference benchmarks
cargo bench --bench inference

# MLX comparison
cargo bench --bench mlx_comparison
```

## Complete Benchmark Results

For comprehensive benchmark data including:
- Detailed methodology and test configuration
- Training throughput metrics
- Memory usage analysis
- Profiling results
- Reproducibility guidelines
- Full MLX comparison data

See **[BENCHMARKS.md](../../BENCHMARKS.md)** in the repository root.

## References

- [BENCHMARKS.md](../../BENCHMARKS.md) - Complete benchmark documentation
- [Performance Analysis](../architecture/performance.md) - Architecture decisions
- [Candle Performance Guide](https://github.com/huggingface/candle/blob/main/docs/performance.md)

