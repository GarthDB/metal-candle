# Benchmarks

metal-candle provides production-quality ML on Apple Silicon with Metal GPU acceleration, prioritizing type safety and ergonomic APIs.

## Performance Highlights

### Metal GPU vs CPU Acceleration

| Operation | Metal GPU | CPU | Speedup |
|-----------|-----------|-----|---------|
| Small (512×512, r=8) | 37.0 µs | 65.0 µs | **1.76x faster** |
| Medium (1024×1024, r=8) | 54.8 µs | 125.6 µs | **2.29x faster** |
| Large (2048×2048, r=8) | 98.4 µs | 262.3 µs | **2.67x faster** |

**Metal GPU provides consistent speedup** for LoRA operations on Apple Silicon.

### Metal GPU Acceleration

| Operation | Metal GPU | CPU | Speedup |
|-----------|-----------|-----|---------|
| LoRA Forward | 37-98 µs | 65-262 µs | **1.76-3.14x** |
| Softmax | 41.5 µs | 216 µs | **5.21x** |
| Layer Norm | 45.8 µs | 116 µs | **2.53x** |
| RMS Norm | 25.0 µs | 60.4 µs | **2.42x** |

## metal-candle Value Proposition

1. **Type Safety**: Rust's compile-time guarantees prevent entire classes of bugs
2. **Single Binary Deployment**: No Python runtime or dependency management
3. **Memory Safety**: No segfaults, use-after-free, or data races
4. **Production Quality**: 160 tests, zero warnings, ≥80% code coverage
5. **Metal GPU Acceleration**: 1.76-3.14x speedup over CPU for LoRA operations

## Use Case Recommendations

- ✅ **Best for**: Rust projects requiring type-safe ML integration
- ✅ **Good for**: Single-binary deployments without Python dependencies
- ✅ **Good for**: Production systems valuing compile-time safety over raw speed
- ⚠️ **Consider MLX for**: Maximum raw performance in Python environments

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

