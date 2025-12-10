# Benchmark Results Summary

## metal-candle vs MLX (Apple's Official ML Framework)

### 🏆 Winner: metal-candle (25.9x faster)

---

## Quick Comparison

### Embeddings Performance (E5-small-v2, Apple Silicon)

```
Single Query (Batch 1):
metal-candle Metal: ████ 3.9ms      ⚡ 2.0x faster than MLX
MLX:                ████████ 7.7ms

Batch Processing (Batch 100):
metal-candle Metal: ██ 4.4ms        ⚡ 25.9x faster than MLX
MLX:                ████████████████████████████████████████████████████ 113.5ms
```

---

## Full Results Table

| Batch Size | metal-candle Metal | MLX | Speedup |
|-----------|-------------------|-----|---------|
| 1         | 3.9ms             | 7.7ms | 2.0x |
| 2         | 3.1ms             | 13.6ms | 4.3x |
| 5         | 3.5ms             | 16.2ms | 4.6x |
| 10        | 3.4ms             | 22.7ms | 6.8x |
| 20        | 3.5ms             | 31.9ms | 9.2x |
| 50        | 4.0ms             | 63.2ms | 15.7x |
| **100**   | **4.4ms**         | **113.5ms** | **25.9x** 🚀 |

---

## Throughput Comparison

```
metal-candle Metal:  22,831 docs/sec  ████████████████████████████████
MLX:                    881 docs/sec  █
```

---

## Key Insight: Near Constant-Time Performance

metal-candle exhibits **remarkable scaling**:

```
Batch Size:  1    2    5    10   20   50   100
Time (ms):   3.9  3.1  3.5  3.4  3.5  4.0  4.4
             ▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▃
             Nearly constant time!
```

**Only 13% increase for 100x more data** = excellent GPU utilization

---

## Methodology

✅ Same model: `intfloat/e5-small-v2`  
✅ Same hardware: Apple Silicon (M-series)  
✅ Same operations: Tokenization + BERT + Pooling + Normalization  
✅ Same input text: 98-word technical paragraph  
✅ Fair comparison: Both using GPU acceleration  
✅ Reproducible: Scripts included in `benchmarks/`

---

## Run Benchmarks Yourself

```bash
# metal-candle
cargo run --release --example embeddings_batch --features embeddings

# MLX
source .venv/bin/activate
python benchmarks/mlx_embeddings_bench.py
```

---

## Documentation

- **[Quick Summary](../PERFORMANCE_SUMMARY.md)** - Visual charts and production recommendations
- **[Detailed Analysis](../MLX_BENCHMARK_COMPARISON.md)** - Complete methodology and technical details
- **[Verification Report](../MLX_VERIFICATION_COMPLETE.md)** - What we did and why it matters

---

**Last Updated**: December 10, 2024  
**metal-candle**: v0.1.0 (vendored BERT + custom LayerNorm)  
**MLX**: v0.30.0


