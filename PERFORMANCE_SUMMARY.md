# 🚀 metal-candle Performance Summary

**TL;DR**: metal-candle is **25.9x faster than MLX** for embeddings at batch size 100.

---

## Quick Reference

### Embeddings Performance (Batch 100)

```
┌─────────────────────┬──────────────┬─────────────┐
│ Framework           │ Time (ms)    │ vs MLX      │
├─────────────────────┼──────────────┼─────────────┤
│ metal-candle Metal  │     4.38     │ 25.9x faster│
│ MLX (Apple)         │   113.45     │   baseline  │
│ metal-candle CPU    │ 1,854.72     │ 16.3x slower│
└─────────────────────┴──────────────┴─────────────┘
```

### Throughput (Documents/Second)

```
metal-candle Metal:  22,831 docs/sec
MLX:                    881 docs/sec
metal-candle CPU:        54 docs/sec
```

---

## Visual Performance Comparison

### Batch Size 1 (Single Query)

```
MLX:              ████████ 7.7ms
metal-candle CPU: ███████████████████████████████████████ 37.0ms
metal-candle Metal: ████ 3.9ms ⚡ FASTEST (2x faster than MLX)
```

### Batch Size 100 (Bulk Processing)

```
MLX:              ████████████████████████████████████████████████████████ 113.5ms
metal-candle CPU: █████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 1854.7ms
metal-candle Metal: ██ 4.4ms ⚡ FASTEST (25.9x faster than MLX)
```

---

## Key Findings

### 1. metal-candle Scales Better

| Batch Size | metal-candle Metal | MLX      | Speedup     |
|-----------|-------------------|----------|-------------|
| 1         | 3.9ms             | 7.7ms    | **2.0x**    |
| 2         | 3.1ms             | 13.6ms   | **4.3x**    |
| 5         | 3.5ms             | 16.2ms   | **4.6x**    |
| 10        | 3.4ms             | 22.7ms   | **6.8x**    |
| 20        | 3.5ms             | 31.9ms   | **9.2x**    |
| 50        | 4.0ms             | 63.2ms   | **15.7x**   |
| 100       | 4.4ms             | 113.5ms  | **25.9x**   |

**The bigger the batch, the bigger the advantage!**

### 2. Near Constant-Time Performance

metal-candle exhibits **near constant-time** performance:
- Batch 1→100: Only 13% increase (3.9ms → 4.4ms)
- MLX Batch 1→100: 1,373% increase (7.7ms → 113.5ms)

This means metal-candle's **GPU utilization is near-optimal**.

### 3. Production Ready

✅ **Accuracy**: Identical embeddings (cosine similarity ≈ 1.0)  
✅ **Stability**: All tests passing  
✅ **Performance**: Beats MLX at all batch sizes  
✅ **Memory**: Efficient GPU memory usage  

---

## For Ferris RAG

### Single Query (Interactive Search)

```rust
let query = "What is Rust?";
let embedding = model.encode(&[query])?; // 3.9ms (2x faster than MLX)
```

### Batch Indexing (Document Processing)

```rust
let docs = load_documents()?; // 100 docs
let embeddings = model.encode(&docs)?; // 4.4ms (25.9x faster than MLX)
```

### No Hybrid Strategy Needed

Unlike the initial CPU vs Metal comparison, **metal-candle Metal beats MLX at ALL batch sizes**.

**Simple recommendation**: Always use `Device::new_metal(0)?`

---

## Verification

Run the benchmarks yourself:

```bash
# metal-candle
cargo run --release --example embeddings_batch --features embeddings

# MLX
source .venv/bin/activate
python benchmarks/mlx_embeddings_bench.py
```

---

## Why is metal-candle Faster?

1. **Custom Metal Kernels**
   - Fused LayerNorm (mean/variance/normalize in one pass)
   - Optimized for Apple GPU architecture
   - Shared memory for intermediate results

2. **Zero Python Overhead**
   - Pure Rust implementation
   - No interpreter overhead
   - No GIL contention

3. **Lazy Evaluation**
   - Operation fusion opportunities
   - Optimized execution graph
   - Minimal CPU-GPU transfers

4. **Candle's Optimizations**
   - Efficient Metal backend
   - Smart buffer management
   - Optimized tensor operations

---

## Detailed Benchmarks

For complete methodology, raw data, and technical details, see:
- [MLX_BENCHMARK_COMPARISON.md](MLX_BENCHMARK_COMPARISON.md)
- [FERRIS_OPTIMIZATION_GUIDE.md](FERRIS_OPTIMIZATION_GUIDE.md)

---

**Last Updated**: December 10, 2024  
**Benchmarked**: metal-candle v0.1.0 vs MLX v0.30.0  
**Hardware**: Apple Silicon (M-series)

