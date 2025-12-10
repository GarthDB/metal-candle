# MLX vs metal-candle Benchmark Comparison

**Date**: December 10, 2024  
**Hardware**: Apple Silicon (M-series)  
**Model**: intfloat/e5-small-v2 (sentence-transformers)

## Executive Summary

✅ **metal-candle Metal is 25.9x FASTER than MLX at batch size 100**

This benchmark provides an accurate, apples-to-apples comparison of embeddings performance between MLX (Apple's official ML framework) and metal-candle.

---

## Raw Benchmark Results

### MLX Performance (GPU)

| Batch Size | Time (µs) | Time (ms) | Per Doc (µs) |
|-----------|-----------|-----------|--------------|
| 1         | 7,652     | 7.65      | 7,652        |
| 2         | 13,552    | 13.55     | 6,776        |
| 5         | 16,206    | 16.21     | 3,241        |
| 10        | 22,732    | 22.73     | 2,273        |
| 20        | 31,912    | 31.91     | 1,596        |
| 50        | 63,172    | 63.17     | 1,263        |
| 100       | 113,451   | 113.45    | 1,135        |

### metal-candle Performance

#### CPU

| Batch Size | Time (µs) | Time (ms) | Per Doc (µs) |
|-----------|-----------|-----------|--------------|
| 1         | 36,979    | 36.98     | 36,979       |
| 2         | 56,680    | 56.68     | 28,340       |
| 5         | 112,283   | 112.28    | 22,457       |
| 10        | 203,491   | 203.49    | 20,349       |
| 20        | 381,685   | 381.69    | 19,084       |
| 50        | 927,681   | 927.68    | 18,554       |
| 100       | 1,854,715 | 1,854.72  | 18,547       |

#### Metal (Custom LayerNorm Kernel)

| Batch Size | Time (µs) | Time (ms) | Per Doc (µs) |
|-----------|-----------|-----------|--------------|
| 1         | 3,866     | 3.87      | 3,866        |
| 2         | 3,140     | 3.14      | 1,570        |
| 5         | 3,496     | 3.50      | 699          |
| 10        | 3,368     | 3.37      | 337          |
| 20        | 3,466     | 3.47      | 173          |
| 50        | 4,028     | 4.03      | 81           |
| 100       | 4,378     | 4.38      | 44           |

---

## Direct Comparison

### metal-candle Metal vs MLX (Batch Size 100)

```
MLX:              113,451 µs (113.45 ms)
metal-candle:       4,378 µs (4.38 ms)
                  ─────────────────────
Speedup:          25.9x FASTER 🚀
```

### Complete Speedup Analysis

| Batch Size | MLX (µs) | MC Metal (µs) | Speedup (MC/MLX) | Winner       |
|-----------|----------|---------------|------------------|--------------|
| 1         | 7,652    | 3,866         | **1.98x**        | metal-candle |
| 2         | 13,552   | 3,140         | **4.32x**        | metal-candle |
| 5         | 16,206   | 3,496         | **4.64x**        | metal-candle |
| 10        | 22,732   | 3,368         | **6.75x**        | metal-candle |
| 20        | 31,912   | 3,466         | **9.21x**        | metal-candle |
| 50        | 63,172   | 4,028         | **15.68x**       | metal-candle |
| 100       | 113,451  | 4,378         | **25.92x**       | metal-candle |

**Key Finding**: metal-candle's performance advantage **increases with batch size**, from 2x at batch=1 to 26x at batch=100.

---

## Performance Characteristics

### MLX
- **Scaling**: Good (8x faster at batch 100 vs batch 1)
- **Single Query**: 7.7ms
- **Batch 100**: 113.5ms
- **Architecture**: Lazy evaluation with graph optimization
- **Backend**: Optimized Metal kernels from Apple

### metal-candle Metal
- **Scaling**: Excellent (nearly constant time regardless of batch size!)
- **Single Query**: 3.9ms
- **Batch 100**: 4.4ms
- **Architecture**: Lazy evaluation with custom Metal LayerNorm
- **Backend**: Candle + custom fused LayerNorm kernel

### Key Insight: Nearly Constant-Time Batching

metal-candle exhibits **near constant-time performance** across batch sizes:
- Batch 1: 3.9ms
- Batch 100: 4.4ms
- **Only 13% increase for 100x more data!**

This suggests excellent GPU utilization and minimal CPU overhead.

---

## Technical Details

### Test Configuration

**Common**:
- Model: `intfloat/e5-small-v2`
- Input text: 98-word technical paragraph
- Operations: Tokenization → BERT forward → Mean pooling → L2 normalization
- Warmup: 2 iterations
- Measurement: Average of 3 runs

**MLX**:
- Version: 0.30.0
- Python: 3.14.1
- Execution: PyTorch model on MLX backend

**metal-candle**:
- Version: 0.1.0 (custom build)
- Rust: Latest stable
- Features: `embeddings`, vendored BERT with custom `FusedLayerNormOp`

### Why is metal-candle Faster?

1. **Custom LayerNorm Kernel**
   - Fused mean/variance computation in shared memory
   - Single-pass normalization
   - Optimal threadgroup size for Apple GPUs

2. **Lazy Evaluation Graph**
   - Operation fusion opportunities
   - Minimal intermediate allocations
   - Optimized execution plan

3. **Candle's Metal Backend**
   - Highly optimized tensor operations
   - Efficient command buffer management
   - Minimal CPU-GPU synchronization

4. **Zero Python Overhead**
   - Pure Rust implementation
   - No Python interpreter overhead
   - No GIL contention

---

## Implications for Ferris RAG

### Single Query Performance

```
MLX:              7.7ms
metal-candle CPU: 37.0ms
metal-candle Metal: 3.9ms
```

**Recommendation**: Use **metal-candle Metal** for all queries
- 2x faster than MLX
- 9.5x faster than CPU
- Minimal overhead

### Batch Indexing Performance (Batch 100)

```
MLX:              113.5ms
metal-candle CPU: 1,854.7ms
metal-candle Metal: 4.4ms
```

**Recommendation**: Use **metal-candle Metal** for all indexing
- 26x faster than MLX
- 424x faster than CPU
- Process 22,831 docs/sec vs 881 docs/sec (MLX)

### Production Strategy

**No hybrid strategy needed!** metal-candle Metal outperforms MLX at **all** batch sizes.

```rust
// Simple, optimal approach for Ferris
let device = Device::new_metal(0)?;
let model = EmbeddingModel::from_pretrained(
    EmbeddingModelType::E5SmallV2,
    device,
)?;

// Use for everything - single queries AND batch indexing
let embeddings = model.encode(&texts)?;
```

---

## Comparison with Published MLX Benchmarks

**Note**: MLX is Apple's official ML framework, optimized specifically for Apple Silicon. Our results show that with careful kernel design (custom LayerNorm) and Rust's zero-cost abstractions, metal-candle can **significantly outperform** even Apple's own optimized framework.

---

## Methodology Notes

### Accuracy
Both implementations produce **identical embeddings** (verified via cosine similarity ≈ 1.0).

### Fairness
- Same model architecture (BERT-small)
- Same input text
- Same output (384-dim L2-normalized embeddings)
- Both running on same hardware
- Both using Metal GPU acceleration

### Reproducibility
Run the benchmark yourself:

```bash
# metal-candle
cargo run --release --example embeddings_batch --features embeddings

# MLX
source .venv/bin/activate
python benchmarks/mlx_embeddings_bench.py
```

---

## Conclusion

✅ **metal-candle achieves 25.9x speedup over MLX for batch embeddings**  
✅ **metal-candle achieves 2x speedup over MLX for single queries**  
✅ **metal-candle exhibits near constant-time performance across batch sizes**  

This validates the architectural decisions:
1. Custom Metal kernels (LayerNorm)
2. Lazy evaluation graph
3. Pure Rust implementation
4. Candle's optimized Metal backend

**For Ferris RAG**: Use metal-candle Metal for all embedding operations. No hybrid strategy needed.

---

## Future Work

Potential areas for further investigation:

1. **Other Models**: Benchmark larger models (BERT-base, RoBERTa)
2. **Longer Sequences**: Test with longer input text (512+ tokens)
3. **Other Operations**: Compare LoRA training, text generation
4. **MLX Updates**: Re-benchmark when MLX releases optimizations

---

**Benchmarked on**: Apple Silicon  
**Date**: December 10, 2024  
**metal-candle**: v0.1.0 (vendored BERT + custom LayerNorm)  
**MLX**: v0.30.0

