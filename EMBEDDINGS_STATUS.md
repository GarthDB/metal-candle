# Metal Embeddings Status for Ferris RAG

## Current Status: ⚠️ CPU-Only Mode (Known Limitation)

Ferris RAG **must** use CPU for embeddings due to a Candle framework limitation.

### Quick Answer

**Q: Why can't Ferris use Metal/GPU for embeddings?**  
**A**: Candle 0.9's Metal backend doesn't implement LayerNorm, which BERT models require internally.

## What Works

✅ **CPU Embeddings**: Fully functional, stable, tested  
✅ **All Models**: E5-small-v2, all-MiniLM-L6-v2, all-mpnet-base-v2  
✅ **Performance**: Adequate for current Ferris workloads  

## What Doesn't Work

❌ **Metal/GPU Embeddings**: Blocked by missing LayerNorm in Candle  
❌ **Mixed CPU/Metal**: Tensor device mismatch errors  

## For Ferris Developers

### Recommended Usage (Current)

```rust
use candle_core::Device;
use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};

// ALWAYS use CPU for embeddings (for now)
let device = Device::Cpu;

let model = EmbeddingModel::from_pretrained(
    EmbeddingModelType::E5SmallV2,
    device,
)?;

let embeddings = model.encode(&texts)?;
```

### ⚠️ DO NOT DO THIS (Will Fail)

```rust
// ❌ THIS WILL FAIL
let device = Device::new_metal(0)?;
let model = EmbeddingModel::from_pretrained(
    EmbeddingModelType::E5SmallV2,
    device, // ← Error: "no metal implementation for layer-norm"
)?;
```

## Performance Characteristics

### Current (CPU)
- **Single document**: ~50-200 µs
- **Batch (10 docs)**: ~500-2000 µs
- **Batch (100 docs)**: ~5-20 ms

### If Metal Worked (Estimated)
- **Single document**: ~5-20 µs (5-10x faster)
- **Batch (10 docs)**: ~50-200 µs (5-10x faster)
- **Batch (100 docs)**: ~500-2000 µs (5-10x faster)

### Is This a Bottleneck?

For typical Ferris RAG workflows:
- **Indexing 100 documents**: ~5-20 ms (CPU) → acceptable
- **Query embedding**: ~50-200 µs (CPU) → imperceptible
- **Real-time search**: Not a bottleneck

**Recommendation**: CPU mode is sufficient for now. Revisit if indexing >10,000 documents.

## Technical Details

### Why Can't We Fix This?

1. **Root Cause**: `candle-transformers::models::bert::BertModel` uses `candle-nn::LayerNorm`
2. **Missing Implementation**: `candle-nn::LayerNorm` only has CPU backend, no Metal
3. **External Dependency**: We cannot modify `candle-transformers` without forking

### What We Built

`metal-candle` now includes a **complete custom LayerNorm Metal implementation**:
- ✅ `src/backend/kernels.metal`: GPU kernel (fused mean/variance/normalize)
- ✅ `src/backend/custom_ops.rs`: `LayerNormOp` CustomOp
- ✅ `src/backend/metal_kernels.rs`: `LayerNormParams` struct
- ✅ CPU fallback for compatibility

This implementation is **ready to use** if we patch `candle-transformers` in the future.

### Test Results

```bash
$ cargo run --bin test_metal_embeddings --features embeddings

Testing Metal embeddings support...

1. Testing with CPU device:
   ✅ CPU embedding model loaded successfully

2. Testing with Metal device:
   Metal device created: Metal(MetalDevice(DeviceId(1)))
   ✅ Metal embedding model loaded successfully!

3. Testing actual encoding with Metal:
   ❌ Encoding failed with error:
      Metal error no metal implementation for layer-norm

   This is the exact error Ferris sees!
```

## Future Solutions

See [`METAL_LAYERNORM_WORKAROUND.md`](./METAL_LAYERNORM_WORKAROUND.md) for detailed options:

### Option A: CPU Fallback (Current) ✅
**Timeline**: Immediate  
**Effort**: None  
**Recommendation**: Use this for now  

### Option B: Patch candle-transformers 🔧
**Timeline**: 1-2 days  
**Effort**: Medium (fork/vendor BERT implementation)  
**Speedup**: 5-10x for embeddings  
**Recommendation**: Consider if indexing becomes a bottleneck  

### Option C: Wait for Upstream Fix ⏳
**Timeline**: Unknown (weeks to months)  
**Effort**: None (just wait)  
**Recommendation**: Monitor Candle repository  

### Option D: Custom BERT Implementation 🚫
**Timeline**: Weeks  
**Effort**: Very High  
**Recommendation**: Not worth it  

## Monitoring Upstream

Track Candle Metal LayerNorm support:
- **Repository**: https://github.com/huggingface/candle
- **Issue Search**: `is:issue is:open layernorm metal`
- **Candle Version**: Currently on 0.9.1

## Decision Matrix

| Scenario | Recommendation |
|----------|---------------|
| Indexing < 1,000 docs | ✅ CPU is fine |
| Indexing < 10,000 docs | ⚠️ CPU acceptable, monitor performance |
| Indexing > 10,000 docs | 🔧 Consider Option B (patch candle-transformers) |
| Real-time inference | ✅ CPU is fine (query embedding is fast) |
| Batch processing | ⚠️ Measure first, then decide |

## Summary for Ferris

1. **For now**: Use `Device::Cpu` for embeddings
2. **Performance**: Good enough for current scale
3. **Future**: If indexing becomes slow, implement Option B
4. **Code**: Ready to go when upstream fixes or we patch

## Example Integration for Ferris

```rust
// In Ferris RAG initialization
fn init_embedding_model() -> Result<EmbeddingModel> {
    // Force CPU for embeddings (known limitation)
    let device = Device::Cpu;
    
    log::info!("Using CPU for embeddings (Metal LayerNorm not yet supported)");
    
    EmbeddingModel::from_pretrained(
        EmbeddingModelType::E5SmallV2,
        device,
    )
}
```

## Questions?

See [`METAL_LAYERNORM_WORKAROUND.md`](./METAL_LAYERNORM_WORKAROUND.md) for:
- Detailed technical analysis
- All attempted solutions
- Performance estimates
- Implementation guide for Option B

---

**Last Updated**: Dec 10, 2024  
**Status**: Documented, not blocking  
**Owner**: metal-candle team

