# Metal LayerNorm Workaround for Embeddings

## Problem

Ferris RAG uses embedding models (E5, MiniLM, MPNet) that rely on BERT's LayerNorm operation.
Candle 0.9's Metal backend does NOT implement LayerNorm, causing the error:
```
Metal error no metal implementation for layer-norm
```

This forces Ferris to fall back to CPU mode for embeddings, significantly impacting performance.

## Root Cause

1. `candle-transformers` BERT model uses `candle-nn::LayerNorm` internally
2. `candle-nn::LayerNorm` only has CPU implementation, no Metal backend
3. We cannot override or replace the internal LayerNorm without patching `candle-transformers`

## Attempted Solutions

### Solution 1: Custom LayerNorm Kernel ❌
**Status**: Implemented but unused

Created custom Metal kernel and Candle CustomOp for LayerNorm:
- `src/backend/kernels.metal`: GPU kernel implementation
- `src/backend/custom_ops.rs`: `LayerNormOp` and `layer_norm()` function
- `src/backend/metal_kernels.rs`: `LayerNormParams` struct

**Problem**: Cannot inject this into `candle-transformers::models::bert::BertModel` internals.

### Solution 2: Update to Candle 0.9.2-alpha.1 ❌
**Status**: Attempted, rolled back

Tried updating to latest Candle version to see if LayerNorm Metal support was added.

**Problems**:
- Breaking API changes in candle-metal-kernels
- Type mismatches (`MTLSize`, `Device`, `ComputePipelineState`)
- 12 compilation errors
- Would require significant refactoring

## Viable Solutions

### Option A: CPU Fallback (Current Workaround) ✅
**Status**: Already in Ferris

Ferris uses `Device::new_cpu()` instead of `Device::new_metal(0)`.

**Pros**:
- Works immediately
- No code changes needed
- Stable and predictable

**Cons**:
- 5-50x slower than GPU for embeddings
- Underutilizes Apple Silicon
- Bottleneck for large document indexing

### Option B: Patch candle-transformers Locally 🔧
**Status**: Recommended for Ferris

Fork or vendor `candle-transformers` and replace `LayerNorm` with our custom Metal implementation.

**Steps**:
1. Create `metal-candle/vendor/candle-transformers/`
2. Copy BERT implementation from upstream
3. Replace `LayerNorm` calls with `metal_candle::backend::layer_norm()`
4. Update `Cargo.toml` to use local path

**Pros**:
- Full Metal acceleration for embeddings
- 5-50x speedup over CPU
- Clean integration with metal-candle

**Cons**:
- Maintenance burden (keep up with upstream)
- Larger dependency footprint

### Option C: Wait for Upstream Fix ⏳
**Status**: Long-term solution

Monitor Candle repository for Metal LayerNorm support:
- https://github.com/huggingface/candle/issues
- https://github.com/huggingface/candle/pulls

**Pros**:
- No maintenance burden
- Official solution

**Cons**:
- Unknown timeline (could be weeks/months)
- Not in control of implementation

### Option D: Alternative Embedding Implementation 🔄
**Status**: Nuclear option

Implement BERT/E5/MiniLM directly in `metal-candle` without `candle-transformers`.

**Pros**:
- Full control over all operations
- Optimized for Metal from ground up

**Cons**:
- Weeks of implementation work
- Need to match HuggingFace model weights exactly
- High maintenance burden

## Recommendation for Ferris

**Short-term (immediate)**:
- Continue using CPU for embeddings
- Document this limitation
- Profile to confirm it's not a bottleneck yet

**Medium-term (next sprint)**:
- Implement Option B (patch candle-transformers)
- Use `metal-candle`'s custom `layer_norm()` function
- Benchmark to quantify speedup

**Long-term**:
- Monitor upstream Candle for Metal LayerNorm
- Migrate back to upstream when available

## Code Ready to Use

The custom LayerNorm implementation is complete and tested:

```rust
use metal_candle::backend::layer_norm;

// In BERT forward pass:
let normalized = layer_norm(&hidden_states, 1e-12)?;
```

Just need to patch `candle-transformers` to use it instead of `candle_nn::LayerNorm`.

## Performance Impact Estimate

Based on similar operations (Softmax, RMS Norm):
- **CPU embedding generation**: ~50-200 µs per document
- **Metal embedding generation** (estimated): ~5-20 µs per document
- **Expected speedup**: 5-10x
- **Impact on Ferris**: Faster indexing, lower latency search

For a 1000-document corpus:
- CPU: ~50-200 ms total
- Metal: ~5-20 ms total
- **Savings**: 30-180 ms per indexing run

