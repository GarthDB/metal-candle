# Phase 4: Inference & Text Generation - Implementation Plan

**Status**: In Progress  
**Branch**: `phase-4-inference`  
**Timeline**: Current development

## Overview

Phase 4 delivers efficient text generation with KV-cache, multiple sampling strategies, and streaming support. This phase focuses on inference performance and ergonomic APIs for end users.

## Objectives

1. ✅ Implement KV-cache for efficient autoregressive generation
2. ✅ Add multiple sampling strategies (greedy, top-k, top-p, temperature)
3. ⏳ Build text generation pipeline with streaming support
4. ⏳ Create comprehensive inference tests
5. ⏳ Develop working examples and documentation

## Components

### 1. KV-Cache (`src/inference/cache.rs`) ✅

**Purpose**: Eliminate redundant computation in autoregressive generation by caching key/value tensors from previous tokens.

**Features**:
- Per-layer cache storage: `HashMap<layer_idx, (key, value)>`
- Position tracking for sequence length management
- Configurable maximum sequence length
- Clear and update operations
- Memory-efficient design (~173 MB for full 2048-token context)

**API**:
```rust
let mut cache = KVCache::new(config, &device)?;
let (full_key, full_value) = cache.update(layer_idx, &key, &value)?;
cache.clear(); // Reset for new sequence
```

**Memory Usage** (for 24 layers, 14 heads, 64 dim/head, 2048 max seq, F16):
```
24 layers * 2 (key+value) * (1 * 14 * 2048 * 64) * 2 bytes ≈ 173 MB
```

**Tests**: 6 tests covering creation, updates, clear, overflow

### 2. Sampling Strategies (`src/inference/sampling.rs`) ✅

**Purpose**: Provide multiple methods for token selection from model logits.

**Strategies**:

1. **Greedy** (deterministic):
   - Selects argmax token
   - Best for factual/consistent generation
   - No randomness

2. **Top-k**:
   - Samples from top k most likely tokens
   - Limits randomness to k candidates
   - Good for creative but constrained generation
   - Typical k: 40-50

3. **Top-p (Nucleus)**:
   - Samples from smallest set with cumulative prob ≥ p
   - Adaptive vocabulary size
   - Best for natural language generation
   - Typical p: 0.9-0.95

4. **Temperature**:
   - Scales logits before softmax
   - T > 1: more random, T < 1: more deterministic
   - Can be combined with top-k/top-p
   - Typical T: 0.7-1.0

**API**:
```rust
let strategy = SamplingStrategy::TopP { p: 0.9 };
let token = sample_token(&logits, &strategy)?;
```

**Tests**: 3 tests for greedy, top-k, and strategy construction

### 3. Text Generation Pipeline (`src/inference/generator.rs`) ⏳

**Purpose**: Orchestrate the full generation loop with model, tokenizer, cache, and sampling.

**Architecture**:
```
User Input (text)
    ↓
Tokenizer (encode)
    ↓
Model Forward Pass (with KV-cache)
    ↓
Sampling (select next token)
    ↓
Tokenizer (decode)
    ↓
Output Token / Text
    ↓
Repeat until EOS or max_tokens
```

**Features**:
- Automatic KV-cache management
- Configurable stopping criteria (EOS, max_tokens, custom)
- Streaming callbacks for real-time generation
- Batch support for efficient multi-sequence generation
- Error recovery and timeout handling

**Configuration**:
```rust
pub struct GeneratorConfig {
    pub max_tokens: usize,        // Maximum tokens to generate
    pub sampling: SamplingStrategy, // Sampling method
    pub temperature: f64,          // Temperature (if used)
    pub eos_token_id: Option<u32>, // End-of-sequence token
    pub repetition_penalty: f64,   // Penalize repeated tokens
    pub stop_sequences: Vec<String>, // Additional stop strings
}
```

**API**:
```rust
let generator = Generator::new(model, tokenizer, config, &device)?;

// Synchronous generation
let text = generator.generate("Once upon a time")?;

// Streaming generation
generator.generate_stream("Hello", |token, text| {
    print!("{}", text);
    Ok(()) // Return Err to stop generation
})?;
```

**Implementation Notes**:
- Model integration deferred to Phase 4 completion
- Tokenizer integration via `tokenizers` crate
- Current implementation is scaffold for full Phase 4 delivery

### 4. Streaming Support ⏳

**Purpose**: Enable real-time token-by-token generation for interactive applications.

**Callback Interface**:
```rust
pub type StreamCallback = dyn FnMut(u32, &str) -> Result<()>;

pub fn generate_stream<F>(
    &mut self,
    prompt: &str,
    callback: F,
) -> Result<String>
where
    F: FnMut(u32, &str) -> Result<()>
{
    // For each generated token:
    callback(token_id, &decoded_text)?;
}
```

**Use Cases**:
- Chat interfaces (real-time response)
- Progress indicators
- Streaming to websockets/SSE
- Early stopping based on content

### 5. Integration Tests ⏳

**Coverage**:
1. End-to-end generation with mock model
2. KV-cache integration with multi-token sequences
3. Sampling strategy correctness
4. Streaming callback functionality
5. Error handling (EOS, max_tokens, cache overflow)
6. Memory efficiency validation

**Test Plan**:
```rust
// tests/inference/generation.rs
#[test]
fn test_generation_with_kv_cache() { }

#[test]
fn test_streaming_callback() { }

#[test]
fn test_max_tokens_limit() { }

#[test]
fn test_eos_stopping() { }
```

### 6. Examples ⏳

**`examples/generate_text.rs`**: Complete text generation demo

**Features**:
- Load pre-trained model (placeholder for now)
- Multiple sampling strategies demo
- Streaming vs synchronous generation
- Configuration examples
- Performance metrics (tokens/sec)

**Structure**:
```rust
fn main() -> Result<()> {
    // Setup
    let device = Device::new_with_fallback(0);
    let config = GeneratorConfig { ... };
    
    // Generate with different strategies
    demo_greedy(&generator)?;
    demo_top_k(&generator)?;
    demo_top_p(&generator)?;
    demo_streaming(&generator)?;
    
    Ok(())
}
```

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| KV-cache speedup | ≥2x vs recompute | ⏳ To measure |
| Token generation | ≥50 tok/s (CPU baseline) | ⏳ To measure |
| Memory overhead | ≤200 MB for 2K context | ✅ ~173 MB |
| Sampling overhead | <5% of forward pass | ⏳ To measure |

## Quality Gates

- [x] KV-cache implementation complete
- [x] All sampling strategies implemented
- [x] Zero clippy warnings
- [x] 100% API documentation
- [ ] Integration tests passing
- [ ] Working example
- [ ] Performance benchmarks

## Dependencies

**Existing**:
- `candle-core` - Tensor operations
- `rand` - Random sampling

**New** (for full implementation):
- `tokenizers` - HuggingFace tokenizers (already in Cargo.toml)

## Timeline

- **Week 1**: KV-cache + sampling ✅ (completed)
- **Week 2**: Generation pipeline + streaming ⏳ (in progress)
- **Week 2**: Tests + examples + docs ⏳ (pending)

## Success Criteria

1. ✅ KV-cache reduces redundant computation
2. ✅ Multiple sampling strategies available
3. ⏳ Text generation works end-to-end
4. ⏳ Streaming supports real-time use cases
5. ⏳ API is ergonomic and well-documented
6. ⏳ Examples demonstrate common patterns

## Notes

- Full model integration will happen with Qwen architecture in later phases
- Current generator is a scaffold - full implementation when model is ready
- Focus on API design and infrastructure first
- Performance optimization deferred to Phase 5

## Next Steps

1. Expand generator implementation with streaming
2. Create integration tests
3. Build comprehensive example
4. Measure performance baselines
5. Document usage patterns

---

**Phase 4 delivers the inference infrastructure needed for production text generation.**

