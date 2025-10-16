# metal-candle Architecture

**Version**: 0.1.0  
**Last Updated**: October 2025

## Overview

`metal-candle` is a production-quality Rust crate for machine learning on Apple Silicon, built on top of [Candle](https://github.com/huggingface/candle) with Metal backend support. It provides high-performance LoRA training, model loading, and text generation for transformer models, with a focus on simplicity, correctness, and performance.

## Design Goals

1. **Production Quality**: Every commit meets production standards with zero warnings
2. **Pure Rust**: No Python dependencies, enabling single-binary deployment
3. **Apple Silicon First**: Optimized for Metal with M-series chips as primary target
4. **Type Safety**: Leverage Rust's type system for correctness
5. **Ergonomic APIs**: Builder patterns and clear error handling
6. **Well Documented**: 100% API documentation coverage with examples

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  (ferris, CLI tools, custom applications)                   │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    metal-candle (Public API)                 │
├─────────────────────────────────────────────────────────────┤
│  Training          │  Inference        │  Models            │
│  • LoRAAdapter     │  • KVCache        │  • ModelLoader     │
│  • Trainer         │  • Sampling       │  • Qwen           │
│  • AdamW           │  • Generator      │  • Config          │
│  • Schedulers      │                   │                    │
│  • Checkpoint      │                   │                    │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Backend Abstractions                      │
│  • Device (Metal/CPU)                                       │
│  • TensorExt (softmax_stable, layer_norm, rms_norm)        │
│  • Error handling (thiserror)                               │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      Candle Framework                        │
│  • Tensor operations                                        │
│  • Metal backend                                            │
│  • Autograd (Var, backward, GradStore)                     │
│  • VarBuilder                                               │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      Apple Metal API                         │
│  (GPU acceleration on Apple Silicon)                        │
└─────────────────────────────────────────────────────────────┘
```

## Module Organization

### Core Modules

#### `src/backend/`
Low-level abstractions over Candle's device and tensor APIs.

- **`device.rs`**: Metal/CPU device creation and management
  - `Device::new_metal(idx)` - Create Metal device
  - `Device::new_cpu()` - Create CPU fallback
  - `Device::new_with_fallback(idx)` - Try Metal, fallback to CPU
  - Device info queries (type, name, memory)

- **`tensor.rs`**: Extended tensor operations
  - `softmax_stable()` - Numerically stable softmax
  - `layer_norm()` - Layer normalization
  - `rms_norm()` - Root Mean Square normalization

#### `src/models/`
Model loading, configuration, and architecture implementations.

- **`config.rs`**: Model configuration parsing from JSON
  - `ModelConfig::from_json()` - Parse config from JSON
  - Validation and helper methods (`head_dim()`, `intermediate_size()`)

- **`loader.rs`**: Model weight loading from safetensors
  - `ModelLoader` with builder pattern
  - `load()`, `load_with_validation()`, `inspect()`
  - Automatic dtype conversion

- **`transformer.rs`**: Reusable transformer components
  - `RotaryEmbedding` - RoPE positional encoding
  - `Attention` - Multi-head attention with GQA support
  - `MLP` - Feed-forward network with SwiGLU

- **`qwen.rs`**: Qwen2.5-Coder architecture
  - `Qwen` - Full model implementation
  - `QwenDecoderLayer` - Transformer layer
  - `RMSNorm` - Normalization layer

#### `src/training/`
LoRA training infrastructure and components.

- **`lora.rs`**: LoRA layer implementation
  - `LoRALayer` - Low-rank adaptation layer
  - `LoRAConfig` - Configuration (rank, alpha, dropout)
  - Trainable `Var` parameters

- **`adapter.rs`**: LoRA adapter management
  - `LoRAAdapter` - Manages multiple LoRA layers
  - `TargetModule` - Which layers to adapt (QProj, VProj, etc.)
  - Methods: `add_layer`, `apply_lora_delta`, `merge_weights`

- **`loss.rs`**: Loss functions
  - `cross_entropy_loss()` - Standard cross-entropy
  - `cross_entropy_loss_with_smoothing()` - Label smoothing variant

- **`optimizer.rs`**: AdamW optimizer
  - `AdamW` - Optimizer with decoupled weight decay
  - `AdamWConfig` - Hyperparameters (lr, betas, epsilon, weight_decay)
  - Per-parameter state management (m, v tensors)

- **`scheduler.rs`**: Learning rate schedulers
  - `Constant` - Fixed learning rate
  - `Linear` - Linear warmup
  - `Cosine` - Cosine annealing
  - `WarmupCosine` - Warmup + cosine decay

- **`trainer.rs`**: Training loop coordination
  - `Trainer` - Main training orchestrator
  - `TrainingStep` - Single step execution
  - `TrainingConfig` - Training hyperparameters
  - Progress tracking with `StepMetrics`

- **`checkpoint.rs`**: Model checkpoint management
  - `save_checkpoint()` - Save LoRA weights + metadata
  - `load_checkpoint()` - Load LoRA weights
  - Safetensors format with JSON metadata

#### `src/inference/`
Efficient text generation infrastructure.

- **`cache.rs`**: KV-cache for autoregressive generation
  - `KVCache` - Per-layer key/value caching
  - ~173 MB memory for 2048 tokens (Qwen 0.5B)
  - O(1) position tracking per token

- **`sampling.rs`**: Token sampling strategies
  - `sample_greedy()` - Deterministic argmax
  - `sample_top_k()` - Top-k sampling
  - `sample_top_p()` - Nucleus (top-p) sampling
  - `sample_temperature()` - Temperature-scaled sampling

- **`generator.rs`**: Text generation pipeline (scaffold)
  - `Generator` - Generation orchestrator
  - `GeneratorConfig` - Generation parameters
  - Future: Full end-to-end generation

#### `src/error.rs`
Centralized error handling using `thiserror`.

- `Error` - Top-level enum
- `ModelError` - Model loading/validation errors
- `TrainingError` - Training-related errors
- `InferenceError` - Inference/generation errors
- `CheckpointError` - Checkpoint I/O errors

## Key Design Patterns

### 1. Builder Pattern for Configuration

Complex types use builder pattern for ergonomic construction:

```rust
let model = ModelLoader::new()
    .with_device(device)
    .with_dtype(DType::F16)
    .load("model.safetensors")?;
```

### 2. Newtype Pattern for Type Safety

Strong types prevent misuse:

```rust
pub struct Rank(usize);
pub struct Alpha(f32);

impl Rank {
    pub fn new(rank: usize) -> Result<Self> {
        if rank == 0 {
            return Err(LoRAError::InvalidRank);
        }
        Ok(Self(rank))
    }
}
```

### 3. Error Handling with `thiserror`

Library errors use `thiserror` (not `anyhow`):

```rust
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("model file not found: {path}")]
    FileNotFound { path: PathBuf },
    
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
```

### 4. Explicit Lifetimes Where Needed

Borrows are explicit to prevent copies:

```rust
pub fn update(&mut self, layer_idx: usize, key: &Tensor, value: &Tensor) 
    -> Result<(Tensor, Tensor)>
```

### 5. Device Abstractions

Consistent device handling across the codebase:

```rust
let device = Device::new_with_fallback(0); // Try Metal, fallback to CPU
let tensor = Tensor::zeros((batch, seq), DType::F32, &device)?;
```

## Training Architecture

### LoRA Training Flow

```
1. Model Setup
   ├─> Load base model (safetensors)
   ├─> Create LoRAAdapter (specify target modules)
   └─> Wrap with Trainer

2. Training Loop
   ├─> For each epoch
   │   ├─> For each batch
   │   │   ├─> Forward pass (base model + LoRA delta)
   │   │   ├─> Compute loss (cross-entropy)
   │   │   ├─> Backward pass (compute gradients)
   │   │   ├─> Optimizer step (AdamW.step_var)
   │   │   └─> LR scheduler update
   │   └─> Save checkpoint (optional)
   └─> Return trained adapter

3. Checkpoint Save
   └─> Serialize LoRA Var tensors to safetensors
```

### Gradient Flow

Candle's autograd system:

```rust
// 1. Create trainable Var
let lora_a = Var::new(tensor_a, &device)?;
let lora_b = Var::new(tensor_b, &device)?;

// 2. Forward pass (using .as_tensor())
let delta = lora_a.as_tensor().matmul(lora_b.as_tensor())?;
let output = input.broadcast_add(&delta)?;

// 3. Compute loss
let loss = cross_entropy_loss(&output, &labels)?;

// 4. Backward pass
let grads = loss.backward()?;

// 5. Extract gradients
let grad_a = grads.get(&lora_a).unwrap();
let grad_b = grads.get(&lora_b).unwrap();

// 6. Optimizer update
optimizer.step_var(&lora_a, grad_a)?;
optimizer.step_var(&lora_b, grad_b)?;
```

## Inference Architecture

### KV-Cache Design

Efficient autoregressive generation by caching attention key/value tensors:

```
Token 1 (Prompt):
  Layer 0: K₁, V₁ → Cache
  Layer 1: K₁, V₁ → Cache
  ...
  Layer N: K₁, V₁ → Cache

Token 2 (Generation):
  Layer 0: K₂, V₂ → Concat([K₁, V₁], [K₂, V₂]) → Cache
  Layer 1: K₂, V₂ → Concat([K₁, V₁], [K₂, V₂]) → Cache
  ...
  (Reuse cached context, only compute new token)

Result: ~2x speedup for long sequences
Memory: ~173 MB for 2048 tokens (Qwen 0.5B, F16)
```

### Sampling Strategies

1. **Greedy**: Always pick highest probability token
   - Deterministic
   - Best for factual generation

2. **Top-k**: Sample from top k candidates
   - Limits randomness
   - Typical k: 40-50

3. **Top-p (Nucleus)**: Sample from smallest set with cumulative prob ≥ p
   - Adaptive vocabulary size
   - Typical p: 0.9-0.95
   - Best for natural language

4. **Temperature**: Scale logits before sampling
   - T > 1: more random
   - T < 1: more deterministic
   - Typical T: 0.7-1.0

## Memory Management

### Tensor Lifecycle

1. **Creation**: Tensors allocated on device
2. **Operations**: Candle manages intermediate tensors
3. **Cleanup**: Automatic via Rust's RAII (Drop trait)
4. **Explicit**: Use `drop()` for early cleanup if needed

### Memory-Intensive Operations

- **Model Loading**: Tensors streamed from safetensors
- **Training**: Gradients computed on-demand
- **KV-Cache**: Incremental concatenation
- **Inference**: Single-token processing after prompt

## Performance Considerations

### Metal Optimization

1. **F16 Precision**: Use F16 for memory and speed
2. **Contiguous Tensors**: Call `.contiguous()` after reshape/transpose
3. **Batch Operations**: Prefer batched over sequential
4. **Memory Layout**: Ensure proper striding for Metal kernels

### Numerical Stability

1. **Stable Softmax**: Subtract max before exp
2. **Epsilon Values**: Use 1e-7 for f32/f64 comparisons
3. **Gradient Clipping**: Prevent gradient explosion
4. **Mixed Precision**: F16 forward, F32 accumulation

## Testing Strategy

### Test Organization

```
src/
├── module/
│   └── tests.rs          # Unit tests (in-module)
tests/
├── integration/          # Integration tests
│   ├── training.rs
│   ├── inference_integration.rs
│   └── gradient_verification.rs
examples/
└── *.rs                  # Runnable examples (manual validation)
```

### Test Types

1. **Unit Tests**: Fast, isolated, in src/
2. **Integration Tests**: End-to-end workflows
3. **Property Tests**: Numerical correctness (future: proptest)
4. **Snapshot Tests**: Model outputs (future: insta)

### Coverage Target

- **Overall**: ≥80%
- **Public APIs**: 100%
- **Core Algorithms**: 100%
- **Backend/Utilities**: ≥80%

## Quality Gates

All PRs must pass:

1. ✅ `cargo clippy -- -D warnings` (zero warnings)
2. ✅ `cargo test` (all tests passing)
3. ✅ `cargo fmt --check` (formatted correctly)
4. ✅ Code coverage ≥80%
5. ✅ Documentation complete for public APIs

## Future Enhancements

### Phase 5 (Current)
- [ ] Comprehensive benchmarking vs MLX
- [ ] Complete documentation
- [ ] Additional examples

### Phase 6
- [ ] Publish to crates.io (v1.0)
- [ ] Ferris integration

### Future Considerations
- [ ] GGUF format support
- [ ] Additional model architectures (LLaMA, Mistral)
- [ ] Quantization (4-bit, 8-bit)
- [ ] Multi-GPU support
- [ ] Flash Attention integration
- [ ] Streaming generation with callbacks

## Dependencies

### Core
- `candle-core`, `candle-nn` - ML framework
- `safetensors` - Model format
- `thiserror` - Error handling

### Optional
- `tokenizers` - HuggingFace tokenizers (future)
- `rand` - Random sampling

### Dev Dependencies
- `criterion` - Benchmarking
- `approx` - Float comparisons
- `tempfile` - Test fixtures
- `anyhow` - Test error handling

## References

- [Candle Framework](https://github.com/huggingface/candle)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Qwen2.5 Model](https://github.com/QwenLM/Qwen2.5)
- [Safetensors Format](https://github.com/huggingface/safetensors)
- [Apple Metal](https://developer.apple.com/metal/)

---

**Maintained by**: metal-candle contributors  
**License**: Apache-2.0  
**Last Updated**: October 2025

