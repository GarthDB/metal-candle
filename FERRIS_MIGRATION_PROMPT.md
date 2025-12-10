# Ferris Migration to metal-candle v1.0

## ✅ Dropout Support Included in v1.0.0

**Dropout support is now fully implemented** in metal-candle v1.0.0 ([Issue #28](https://github.com/GarthDB/metal-candle/issues/28) - closed):

**Features**:
- ✅ Training/eval mode control (`set_training()`, `eval()`, `is_training()`)
- ✅ Dropout applied between A and B matrices (per LoRA paper)
- ✅ Automatic disable in eval mode for deterministic inference
- ✅ Preserves gradient flow for backpropagation
- ✅ 8 comprehensive tests included

**Usage**:
```rust
let mut layer = LoRALayer::new(64, 64, &config, &device)?;
layer.set_training(true);  // Enable dropout (if configured)
layer.eval();              // Disable dropout for inference
```

---

## Context

I need to migrate the Ferris AI Assistant project from using PyO3+MLX to the new pure-Rust `metal-candle` v1.0 library. This will eliminate Python dependencies and provide native Rust performance with Metal acceleration on Apple Silicon.

## Current State

**Project**: `~/Projects/Ferris` (monorepo)  
**Module**: `apps/ferris-mlx` (MLX integration via PyO3)  
**Dependencies**: 
- `pyo3 = "0.26"` (Python bridge)
- `candle-core = "0.3"` (outdated)
- `candle-transformers = "0.3"` (outdated)
- `candle-nn = "0.3"` (outdated)

**Current Modules in `ferris-mlx`**:
- `bridge.rs` - PyO3 bridge to Python/MLX
- `lora.rs` - LoRA configuration
- `lora_adapter.rs` - LoRA adapter using MLX operations
- `trainer.rs` - MLX training engine
- `evaluation.rs` - Model evaluation
- `data_converter.rs` - Training data management
- `scheduler.rs` - Fine-tuning scheduler

## Target State

Replace PyO3+MLX with `metal-candle v1.0` from local path:
```toml
metal-candle = { path = "../metal-candle" }
```

## metal-candle v1.0 Capabilities

### ✅ Available Features

**LoRA Training**:
```rust
use metal_candle::training::{
    LoRAAdapter, LoRAAdapterConfig, TargetModule,
    Trainer, TrainingConfig, LRScheduler, AdamWConfig
};

// Create LoRA adapter
let lora_config = LoRAAdapterConfig {
    rank: 8,
    alpha: 16.0,
    dropout: 0.0,
    target_modules: vec![TargetModule::QProj, TargetModule::VProj],
};
let adapter = LoRAAdapter::new(&model, lora_config, &device)?;

// Train
let training_config = TrainingConfig {
    num_epochs: 3,
    lr_scheduler: LRScheduler::warmup_cosine(100, 1000, 1e-4, 1e-6),
    optimizer_config: AdamWConfig::default(),
    max_grad_norm: Some(1.0),
};
let trainer = Trainer::new(adapter, training_config)?;
let metrics = trainer.train(&dataset)?;

// Save checkpoint
save_checkpoint(&trainer.lora_adapter(), "checkpoint.safetensors", None)?;
```

**Model Loading**:
```rust
use metal_candle::models::{ModelConfig, ModelLoader};
use metal_candle::Device;

let device = Device::new_with_fallback(0);
let config = ModelConfig::from_file("config.json")?;
let loader = ModelLoader::new(device).with_dtype(DType::F16);
let weights = loader.load("model.safetensors")?;
```

**Inference & Text Generation**:
```rust
use metal_candle::inference::{
    KVCache, KVCacheConfig, SamplingStrategy, sample_token
};

let cache_config = KVCacheConfig {
    max_seq_len: 2048,
    num_layers: 24,
    num_heads: 14,
    head_dim: 64,
    batch_size: 1,
};
let mut cache = KVCache::new(cache_config, &device)?;

// Generate with different strategies
let strategy = SamplingStrategy::TopP { p: 0.9 };
let token = sample_token(&logits, &strategy)?;
```

**Semantic Embeddings** (25.9x faster than MLX!):
```rust
use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};

let device = Device::new_metal(0)?;
let model = EmbeddingModel::from_pretrained(
    EmbeddingModelType::E5SmallV2,
    device,
)?;

// 100 docs in 4.4ms vs MLX's 113.5ms
let embeddings = model.encode(&texts)?;
```

### 📦 What's Included

- ✅ **LoRA Training**: Rank 2-16, Q/K/V/O projection targets
- ✅ **AdamW Optimizer**: With decoupled weight decay
- ✅ **LR Schedulers**: Constant, Linear, Cosine, WarmupCosine
- ✅ **Checkpoint Management**: Save/load with metadata
- ✅ **Safetensors Format**: Primary model format
- ✅ **KV-Cache**: ~173 MB for 2048 tokens (Qwen 0.5B F16)
- ✅ **Sampling**: Greedy, Top-k, Top-p, Temperature
- ✅ **Embeddings**: E5, MiniLM, MPNet with Metal acceleration
- ✅ **Error Handling**: Comprehensive `thiserror`-based errors
- ✅ **Metal Acceleration**: Native Apple Silicon performance

### ⚠️ Limitations (v1.0)

- **Model Format**: Safetensors only (GGUF in v1.1+)
- **Architecture**: Qwen2.5-Coder for text, BERT for embeddings
- **Single GPU**: Multi-GPU in v2.0
- **Apple Silicon Only**: M1/M2/M3/M4 required

## Migration Strategy

### Phase 1: Update Dependencies

**In `Cargo.toml` (workspace root)**:
```toml
[workspace.dependencies]
# Remove old candle versions
# candle-core = "0.3"  # DELETE
# candle-transformers = "0.3"  # DELETE
# candle-nn = "0.3"  # DELETE

# Add metal-candle (use candle 0.9 from it)
metal-candle = { path = "../metal-candle", features = ["embeddings"] }
```

**In `apps/ferris-mlx/Cargo.toml`**:
```toml
[dependencies]
# Remove PyO3
# pyo3 = { version = "0.26", features = ["auto-initialize"] }  # DELETE
# pyo3-async-runtimes = { version = "0.26", features = ["tokio-runtime"] }  # DELETE
# numpy = "0.26"  # DELETE

# Add metal-candle
metal-candle.workspace = true
```

### Phase 2: Replace Modules

**Priority Order**:

1. **Error Types** (`error.rs`):
   - Replace MLX errors with `metal_candle::Error`
   - Map Python exceptions to Rust errors

2. **LoRA Configuration** (`lora.rs`):
   - Use `metal_candle::training::LoRAConfig`
   - Map `TargetModule` enum

3. **LoRA Adapter** (`lora_adapter.rs`):
   - Replace with `metal_candle::training::LoRAAdapter`
   - Remove PyO3 bridge code

4. **Training** (`trainer.rs`):
   - Use `metal_candle::training::Trainer`
   - Implement `TrainingConfig` with schedulers
   - Replace Python training loop with Rust

5. **Data Loading** (`data_converter.rs`):
   - Keep JSON parsing logic
   - Convert to Candle tensors directly
   - Remove numpy conversion

6. **Evaluation** (`evaluation.rs`):
   - Use native Rust model forward pass
   - Keep metrics calculation logic

7. **Bridge** (`bridge.rs`):
   - **DELETE** - No longer needed!

### Phase 3: API Migration Map

**Old PyO3/MLX → New metal-candle**:

```rust
// OLD: PyO3 bridge
let mlx_module = Python::with_gil(|py| { ... });

// NEW: Direct Rust
let device = Device::new_metal(0)?;
```

```rust
// OLD: LoRA via Python
py.run(r#"
    model.add_lora(rank=8, alpha=16)
"#, ...)?;

// NEW: Native Rust LoRA
let lora_config = LoRAAdapterConfig {
    rank: 8,
    alpha: 16.0,
    dropout: 0.0,
    target_modules: vec![TargetModule::QProj, TargetModule::VProj],
};
let adapter = LoRAAdapter::new(&model, lora_config, &device)?;
```

```rust
// OLD: Training via Python
mlx.train(model, data, epochs=3);

// NEW: Native Rust training
let config = TrainingConfig {
    num_epochs: 3,
    lr_scheduler: LRScheduler::warmup_cosine(100, 1000, 1e-4, 1e-6),
    optimizer_config: AdamWConfig::default(),
    max_grad_norm: Some(1.0),
};
let trainer = Trainer::new(adapter, config)?;
let metrics = trainer.train(&dataset)?;
```

```rust
// OLD: Checkpoints via Python
mlx.save_checkpoint(model, "checkpoint.npz");

// NEW: Safetensors
save_checkpoint(&adapter, "checkpoint.safetensors", None)?;
```

### Phase 4: Feature Additions

**New Capabilities** (not in MLX version):

1. **Embeddings** (25.9x faster):
```rust
let model = EmbeddingModel::from_pretrained(
    EmbeddingModelType::E5SmallV2,
    device,
)?;
let embeddings = model.encode(&documents)?;  // Batch: 4.4ms for 100 docs
```

2. **KV-Cache for Generation**:
```rust
let cache_config = KVCacheConfig { /* ... */ };
let mut cache = KVCache::new(cache_config, &device)?;
// ~173 MB for 2048 tokens
```

3. **Multiple Sampling Strategies**:
```rust
// Greedy
let token = sample_token(&logits, &SamplingStrategy::Greedy)?;

// Top-p (nucleus)
let token = sample_token(&logits, &SamplingStrategy::TopP { p: 0.9 })?;

// Top-k
let token = sample_token(&logits, &SamplingStrategy::TopK { k: 50 })?;

// Temperature
let token = sample_token(&logits, &SamplingStrategy::Temperature { temp: 0.7 })?;
```

## Testing Strategy

### Compatibility Tests

Create `apps/ferris-mlx/tests/metal_candle_migration.rs`:

```rust
//! Migration compatibility tests
//! Verify metal-candle produces same results as PyO3+MLX

#[test]
fn test_lora_output_compatibility() {
    // 1. Load same model/config as old system
    // 2. Run forward pass with metal-candle
    // 3. Compare outputs (should match within 1e-4)
}

#[test]
fn test_checkpoint_loading() {
    // Verify metal-candle can load old MLX checkpoints
    // (after conversion to safetensors)
}

#[test]
fn test_training_metrics() {
    // Train for 1 epoch with same data
    // Compare loss curves
}
```

### Performance Tests

```rust
#[test]
fn test_embeddings_performance() {
    // Verify 25.9x speedup claim
    let start = Instant::now();
    let embeddings = model.encode(&batch_of_100)?;
    let duration = start.elapsed();
    assert!(duration.as_millis() < 10);  // Should be ~4.4ms
}
```

## Expected Benefits

| Metric | Before (PyO3+MLX) | After (metal-candle) | Improvement |
|--------|-------------------|----------------------|-------------|
| **Startup Time** | ~200ms (Python init) | <10ms | **20x faster** |
| **Dependencies** | Python, MLX, numpy | Pure Rust | **-3 external deps** |
| **Binary Size** | N/A (requires Python) | Single binary | **Portable** |
| **Memory** | Python overhead | Native Rust | **~50MB less** |
| **Type Safety** | Runtime (Python) | Compile-time | **100% safe** |
| **Embeddings** | 113.5ms (100 docs) | 4.4ms | **25.9x faster** |
| **Deployment** | Python + venv | Single binary | **Easy** |

## Migration Checklist

- [ ] Update workspace `Cargo.toml` dependencies
- [ ] Update `ferris-mlx/Cargo.toml` dependencies
- [ ] Migrate `error.rs` to `metal_candle::Error`
- [ ] Migrate `lora.rs` to `metal_candle::training::LoRAConfig`
- [ ] Migrate `lora_adapter.rs` to `metal_candle::training::LoRAAdapter`
- [ ] Migrate `trainer.rs` to `metal_candle::training::Trainer`
- [ ] Migrate `data_converter.rs` to use Candle tensors
- [ ] Migrate `evaluation.rs` to native Rust
- [ ] Delete `bridge.rs` (no longer needed!)
- [ ] Update tests to use metal-candle APIs
- [ ] Add compatibility tests
- [ ] Add performance benchmarks
- [ ] Update documentation
- [ ] Remove Python virtual environment setup
- [ ] Test end-to-end training pipeline
- [ ] Verify checkpoint loading/saving
- [ ] Test inference with KV-cache
- [ ] Add embeddings integration (new feature!)

## Known Gotchas

1. **Checkpoint Format**: Old MLX checkpoints need conversion to safetensors
2. **Model Config**: Ensure `config.json` matches Qwen2.5-Coder schema
3. **Device Selection**: Use `Device::new_with_fallback(0)` for CPU fallback
4. **Tensor Shapes**: Candle uses `[batch, seq, features]` order
5. **Error Handling**: All operations return `Result` - use `?` operator

## Resources

- **metal-candle docs**: `~/Projects/metal-candle/README.md`
- **Examples**: `~/Projects/metal-candle/examples/`
- **API reference**: Run `cargo doc --open` in metal-candle
- **Architecture**: `~/Projects/metal-candle/ARCHITECTURE.md`
- **Benchmarks**: `~/Projects/metal-candle/BENCHMARKS.md`

## Questions to Answer

1. Do we have existing MLX checkpoints that need conversion?
2. What model architectures are we using? (Qwen2.5-Coder works)
3. Are we using features beyond LoRA training? (quantization, flash attention, etc.)
4. What's our target inference latency?
5. Do we need multi-GPU support? (v2.0 feature)

## Success Criteria

- ✅ All tests passing with metal-candle
- ✅ Training produces same/better loss curves
- ✅ Inference latency < MLX version
- ✅ Zero Python dependencies
- ✅ Single binary deployment works
- ✅ Embeddings integration functional
- ✅ Documentation updated
- ✅ Performance benchmarks documented

---

**Ready to migrate!** Start with Phase 1 (dependencies), then proceed module by module.

