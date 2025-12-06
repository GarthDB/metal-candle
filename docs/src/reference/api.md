# API Documentation

Complete API reference for metal-candle.

## Online Documentation

The complete API documentation is available on docs.rs:

**[docs.rs/metal-candle](https://docs.rs/metal-candle)**

## Building Docs Locally

Build and view the documentation locally:

```bash
cd metal-candle
cargo doc --all-features --no-deps --open
```

## Main Modules

### `backend`

Device and tensor abstractions.

**Key Types**:
- `Device` - Metal GPU or CPU device
- `DeviceInfo` - Device metadata

[View Module](https://docs.rs/metal-candle/latest/metal_candle/backend/)

### `models`

Model loading and architecture.

**Key Types**:
- `ModelConfig` - Model configuration
- `ModelLoader` - Load models from files
- `Qwen` - Qwen architecture

[View Module](https://docs.rs/metal-candle/latest/metal_candle/models/)

### `training`

LoRA training infrastructure.

**Key Types**:
- `LoRAAdapter` - LoRA adapter
- `LoRAAdapterConfig` - Configuration
- `Trainer` - Training loop
- `TrainingConfig` - Training parameters
- `AdamWConfig` - Optimizer config
- `LRScheduler` - Learning rate schedules

[View Module](https://docs.rs/metal-candle/latest/metal_candle/training/)

### `inference`

Text generation and sampling.

**Key Types**:
- `KVCache` - Key-value cache
- `KVCacheConfig` - Cache configuration
- `SamplingStrategy` - Sampling methods
- `sample_token` - Sample next token

[View Module](https://docs.rs/metal-candle/latest/metal_candle/inference/)

### `embeddings`

Sentence-transformers (feature: `embeddings`).

**Key Types**:
- `EmbeddingModel` - Embedding model
- `EmbeddingModelType` - Model variants
- `EmbeddingConfig` - Configuration

[View Module](https://docs.rs/metal-candle/latest/metal_candle/embeddings/)

### `error`

Error types.

**Key Types**:
- `ModelError` - Model loading errors
- `TrainingError` - Training errors
- `InferenceError` - Inference errors

[View Module](https://docs.rs/metal-candle/latest/metal_candle/error/)

## Quick Reference

### Loading Models

```rust
use metal_candle::models::{ModelConfig, ModelLoader};
use metal_candle::Device;

let device = Device::new_metal(0)?;
let loader = ModelLoader::new(device)
    .with_dtype(DType::F16);
let weights = loader.load("model.safetensors")?;
```

### LoRA Training

```rust
use metal_candle::training::*;

let adapter = LoRAAdapter::new(&model, lora_config, &device)?;
let trainer = Trainer::new(adapter, training_config)?;
let metrics = trainer.train(&dataset)?;
```

### Text Generation

```rust
use metal_candle::inference::*;

let mut cache = KVCache::new(cache_config, &device)?;
let strategy = SamplingStrategy::TopP { p: 0.9 };
let token = sample_token(&logits, &strategy)?;
```

### Embeddings

```rust
use metal_candle::embeddings::*;

let model = EmbeddingModel::from_pretrained(
    EmbeddingModelType::E5SmallV2,
    device,
)?;
let embeddings = model.encode(&texts)?;
```

## See Also

- [User Guide](../guide/devices.md)
- [Examples](https://github.com/GarthDB/metal-candle/tree/main/examples)
