# Model Loading

Loading and configuring transformer models in metal-candle.

## Supported Formats

### Safetensors (Primary)

Safetensors is the recommended format:

```rust
use metal_candle::models::{ModelConfig, ModelLoader};
use metal_candle::Device;
use candle_core::DType;

// Load configuration
let config = ModelConfig::from_file("config.json")?;

// Create loader with options
let device = Device::new_metal(0)?;
let loader = ModelLoader::new(device)
    .with_dtype(DType::F16);

// Load weights
let weights = loader.load("model.safetensors")?;
```

### Future Formats (v1.1+)

- **GGUF**: llama.cpp compatibility
- **PyTorch**: Legacy model support

## Supported Models

### Qwen2.5-Coder (v1.0)

Full support for Qwen architecture:

```rust
use metal_candle::models::{Qwen, ModelConfig, ModelLoader};
use candle_nn::VarBuilder;

// Load configuration and weights
let config = ModelConfig::from_file("config.json")?;
let loader = ModelLoader::new(device.clone());
let tensors = loader.load("model.safetensors")?;

// Create model with VarBuilder
let vb = VarBuilder::from_tensors(tensors, candle_core::DType::F16, &device);
let model = Qwen::new(&config, vb)?;

// Forward pass
let output = model.forward(&input_ids, None)?;
```

**Qwen Sizes:**
- qwen2.5-coder-0.5b
- qwen2.5-coder-1.5b  
- qwen2.5-coder-3b
- qwen2.5-coder-7b (requires significant RAM)

### Model Configuration

Configuration from JSON:

```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "vocab_size": 32000,
  "hidden_size": 896,
  "intermediate_size": 4864,
  "num_hidden_layers": 24,
  "num_attention_heads": 14,
  "num_key_value_heads": 2,
  "max_position_embeddings": 32768,
  "rope_theta": 1000000.0
}
```

Load it:

```rust
let config = ModelConfig::from_file("config.json")?;

println!("Model: {:?}", config.architectures);
println!("Layers: {}", config.num_hidden_layers);
println!("Hidden size: {}", config.hidden_size);
```

## Builder Pattern

The `ModelLoader` uses the builder pattern for configuration:

```rust
let device = Device::new_metal(0)?;
let weights = ModelLoader::new(device)  // Device set in constructor
    .with_dtype(DType::F16)             // Weight precision  
    .load("model.safetensors")?;        // Load weights
```

**Options:**
- Constructor takes `Device` (Metal/CPU)
- `.with_dtype(dtype)` - Set weight precision (F16/F32)

## Model Inspection

After loading, inspect the configuration:

```rust
use metal_candle::models::ModelConfig;

let config = ModelConfig::from_file("config.json")?;

// Configuration details
println!("Vocabulary: {}", config.vocab_size);
println!("Layers: {}", config.num_hidden_layers);
println!("Hidden size: {}", config.hidden_size);
```

## Memory Considerations

### Model Size Estimates

| Model | Params | F16 Memory | F32 Memory |
|-------|--------|------------|------------|
| 0.5B | 494M | ~1 GB | ~2 GB |
| 1.5B | 1.5B | ~3 GB | ~6 GB |
| 3B | 3B | ~6 GB | ~12 GB |
| 7B | 7B | ~14 GB | ~28 GB |

Formula: `params * bytes_per_param`
- F16: 2 bytes per parameter
- F32: 4 bytes per parameter

### Tips for Large Models

**Use F16 precision:**
```rust
let device = Device::new_metal(0)?;
let loader = ModelLoader::new(device)
    .with_dtype(DType::F16);  // Half memory vs F32
```

**Monitor memory:**
```bash
# macOS Activity Monitor
# Watch "Memory Pressure" gauge
```

**For 7B+ models:**
- Need 16GB+ unified memory
- Consider model sharding (v2.0 feature)
- Use quantization (planned for v1.1)

## Error Handling

Common loading errors:

```rust
use metal_candle::error::ModelError;

match loader.load("model.safetensors") {
    Ok(weights) => println!("✅ Loaded successfully"),
    Err(ModelError::FileNotFound { path }) => {
        eprintln!("❌ Model file not found: {:?}", path);
    }
    Err(ModelError::InvalidFormat { reason }) => {
        eprintln!("❌ Invalid model format: {}", reason);
    }
    Err(ModelError::IncompatibleVersion { expected, found }) => {
        eprintln!("❌ Version mismatch: expected {}, found {}", expected, found);
    }
    Err(e) => {
        eprintln!("❌ Error: {}", e);
    }
}
```

## Download Models

### From HuggingFace

Models auto-download (embeddings):

```rust
use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};

// Auto-downloads from HuggingFace Hub
let model = EmbeddingModel::from_pretrained(
    EmbeddingModelType::E5SmallV2,
    device,
)?;
```

### Manual Download

For larger models, download manually:

```bash
# Using huggingface-cli
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --include "*.safetensors" "config.json" \
    --local-dir ./models/qwen
```

Then load:
```rust
let config = ModelConfig::from_file("./models/qwen/config.json")?;
let device = Device::new_with_fallback(0);
let loader = ModelLoader::new(device);
let weights = loader.load("./models/qwen/model.safetensors")?;
```

## See Also

- [Device Management](./devices.md) - Choosing GPU vs CPU
- [LoRA Training](./lora.md) - Fine-tuning models
- [Supported Models](../reference/models.md) - Full model list
