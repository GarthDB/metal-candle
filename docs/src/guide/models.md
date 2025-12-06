# Model Loading

Loading and configuring transformer models in metal-candle.

## Supported Formats

### Safetensors (Primary)

Safetensors is the recommended format:

```rust
use metal_candle::{ModelConfig, ModelLoader};
use candle_core::{Device, DType};

// Load configuration
let config = ModelConfig::from_json("config.json")?;

// Create loader with options
let loader = ModelLoader::new()
    .with_device(Device::new_metal(0)?)
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
use metal_candle::models::Qwen;

// Load Qwen model
let model = Qwen::from_pretrained("qwen2.5-coder-0.5b", &device)?;

// Forward pass
let output = model.forward(&input_ids, 0)?; // position 0
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
let config = ModelConfig::from_json("config.json")?;

println!("Model: {:?}", config.architectures);
println!("Layers: {}", config.num_hidden_layers);
println!("Hidden size: {}", config.hidden_size);
```

## Builder Pattern

The `ModelLoader` uses the builder pattern for configuration:

```rust
let weights = ModelLoader::new()
    .with_device(metal_device)      // Target device
    .with_dtype(DType::F16)          // Weight precision  
    .load("model.safetensors")?;     // Load weights
```

**Options:**
- `.with_device(device)` - Set target device (Metal/CPU)
- `.with_dtype(dtype)` - Set weight precision (F16/F32)

## Model Inspection

After loading, inspect the model:

```rust
use metal_candle::models::Qwen;

let model = Qwen::from_pretrained("qwen", &device)?;

// Model info (via debug print)
println!("Model structure: {:#?}", model);

// Get configuration
let config = model.config();
println!("Vocabulary: {}", config.vocab_size);
println!("Layers: {}", config.num_hidden_layers);
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
let loader = ModelLoader::new()
    .with_dtype(DType::F16)  // Half memory vs F32
    .with_device(metal_device);
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
let config = ModelConfig::from_json("./models/qwen/config.json")?;
let weights = loader.load("./models/qwen/model.safetensors")?;
```

## See Also

- [Device Management](./devices.md) - Choosing GPU vs CPU
- [LoRA Training](./lora.md) - Fine-tuning models
- [Supported Models](../reference/models.md) - Full model list
