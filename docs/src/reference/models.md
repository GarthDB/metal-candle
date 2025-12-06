# Supported Models

Models supported in metal-candle.

## v1.0 Support

### Qwen2.5-Coder

**Architecture**: Qwen2ForCausalLM  
**Format**: Safetensors  
**Status**: ✅ Full Support

**Available Sizes**:

| Model | Parameters | Memory (F16) | Memory (F32) |
|-------|------------|--------------|--------------|
| Qwen2.5-Coder-0.5B | 494M | ~1 GB | ~2 GB |
| Qwen2.5-Coder-1.5B | 1.5B | ~3 GB | ~6 GB |
| Qwen2.5-Coder-3B | 3B | ~6 GB | ~12 GB |
| Qwen2.5-Coder-7B | 7B | ~14 GB | ~28 GB |

**Usage**:
```rust
use metal_candle::models::{Qwen, ModelConfig, ModelLoader};
use candle_nn::VarBuilder;

let config = ModelConfig::from_file("config.json")?;
let loader = ModelLoader::new(device.clone());
let tensors = loader.load("model.safetensors")?;
let vb = VarBuilder::from_tensors(tensors, candle_core::DType::F16, &device);
let model = Qwen::new(&config, vb)?;
```

**Features**:
- ✅ Full forward pass
- ✅ LoRA fine-tuning
- ✅ Text generation
- ✅ KV-cache support

## Embedding Models (feature: `embeddings`)

### E5-small-v2

**Type**: Sentence Transformer  
**Dimension**: 384  
**Use Case**: Semantic search, RAG

```rust
use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};

let model = EmbeddingModel::from_pretrained(
    EmbeddingModelType::E5SmallV2,
    device,
)?;
```

### MiniLM-L6-v2

**Type**: Sentence Transformer  
**Dimension**: 384  
**Use Case**: General-purpose embeddings

```rust
let model = EmbeddingModel::from_pretrained(
    EmbeddingModelType::MiniLML6V2,
    device,
)?;
```

### MPNet-base-v2

**Type**: Sentence Transformer  
**Dimension**: 768  
**Use Case**: High-quality embeddings

```rust
let model = EmbeddingModel::from_pretrained(
    EmbeddingModelType::MPNetBaseV2,
    device,
)?;
```

## Future Support (v1.1+)

### LLaMA 2/3

**Status**: Planned  
**Timeline**: v1.1

### Mistral

**Status**: Planned  
**Timeline**: v1.1

### Additional Architectures

Submit feature requests on [GitHub](https://github.com/GarthDB/metal-candle/issues).

## Model Format Support

### v1.0

- ✅ **Safetensors**: Primary format

### v1.1+

- 🚧 **GGUF**: llama.cpp compatibility
- 🚧 **PyTorch**: Legacy support

## Downloading Models

### HuggingFace Hub (Automatic)

Embedding models auto-download:
```rust
// Auto-downloads and caches
let model = EmbeddingModel::from_pretrained(
    EmbeddingModelType::E5SmallV2,
    device,
)?;
```

### Manual Download

For larger models:
```bash
huggingface-cli download Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --include "*.safetensors" "config.json" \
    --local-dir ./models/qwen
```

## Model Configuration

Example `config.json`:
```json
{
  "architectures": ["Qwen2ForCausalLM"],
  "vocab_size": 32000,
  "hidden_size": 896,
  "intermediate_size": 4864,
  "num_hidden_layers": 24,
  "num_attention_heads": 14,
  "num_key_value_heads": 2,
  "max_position_embeddings": 32768
}
```

Load configuration:
```rust
use metal_candle::models::ModelConfig;

let config = ModelConfig::from_file("config.json")?;
```

## See Also

- [Model Loading Guide](../guide/models.md)
- [LoRA Training](../guide/lora.md)
