# metal-candle

[![Crates.io](https://img.shields.io/crates/v/metal-candle.svg)](https://crates.io/crates/metal-candle)
[![CI](https://github.com/GarthDB/metal-candle/workflows/CI/badge.svg)](https://github.com/GarthDB/metal-candle/actions)
[![codecov](https://codecov.io/gh/GarthDB/metal-candle/branch/main/graph/badge.svg)](https://codecov.io/gh/GarthDB/metal-candle)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Production-quality Rust ML library for Apple Silicon - LoRA training, text generation, and semantic embeddings

## Overview

Pure Rust machine learning library optimized for Apple Silicon:

- **LoRA Training**: Fine-tune transformer models efficiently
- **Text Generation**: Streaming, multiple sampling strategies, repetition penalty
- **Semantic Embeddings**: E5, MiniLM, MPNet models for RAG and search
- **Metal Acceleration**: Native GPU acceleration on M-series chips

**Why metal-candle?** 25.9x faster than MLX for embeddings, single binary deployment, type-safe ML, production-ready (407 tests, 81.6% coverage)

### Performance

metal-candle demonstrates exceptional performance on Apple Silicon:

| Task | Batch Size | metal-candle | MLX | Speedup |
|------|-----------|-------------|-----|---------|
| **Embeddings** | 100 docs | 4.4ms | 113.5ms | **25.9x** 🚀 |
| **Embeddings** | Single query | 3.9ms | 7.7ms | **2.0x** |
| **Throughput** | - | 22,831 docs/sec | 881 docs/sec | **25.9x** |

**Near constant-time performance**: Batch 1→100 only increases by 13% (3.9ms → 4.4ms)

See [BENCHMARKS.md](BENCHMARKS.md) for detailed performance analysis and methodology.

## Installation

```toml
[dependencies]
metal-candle = "1.2"  # or latest from crates.io
```

**Requirements**: Rust 1.75+, Apple Silicon (M1/M2/M3/M4), macOS 12.0+

## Quick Start

### Text Generation

```rust
use metal_candle::inference::{Generator, GeneratorConfig, SamplingStrategy};
use metal_candle::models::Qwen;

// Load model
let model = Qwen::new(&config, vb)?;

// Configure generation
let gen_config = GeneratorConfig {
    max_tokens: 128,
    sampling: SamplingStrategy::TopP { p: 0.95 },
    temperature: 0.7,
    repetition_penalty: 1.1,  // Reduce repetition
    stop_on_eos: true,
    eos_token_id: Some(151643),  // Qwen EOS token
    ..Default::default()
};

// Generate tokens
let mut generator = Generator::new(Box::new(model), gen_config)?;
let output_ids = generator.generate(&input_ids)?;

// Or use streaming for real-time generation
generator.generate_stream(&input_ids, |token| {
    print!("{} ", token);
    true // Continue generation
})?;
```

### Semantic Embeddings (RAG & Search)

```rust
use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};
use metal_candle::Device;

// Load embedding model with Metal acceleration (25.9x faster than MLX!)
let device = Device::new_metal(0)?;
let model = EmbeddingModel::from_pretrained(
    EmbeddingModelType::E5SmallV2,
    device,
)?;

// Generate embeddings for semantic search
let texts = vec![
    "Rust is a systems programming language",
    "Python is a high-level language",
];
let embeddings = model.encode(&texts)?;  // [batch, 384] in 3.9ms

// Batch processing: 100 docs in 4.4ms (22,831 docs/sec throughput)
let large_corpus = load_documents()?;
let batch_embeddings = model.encode(&large_corpus)?;
```

### LoRA Training

```rust
use metal_candle::training::{
    LoRAAdapter, LoRAAdapterConfig, TargetModule,
    Trainer, TrainingConfig, LRScheduler
};

// Create LoRA adapter
let lora_config = LoRAAdapterConfig {
    rank: 8,
    alpha: 16.0,
    dropout: 0.0,
    target_modules: vec![TargetModule::QProj, TargetModule::VProj],
};
let adapter = LoRAAdapter::new(&model, lora_config, &device)?;

// Configure and train
let training_config = TrainingConfig {
    num_epochs: 3,
    lr_scheduler: LRScheduler::warmup_cosine(100, 1000, 1e-4, 1e-6),
    ..Default::default()
};
let trainer = Trainer::new(adapter, training_config)?;
let metrics = trainer.train(&dataset)?;
```

## Features

- **Training**: LoRA with dropout, AdamW optimizer, learning rate schedulers, checkpoint management
- **Inference**: KV-cache, multiple sampling strategies, streaming generation, repetition penalty
- **Models**: Qwen2.5-Coder, safetensors format, transformer components (RoPE, GQA, MLP)
- **Embeddings**: E5, MiniLM, MPNet with HuggingFace Hub integration
- **Quality**: 407 tests, 81.6% coverage, strict clippy linting, 100% API documentation

## Architecture

Built on [Candle](https://github.com/huggingface/candle) with Metal backend:

```
┌─────────────────────────────────────────────────────────────┐
│                    metal-candle (Public API)                 │
├─────────────────────────────────────────────────────────────┤
│  Training          │  Inference        │  Models            │
│  • LoRAAdapter     │  • KVCache        │  • ModelLoader     │
│  • Trainer         │  • Sampling       │  • Qwen           │
│  • AdamW           │  • Generator      │  • Config          │
│  • Schedulers      │                   │                    │
│  • Checkpoint      │  Embeddings       │                    │
│                    │  • EmbeddingModel │                    │
│                    │  • E5/MiniLM/MPNet│                    │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      Candle Framework                        │
│  • Tensor operations  • Metal backend  • Autograd           │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      Apple Metal API                         │
│  (GPU acceleration on Apple Silicon)                        │
└─────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Documentation

- **[API Reference](https://docs.rs/metal-candle)** - Complete API documentation
- **[Architecture Guide](ARCHITECTURE.md)** - System design and implementation details
- **[Contributing Guide](CONTRIBUTING.md)** - Development standards and guidelines
- **[Benchmarks](BENCHMARKS.md)** - Performance analysis and methodology
- **[Project Plan](PLAN.md)** - Development roadmap and future plans

### Examples

| Example | Description |
|---------|-------------|
| [`generate_text.rs`](examples/generate_text.rs) | Text generation with streaming and sampling |
| [`train_lora.rs`](examples/train_lora.rs) | End-to-end LoRA training |
| [`embeddings_demo.rs`](examples/embeddings_demo.rs) | Semantic search with embeddings |
| [`inference_demo.rs`](examples/inference_demo.rs) | KV-cache and sampling demo |
| [`load_model.rs`](examples/load_model.rs) | Model loading and inspection |

Run examples:
```bash
cargo run --example generate_text
cargo run --example train_lora
cargo run --example embeddings_demo --features embeddings
```

## Development

```bash
git clone https://github.com/GarthDB/metal-candle.git
cd metal-candle
cargo build && cargo test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines. Quality standards: zero clippy warnings (pedantic), ≥80% coverage, 100% API docs.

## Roadmap

### Current: v1.2 ✅

Core features complete: LoRA training, text generation, KV-cache, embeddings, custom Metal kernels

### Future: v1.3+

- [ ] GGUF format support
- [ ] Additional model architectures (LLaMA, Mistral)
- [ ] Quantization (4-bit, 8-bit)
- [ ] Flash Attention integration
- [ ] Multi-GPU support

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development standards and testing requirements.

## License

Licensed under Apache-2.0 ([LICENSE](LICENSE)). Provides explicit patent protection for production ML.

## Acknowledgments

- Built on the excellent [Candle](https://github.com/huggingface/candle) framework by Hugging Face
- Inspired by [MLX](https://github.com/ml-explore/mlx) and [llama.cpp](https://github.com/ggerganov/llama.cpp)
- LoRA implementation based on [LoRA paper](https://arxiv.org/abs/2106.09685)

## Known Advisories

Two unmaintained transitive dependencies (non-security): `number_prefix`, `paste` from trusted upstream (Candle, HF). See `deny.toml` for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/GarthDB/metal-candle/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GarthDB/metal-candle/discussions)
- **Documentation**: [ARCHITECTURE.md](ARCHITECTURE.md) | [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Maintained by**: [@GarthDB](https://github.com/GarthDB)
