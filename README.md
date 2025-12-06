# metal-candle

[![CI](https://github.com/GarthDB/metal-candle/workflows/CI/badge.svg)](https://github.com/GarthDB/metal-candle/actions)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)

> Production-quality Rust ML crate for Apple Silicon - LoRA training, inference, text generation, and semantic embeddings using Candle with Metal backend

## 🎯 Overview

`metal-candle` is a pure Rust machine learning library designed specifically for Apple Silicon, providing production-ready tools for:

- **🎓 LoRA Training**: Fine-tune transformer models efficiently using Low-Rank Adaptation
- **📦 Model Loading**: Safetensors format with comprehensive validation
- **⚡ Text Generation**: Fast inference with KV-cache and multiple sampling strategies
- **🔍 Semantic Embeddings**: Sentence-transformers (E5, MiniLM, MPNet) for RAG and search
- **🔧 Metal Acceleration**: Native Metal backend for optimal M-series chip performance
- **🏗️ Qwen Support**: Full Qwen2.5-Coder architecture implementation

### Why metal-candle?

- **🚀 Single Binary**: No Python runtime or virtual environments required
- **⚡ Pure Rust**: Type-safe ML with compile-time guarantees
- **🛡️ Production Ready**: 160 tests, zero warnings, 100% API documentation
- **🎨 Ergonomic API**: Builder patterns, sensible defaults, clear error messages
- **📊 Well Tested**: ≥80% code coverage with comprehensive test suites
- **🔧 Easy Integration**: Works seamlessly with any Rust project

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
metal-candle = "1.0"
```

Or use the Git dependency for the latest:

```toml
[dependencies]
metal-candle = { git = "https://github.com/GarthDB/metal-candle", tag = "v1.0.0" }
```

### Requirements

- **Rust** 1.75+ (latest stable recommended)
- **Apple Silicon Mac** (M1/M2/M3/M4)
- **macOS** 12.0+ (for Metal support)

## 🚀 Quick Start

### Loading a Model

```rust
use metal_candle::models::{ModelConfig, ModelLoader};
use metal_candle::Device;
use candle_core::DType;

// Setup device (Metal with CPU fallback)
let device = Device::new_with_fallback(0);

// Load model configuration
let config = ModelConfig::from_file("config.json")?;

// Load model weights
let loader = ModelLoader::new(device)
    .with_dtype(DType::F16);

let weights = loader.load("model.safetensors")?;
```

### LoRA Training

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

// Configure training
let training_config = TrainingConfig {
    num_epochs: 3,
    lr_scheduler: LRScheduler::warmup_cosine(100, 1000, 1e-4, 1e-6),
    optimizer_config: AdamWConfig::default(),
    max_grad_norm: Some(1.0),
};

// Train
let trainer = Trainer::new(adapter, training_config)?;
let metrics = trainer.train(&dataset)?;

// Save checkpoint
save_checkpoint(&trainer.lora_adapter(), "checkpoint.safetensors", None)?;
```

### Text Generation

```rust
use metal_candle::inference::{
    KVCache, KVCacheConfig, SamplingStrategy, sample_token
};

// Setup KV-cache for efficient generation
let cache_config = KVCacheConfig {
    max_seq_len: 2048,
    num_layers: 24,
    num_heads: 14,
    head_dim: 64,
    batch_size: 1,
};

let mut cache = KVCache::new(cache_config, &device)?;

// Generate with different sampling strategies
let strategy = SamplingStrategy::TopP { p: 0.9 };
let token = sample_token(&logits, &strategy)?;

// Or use greedy decoding
let strategy = SamplingStrategy::Greedy;
let token = sample_token(&logits, &strategy)?;
```

### Semantic Embeddings (RAG & Search)

```rust
use candle_core::Device;
use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};

// Load embedding model (auto-downloads from HuggingFace)
let device = Device::Cpu;
let model = EmbeddingModel::from_pretrained(
    EmbeddingModelType::E5SmallV2,
    device,
)?;

// Generate embeddings for semantic search
let texts = vec![
    "Rust is a systems programming language",
    "Python is a high-level language",
];
let embeddings = model.encode(&texts)?;  // [batch, 384]

// Embeddings are L2-normalized for cosine similarity
let vecs = embeddings.to_vec2::<f32>()?;
let similarity: f32 = vecs[0]
    .iter()
    .zip(&vecs[1])
    .map(|(a, b)| a * b)
    .sum();
```

## 📊 Project Status

**Current Phase**: Phase 5 - Quality & Documentation  
**Version**: v1.0.0 🎉  
**Tests**: 160 passing (144 lib + 6 gradient + 10 inference + 43 doctests)  
**Warnings**: Zero ✅  
**Coverage**: 84.69% (exceeds 80% requirement)  
**Focus**: Type safety, ergonomic APIs, and single-binary deployment

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Initial Setup | ✅ Complete |
| Phase 1 | Foundation & Metal Backend | ✅ Complete |
| Phase 2 | Model Loading & Architecture | ✅ Complete |
| Phase 3 | LoRA Training Pipeline | ✅ Complete |
| Phase 4 | Inference & Text Generation | ✅ Complete |
| Phase 5 | Quality & Benchmarking | ✅ Complete |
| Phase 6 | v1.0 Release & Integration | ✅ Complete |

See [PLAN.md](PLAN.md) for detailed roadmap.

## 🏗️ Architecture

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

## ✨ Features

### Training

- ✅ **LoRA Layers**: Low-rank adaptation for efficient fine-tuning
- ✅ **AdamW Optimizer**: With decoupled weight decay
- ✅ **LR Schedulers**: Constant, Linear, Cosine, WarmupCosine
- ✅ **Checkpoint Management**: Save/load LoRA weights with metadata
- ✅ **Gradient Flow**: Full autograd support via Candle's `Var`
- ✅ **Loss Functions**: Cross-entropy with optional label smoothing

### Inference

- ✅ **KV-Cache**: ~173 MB for 2048 tokens (Qwen 0.5B, F16)
- ✅ **Sampling Strategies**: Greedy, Top-k, Top-p, Temperature
- ✅ **Memory Efficient**: O(1) position tracking per token
- ✅ **Fast**: <1% sampling overhead vs forward pass

### Models

- ✅ **Qwen2.5-Coder**: Full architecture implementation
- ✅ **Safetensors**: Primary model format with validation
- ✅ **Transformer Components**: RoPE, Multi-head Attention (GQA), MLP
- ✅ **Model Loading**: Builder pattern with dtype conversion

### Embeddings (feature: `embeddings`)

- ✅ **Sentence Transformers**: E5-small-v2, MiniLM-L6-v2, MPNet-base-v2
- ✅ **HuggingFace Hub**: Auto-download and caching
- ✅ **Mean Pooling**: Attention-weighted token averaging
- ✅ **L2 Normalization**: Ready for cosine similarity
- ✅ **CPU & Metal**: Works on both devices

### Quality

- ✅ **160 Tests**: Comprehensive test coverage
- ✅ **Zero Warnings**: Strict clippy (pedantic level)
- ✅ **100% API Docs**: All public APIs documented with examples
- ✅ **CI/CD**: GitHub Actions on Apple Silicon runners
- ✅ **Type Safe**: Leverages Rust's type system for correctness

## 📚 Documentation

### User Documentation

- **[📖 API Reference](https://docs.rs/metal-candle)** - Complete API documentation (coming soon)
- **[🏗️ Architecture Guide](ARCHITECTURE.md)** - System design and implementation details
- **[🤝 Contributing Guide](CONTRIBUTING.md)** - Development standards and guidelines
- **[⚡ Benchmarks](BENCHMARKS.md)** - Performance metrics and optimization opportunities
- **[📋 Project Plan](PLAN.md)** - 12-week implementation roadmap

### Examples

| Example | Description |
|---------|-------------|
| [`load_model.rs`](examples/load_model.rs) | Model loading and inspection |
| [`forward_pass.rs`](examples/forward_pass.rs) | Qwen model forward pass |
| [`train_lora.rs`](examples/train_lora.rs) | End-to-end LoRA training |
| [`inference_demo.rs`](examples/inference_demo.rs) | KV-cache and sampling demo |
| [`embeddings_demo.rs`](examples/embeddings_demo.rs) | Semantic search with embeddings |

Run examples:
```bash
cargo run --example inference_demo
cargo run --example train_lora
cargo run --example embeddings_demo --features embeddings
```

## 🧪 Development

### Setup

```bash
git clone https://github.com/GarthDB/metal-candle.git
cd metal-candle

# Build
cargo build

# Run tests
cargo test

# Check code quality
cargo clippy -- -D warnings
cargo fmt --check
```

### Testing

```bash
# Run all tests
cargo test

# Run specific test suite
cargo test training
cargo test inference

# Run with output
cargo test -- --nocapture

# Run doctests
cargo test --doc
```

### Coverage

```bash
# Install coverage tool
cargo install cargo-llvm-cov

# Generate HTML report
cargo llvm-cov --all-features --workspace --html
open target/llvm-cov/html/index.html

# Check coverage percentage
cargo llvm-cov --all-features --workspace --summary-only
```

### Benchmarking

```bash
# Run benchmarks (local only)
cargo bench --bench training
cargo bench --bench inference

# Profile with Instruments (macOS)
cargo instruments -t Allocations --release --example train_lora
cargo instruments -t Time --release --example train_lora
cargo instruments -t Metal --release --example train_lora
```

### Local CI Testing

```bash
# Install act
brew install act

# Run CI jobs locally
act -j clippy    # Run clippy check
act -j test      # Run test suite
act -j fmt       # Run format check
```

## 🎯 Quality Standards

This project maintains strict production-quality standards:

| Standard | Requirement | Status |
|----------|-------------|--------|
| **Clippy** | Zero warnings (pedantic) | ✅ Passing |
| **Tests** | All passing | ✅ 160/160 |
| **Coverage** | ≥80% enforced | ✅ Met |
| **Documentation** | 100% public APIs | ✅ Complete |
| **Format** | `rustfmt` compliant | ✅ Passing |

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed coding standards.

## 🚀 Performance & Trade-offs

### Strengths

- **LoRA Overhead**: Minimal (~5-10% vs base model)
- **Memory Efficiency**: Trainable params only (0.1% of model)
- **KV-Cache**: ~173 MB for 2048 tokens (Qwen 0.5B, F16)
- **Type Safety**: Compile-time error catching
- **Zero-Cost Abstractions**: Rust's performance guarantees

### Current Limitations

- **Raw Throughput**: Currently optimized for ergonomics and correctness over raw speed
- **Optimization Opportunities**: Performance improvements planned for v1.1+

See [BENCHMARKS.md](BENCHMARKS.md) for detailed metrics and optimization roadmap.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Code quality standards
- Testing requirements
- Documentation guidelines
- PR process
- Development setup

### Quick Contribution Checklist

- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo test` passes
- [ ] `cargo fmt` applied
- [ ] New code has tests
- [ ] Public APIs documented
- [ ] No `unwrap()` in library code

## 📈 Roadmap

### v1.0 ✅ Complete

- ✅ Phase 1: Foundation & Metal Backend
- ✅ Phase 2: Model Loading & Architecture
- ✅ Phase 3: LoRA Training Pipeline
- ✅ Phase 4: Inference & Text Generation
- ✅ Phase 5: Quality & Documentation
- ✅ Phase 6: v1.0 Release & Integration

### v1.1+ (Future)

- [ ] GGUF format support
- [ ] Additional model architectures (LLaMA, Mistral)
- [ ] Quantization (4-bit, 8-bit)
- [ ] Flash Attention integration
- [ ] Multi-GPU support
- [ ] Streaming generation with callbacks

## 📜 License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).

The Apache License provides explicit patent protection, which is important for production machine learning libraries.

## 🙏 Acknowledgments

- Built on the excellent [Candle](https://github.com/huggingface/candle) framework by Hugging Face
- Inspired by [MLX](https://github.com/ml-explore/mlx) and [llama.cpp](https://github.com/ggerganov/llama.cpp)
- LoRA implementation based on [LoRA paper](https://arxiv.org/abs/2106.09685)

## ⚠️ Known Advisories

This project has two transitive dependencies flagged as unmaintained (not security issues):
- `number_prefix` (via hf-hub → indicatif)
- `paste` (via candle-core → gemm/metal)

These are from major, trusted dependencies (Candle, HuggingFace) and pose no security risk. They will be resolved when upstream updates. See `deny.toml` for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/GarthDB/metal-candle/issues)
- **Discussions**: [GitHub Discussions](https://github.com/GarthDB/metal-candle/discussions)
- **Documentation**: [ARCHITECTURE.md](ARCHITECTURE.md) | [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Status**: ✅ v1.0.0 Released - Production Ready  
**Maintained by**: [@GarthDB](https://github.com/GarthDB)  
**License**: Apache-2.0
