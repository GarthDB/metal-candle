# metal-candle

[![CI](https://github.com/GarthDB/metal-candle/workflows/CI/badge.svg)](https://github.com/GarthDB/metal-candle/actions)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> Production-quality Rust ML crate for Apple Silicon - LoRA training, inference, and text generation using Candle with Metal backend

## 🎯 Project Overview

`metal-candle` is a pure Rust machine learning library designed specifically for Apple Silicon, providing production-ready tools for:

- **LoRA Training**: Fine-tune transformer models efficiently using Low-Rank Adaptation
- **Model Loading**: Support for safetensors (primary) with extensibility for GGUF and other formats
- **Text Generation**: Fast inference with multiple sampling strategies and KV-cache optimization
- **Metal Acceleration**: Native Metal backend for optimal Apple Silicon performance

### Why metal-candle?

- **🚀 Single Binary Deployment**: No Python runtime required
- **⚡ Native Performance**: Direct Rust-to-Metal calls, no PyO3 overhead
- **🛡️ Production Quality**: 80%+ code coverage, zero clippy warnings, comprehensive documentation
- **🎨 Ergonomic API**: Builder patterns, sensible defaults, clear error messages

## 📊 Project Status

**Current Phase**: Initial Setup (Phase 0)  
**Target**: v1.0.0 in 12 weeks  
**Tracking**: [GitHub Issues](https://github.com/GarthDB/metal-candle/issues)

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Initial Setup | 🚧 In Progress |
| Phase 1 | Foundation & Metal Backend | ⏳ Planned |
| Phase 2 | Model Loading & Architecture | ⏳ Planned |
| Phase 3 | LoRA Training Pipeline | ⏳ Planned |
| Phase 4 | Inference & Generation | ⏳ Planned |
| Phase 5 | Quality & Benchmarking | ⏳ Planned |
| Phase 6 | v1.0 Release & Integration | ⏳ Planned |

See [PLAN.md](PLAN.md) for detailed roadmap.

## 🚀 Quick Start

> **Note**: This crate is under active development. The API is not stable yet.

```rust
use metal_candle::{ModelLoader, LoRAAdapter, Trainer, Generator};

// Load a model
let model = ModelLoader::new()
    .with_dtype(DType::F16)
    .load("qwen2.5-coder.safetensors")?;

// Create LoRA adapter for fine-tuning
let lora = LoRAAdapter::builder()
    .rank(8)
    .alpha(16.0)
    .target_modules(&["q_proj", "v_proj"])
    .build(&model)?;

// Train on your data
let config = TrainingConfig::default();
let trainer = Trainer::new(lora, config);
let checkpoint = trainer.train(dataset)?;

// Generate text
let generator = Generator::new(checkpoint)
    .with_temperature(0.7)
    .with_top_p(0.9);

for token in generator.generate("Write a function")? {
    print!("{}", token);
}
```

## 🏗️ Architecture

Built on [Candle](https://github.com/huggingface/candle) with Metal backend, providing:

```
┌─────────────────────────────────────────────────────────┐
│                    metal-candle API                      │
├─────────────────────────────────────────────────────────┤
│  ModelLoader | LoRATrainer | Generator | Checkpoint     │
├─────────────────────────────────────────────────────────┤
│  Models: Qwen2.5-Coder | Generic Transformer            │
├─────────────────────────────────────────────────────────┤
│  Backend: Candle (Metal Device | Tensor Ops)            │
├─────────────────────────────────────────────────────────┤
│  Apple Metal Performance Shaders                         │
└─────────────────────────────────────────────────────────┘
```

## 📦 Installation

> **Coming Soon**: Will be published to [crates.io](https://crates.io) as v1.0.0

For now, use the Git dependency:

```toml
[dependencies]
metal-candle = { git = "https://github.com/GarthDB/metal-candle" }
```

## 🎯 Quality Standards

This project maintains production-quality standards:

- ✅ **Zero Clippy Warnings** (pedantic level)
- ✅ **≥80% Code Coverage** (enforced in CI)
- ✅ **Comprehensive Documentation** (all public APIs)
- ✅ **Performance Benchmarking** (vs MLX+PyO3 baseline)

See [.cursorrules](.cursorrules) for detailed coding standards.

## 🧪 Development

### Prerequisites

- Rust 1.75+ (latest stable recommended)
- Apple Silicon Mac (M1/M2/M3)
- Xcode Command Line Tools

### Setup

```bash
git clone https://github.com/GarthDB/metal-candle.git
cd metal-candle
cargo build
```

### Testing

```bash
# Run all tests
cargo test

# Check for clippy warnings (must pass)
cargo clippy -- -D warnings

# Measure code coverage
cargo llvm-cov --all-features --workspace --html
open target/llvm-cov/html/index.html
```

### Benchmarking

```bash
# Run benchmarks locally on Apple Silicon
cargo bench --bench mlx_comparison

# Profile with Instruments
cargo instruments -t Time --release --example train_lora
```

## 📚 Documentation

- [PLAN.md](PLAN.md) - Detailed 12-week implementation roadmap
- [.cursorrules](.cursorrules) - Coding standards and guidelines
- `ARCHITECTURE.md` - Coming soon
- `BENCHMARKS.md` - Coming soon

## 🤝 Contributing

Contributions are welcome! This project is in active development.

Before contributing:
1. Read [.cursorrules](.cursorrules) for coding standards
2. Check [open issues](https://github.com/GarthDB/metal-candle/issues)
3. Ensure all tests pass and clippy is happy

## 📈 Roadmap

### v1.0 (12 weeks)
- ✅ Core ML operations on Metal
- ✅ Safetensors model loading
- ✅ LoRA training pipeline
- ✅ Text generation with KV-cache
- ✅ Qwen2.5-Coder support

### v1.1+ (Future)
- GGUF format support
- Additional model architectures
- Quantization (below fp16)
- More optimizations

## 📜 License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).

The Apache License provides explicit patent protection, which is important for production machine learning libraries.

## 🙏 Acknowledgments

Built on the excellent [Candle](https://github.com/huggingface/candle) framework by Hugging Face.

---

**Status**: 🚧 Under Active Development  
**Target**: v1.0.0 Release (12 weeks)  
**Maintained by**: [@GarthDB](https://github.com/GarthDB)

