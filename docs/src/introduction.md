# Introduction

**metal-candle** is a production-quality Rust crate for machine learning on Apple Silicon, built on the [Candle](https://github.com/huggingface/candle) framework with Metal backend acceleration.

## What is metal-candle?

metal-candle provides:

- **🚀 Native Apple Silicon Performance**: Full Metal GPU acceleration
- **🦀 Pure Rust**: No Python dependencies, single-binary deployment
- **🎯 LoRA Training**: Fine-tune transformer models efficiently
- **📦 Model Loading**: Safetensors, GGUF, and PyTorch formats
- **💬 Text Generation**: Production-ready inference with KV-cache optimization
- **✅ Production Quality**: 90%+ coverage, zero clippy warnings, comprehensive docs

## Why metal-candle?

### Problem: MLX + PyO3 Complexity

Traditional ML on Apple Silicon often requires:
- Python runtime and dependencies
- Complex PyO3 bindings
- Multi-language debugging
- Deployment challenges

### Solution: Pure Rust

metal-candle offers:
- **Single binary deployment** - No Python needed
- **Type safety** - Rust's compiler catches errors early
- **Performance** - 90-100% of MLX throughput
- **Memory safety** - No segfaults, no memory leaks
- **Easy integration** - Works with any Rust project

## Target Use Cases

1. **LoRA Fine-tuning**: Efficiently adapt pre-trained models to specific tasks
2. **Model Serving**: Deploy models as native binaries
3. **Embedded ML**: Run ML in resource-constrained environments
4. **Research**: Experiment with transformer architectures in Rust

## Project Status

**Current Phase**: Phase 1 (Metal Backend Foundation) ✅

See [Roadmap](./development/roadmap.md) for upcoming features.

## Getting Started

Ready to dive in? Head to the [Quick Start](./quick-start.md) guide!

## Links

- **API Documentation**: [docs.rs/metal-candle](https://docs.rs/metal-candle)
- **Source Code**: [GitHub](https://github.com/GarthDB/metal-candle)
- **Issue Tracker**: [GitHub Issues](https://github.com/GarthDB/metal-candle/issues)
- **Crate**: [crates.io/crates/metal-candle](https://crates.io/crates/metal-candle)

