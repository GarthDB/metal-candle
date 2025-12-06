# Introduction

**metal-candle** is a production-quality Rust crate for machine learning on Apple Silicon, built on the [Candle](https://github.com/huggingface/candle) framework with Metal backend acceleration.

## What is metal-candle?

metal-candle provides:

- **🚀 Native Apple Silicon Performance**: Full Metal GPU acceleration
- **🦀 Pure Rust**: No Python dependencies, single-binary deployment
- **🎯 LoRA Training**: Fine-tune transformer models efficiently
- **📦 Model Loading**: Safetensors format (GGUF planned for v1.1+)
- **💬 Text Generation**: Production-ready inference with KV-cache optimization
- **🔍 Semantic Embeddings**: Sentence-transformers for RAG and search
- **✅ Production Quality**: ≥80% coverage, zero clippy warnings, comprehensive docs

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
- **Production quality** - Comprehensive tests, docs, zero warnings
- **Memory safety** - No segfaults, no memory leaks
- **Easy integration** - Works with any Rust project
- **Ergonomic APIs** - Builder patterns, clear error messages

## Target Use Cases

1. **LoRA Fine-tuning**: Efficiently adapt pre-trained models to specific tasks
2. **Model Serving**: Deploy models as native binaries
3. **Embedded ML**: Run ML in resource-constrained environments
4. **Research**: Experiment with transformer architectures in Rust

## Project Status

**Current Version**: v1.0.0 🎉  
**All Phases Complete**: Foundation → LoRA Training → Inference → Quality ✅

See [Roadmap](./development/roadmap.md) for v1.1+ features.

## Getting Started

Ready to dive in? Head to the [Quick Start](./quick-start.md) guide!

## Links

- **API Documentation**: [docs.rs/metal-candle](https://docs.rs/metal-candle)
- **Source Code**: [GitHub](https://github.com/GarthDB/metal-candle)
- **Issue Tracker**: [GitHub Issues](https://github.com/GarthDB/metal-candle/issues)
- **Crate**: [crates.io/crates/metal-candle](https://crates.io/crates/metal-candle)

